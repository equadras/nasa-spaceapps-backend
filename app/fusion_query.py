from typing import Dict, List, Tuple
import math
import numpy as np
from sentence_transformers import SentenceTransformer

# IMPORTS: -m => relativo para módulos dentro de app/, absoluto para pacotes no topo
from .hybrid_query import HybridPaperQueryEngine
from my_neo4j.connect import load_graphindex

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_dict_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    minv, maxv = float(vals.min()), float(vals.max())
    if math.isclose(maxv, minv):
        return {k: (1.0 if maxv != 0.0 else 0.0) for k in scores}
    return {k: float((v - minv) / (maxv - minv)) for k, v in scores.items()}


class FusionQueryEngine:
    """
    Fusion engine combining HybridPaperQueryEngine (vector+BM25) and the Neo4j KG.

    Commands/UX compatível com HybridPaperQueryEngine:
      - show_help()
      - change_threshold()
      - change_weights()  (alfa vetor/BM25)
      - change_beta()     (peso do KG)
      - show_stats()
      - display_results()
    """

    def __init__(
        self,
        alpha=0.4,
        similarity_threshold=0.3,
        beta=0.25,
        sentence_model_name="sentence-transformers/all-mpnet-base-v2",
    ):
        print("Initializing FusionQueryEngine...")

        # motor híbrido (mantém mesmos métodos/estado)
        self.hybrid_engine = HybridPaperQueryEngine(
            alpha=alpha,
            similarity_threshold=similarity_threshold,
        )

        # peso do KG
        self.beta = float(beta)

        # encoder para fallback por embedding no KG
        self.sentence_model = SentenceTransformer(sentence_model_name)

        # carrega índice/driver do Neo4j
        self.kg_index, self.graph_store = load_graphindex()

        # propriedades padrão dos nós
        self.node_text_property = "text"
        self.node_title_property = "title"
        self.node_embedding_property = "embedding"

        print(f"Fusion pronto. KG={self.beta*100:.0f}% | Hybrid={((1-self.beta)*100):.0f}%\n")

    # ---------- Propriedades para espelhar a UX do híbrido ----------
    @property
    def alpha(self) -> float:
        return self.hybrid_engine.alpha

    @alpha.setter
    def alpha(self, v: float):
        self.hybrid_engine.alpha = float(v)

    @property
    def similarity_threshold(self) -> float:
        return self.hybrid_engine.similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, v: float):
        self.hybrid_engine.similarity_threshold = float(v)

    # ---------- Retrieval no KG ----------
    def _kg_candidates_fulltext(self, query: str, top_k: int = 50) -> Dict[str, Tuple[float, dict]]:
        results = {}
        driver = self.graph_store._driver
        cypher = """
        CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
        RETURN id(node) AS nid, score AS score, properties(node) AS properties
        LIMIT $limit
        """
        try:
            with driver.session() as session:
                records = session.run(cypher, index="kg_fulltext", q=query, limit=top_k)
                for rec in records:
                    nid = str(rec["nid"])
                    results[nid] = (float(rec["score"]), dict(rec["properties"]))
        except Exception as e:
            # índice pode não existir; fallback acontecerá
            print(f"[KG FULLTEXT] fallback: {e}")
            return {}
        return results

    def _kg_candidates_by_embedding(self, query: str, top_k: int = 200) -> Dict[str, Tuple[float, dict]]:
        results = {}
        driver = self.graph_store._driver
        q_emb = self.sentence_model.encode(query).tolist()
        cypher = f"""
        MATCH (n:Paper)
        WHERE n.{self.node_embedding_property} IS NOT NULL
        RETURN id(n) AS nid, n.{self.node_embedding_property} AS emb, properties(n) AS properties
        """
        try:
            with driver.session() as session:
                for rec in session.run(cypher):
                    nid = str(rec["nid"])
                    emb = rec["emb"]
                    try:
                        node_emb = list(map(float, emb))
                    except Exception as e:
                        print(f"[KG NODE EMB] fallback: {e}")
                        continue
                    sim = _cosine_similarity(q_emb, node_emb)
                    results[nid] = (sim, dict(rec["properties"]))
        except Exception as e:
            print(f"[KG FULLTEXT] fallback: {e}")
            return {}
        if not results:
            return {}
        return dict(sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:top_k])

    def _kg_candidates_text_containment(self, query: str, top_k: int = 200) -> Dict[str, Tuple[float, dict]]:
        results = {}
        q_tokens = set(query.lower().split())
        driver = self.graph_store._driver
        cypher = f"""
        MATCH (n:Paper)
        RETURN id(n) AS nid, n.search AS text, properties(n) AS properties
        """
        try:
            with driver.session() as session:
                for rec in session.run(cypher):
                    nid = str(rec["nid"])
                    text = rec["text"] or ""
                    tokens = set(str(text).lower().split())
                    overlap = len(q_tokens & tokens)
                    score = overlap / max(1, len(q_tokens))
                    if score > 0:
                        results[nid] = (score, dict(rec["properties"]))
        except Exception as e:
            print(f"[KG TEXT CONTAINMENT] fallback: {e}")
            return {}
        if not results:
            print(f"[KG TEXT CONTAINMENT NO RESULTS]")
            return {}
        return dict(sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:top_k])

    def _get_kg_candidates(self, query: str, top_k: int = 50) -> Dict[str, Tuple[float, dict]]:
        return (
            self._kg_candidates_fulltext(query, top_k)
            or self._kg_candidates_by_embedding(query, top_k)
            or self._kg_candidates_text_containment(query, top_k)
        )

    # ---------- Consulta + fusão ----------
    def query(self, question: str, top_k: int = 20) -> dict:
        # 1) híbrido (pega mais para re-rankeamento)
        hybrid_res = self.hybrid_engine.query(question, top_k=top_k * 2)
        hybrid_ids = hybrid_res["ids"][0]
        hybrid_scores = {doc_id: 1.0 - dist for doc_id, dist in zip(hybrid_ids, hybrid_res["distances"][0])}

        # 2) KG candidatos
        kg_candidates = self._get_kg_candidates(question, top_k=200)
        kg_raw_scores = {nid: s for nid, (s, _) in kg_candidates.items()}

        # 3) normalização para [0,1]
        hybrid_norm = _normalize_dict_scores(hybrid_scores)
        kg_norm = _normalize_dict_scores(kg_raw_scores)

        # 4) merge (documento/chunk + nó/paper)
        merged_scores: Dict[str, float] = {}

        for doc_id, s in hybrid_norm.items():
            merged_scores[doc_id] = (1 - self.beta) * s

        for nid, s in kg_norm.items():
            props = kg_candidates[nid][1]
            paper_id = props.get("paper_id") or props.get("paperId")
            key = f"paper::{paper_id}" if paper_id else f"kgnode::{nid}"
            merged_scores[key] = max(merged_scores.get(key, 0.0), self.beta * s)

        # 5) aplica o MESMO limiar do híbrido (coerência UX)
        if self.similarity_threshold is not None:
            merged_scores = {k: v for k, v in merged_scores.items() if v >= self.similarity_threshold}

        sorted_merged = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_ids, final_documents, final_metadatas, final_distances = [], [], [], []
        for key, score in sorted_merged:
            final_ids.append(key)
            if key in self.hybrid_engine.doc_map:
                idx = self.hybrid_engine.doc_map[key]
                final_documents.append(self.hybrid_engine.all_documents[idx])
                final_metadatas.append(self.hybrid_engine.all_metadatas[idx])
            elif key.startswith("paper::"):
                paper_id = key.split("::", 1)[1]
                found = False
                for didx, md in enumerate(self.hybrid_engine.all_metadatas):
                    if md.get("paper_id") == paper_id:
                        final_documents.append(self.hybrid_engine.all_documents[didx])
                        final_metadatas.append(md)
                        found = True
                        break
                if not found:
                    meta = None
                    for _, (_, props) in kg_candidates.items():
                        if props.get("paper_id") == paper_id:
                            meta = props
                            break
                    final_documents.append((meta or {}).get(self.node_text_property, ""))
                    final_metadatas.append(meta or {"paper_id": paper_id})
            else:
                nid = key.split("::", 1)[1]
                props = kg_candidates.get(nid, (0.0, {}))[1]
                final_documents.append(props.get(self.node_text_property, ""))
                final_metadatas.append(props)

            final_distances.append(1.0 - float(score))

        return {
            "ids": [final_ids],
            "documents": [final_documents],
            "metadatas": [final_metadatas],
            "distances": [final_distances],
        }

    # ---------- UI/UX (mesmos comandos do híbrido) ----------
    def display_results(self, question: str, results: dict):
        print(f"\n{'=' * 80}")
        print(f"Query: {question}")
        print(f"Threshold: {self.similarity_threshold:.2f} | "
              f"Hybrid α={self.alpha*100:.0f}% vec | KG β={self.beta*100:.0f}%")
        print('=' * 80)

        ids = results["ids"][0]
        if not ids:
            print("\nNo fused results above threshold.")
            print(f"Try lowering the threshold (current: {self.similarity_threshold:.2f})")
            return

        # agrupar por paper_id
        papers = {}
        for key, meta, dist in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
            score = 1.0 - dist
            paper_id = None
            if key in self.hybrid_engine.doc_map:
                paper_id = meta.get("paper_id")
            elif key.startswith("paper::"):
                paper_id = key.split("::", 1)[1]
            else:  # kgnode
                paper_id = meta.get("paper_id")

            pid = paper_id or key  # se não houver, agrupa pelo próprio id
            info = papers.setdefault(pid, {"best_score": score, "chunk_count": 0, "meta": meta})
            info["chunk_count"] += 1
            if score > info["best_score"]:
                info["best_score"] = score
                info["meta"] = meta

        ranked = sorted(papers.items(), key=lambda x: x[1]["best_score"], reverse=True)

        print(f"\nTop {min(10, len(ranked))} papers/nodes (above threshold):\n")
        for i, (pid, data) in enumerate(ranked[:10], 1):
            meta = data["meta"]
            title = meta.get("title") or meta.get("paper_title") or meta.get("paper_id") or pid
            print(f"{i}. {title}")
            print(f"   Relevance: {data['best_score']:.4f} ({data['chunk_count']} chunk{'s' if data['chunk_count'] > 1 else ''})")
            if meta.get("authors"):
                print(f"   Authors: {str(meta['authors'])[:80]}")
            if meta.get("year"):
                print(f"   Year: {meta['year']}")
            if meta.get("journal"):
                print(f"   Journal: {str(meta['journal'])[:60]}")
            if meta.get("pmc_link"):
                print(f"   Link: {meta['pmc_link']}")
            snippet = (meta.get("abstract") or meta.get("text") or "")[:300]
            if snippet:
                print(f"   {snippet}...")
            print()

        print("-" * 80)

    def change_threshold(self):
        print(f"\nCurrent threshold: {self.similarity_threshold:.2f}")
        try:
            new_threshold = float(input("\nEnter new threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                self.similarity_threshold = new_threshold
                print(f"Updated threshold to: {self.similarity_threshold:.2f}")
            else:
                print("Invalid. Must be 0.0-1.0")
        except ValueError:
            print("Invalid input")

    def change_weights(self):
        print(f"\nCurrent: Hybrid α={self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25 | KG β={self.beta*100:.0f}%")
        try:
            new_alpha = float(input("\nEnter vector weight α (0-100): ")) / 100.0
            if 0.0 <= new_alpha <= 1.0:
                self.alpha = new_alpha
                print(f"Updated α: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
            else:
                print("Invalid α. Must be 0-100")
        except ValueError:
            print("Invalid input for α")

        try:
            new_beta = input("\nEnter KG weight β (0-100, blank to keep): ").strip()
            if new_beta != "":
                new_beta = float(new_beta) / 100.0
                if 0.0 <= new_beta <= 1.0:
                    self.beta = new_beta
                    print(f"Updated β (KG): {self.beta*100:.0f}%")
                else:
                    print("Invalid β. Must be 0-100")
        except ValueError:
            print("Invalid input for β")

    def change_beta(self):
        print(f"\nCurrent KG β: {self.beta*100:.0f}%")
        try:
            new_beta = float(input("\nEnter KG weight β (0-100): ")) / 100.0
            if 0.0 <= new_beta <= 1.0:
                self.beta = new_beta
                print(f"Updated β (KG): {self.beta*100:.0f}%")
            else:
                print("Invalid. Must be 0-100")
        except ValueError:
            print("Invalid input")

    def show_help(self):
        print("\n" + "=" * 80)
        print("FUSION SEARCH (Hybrid vector+BM25 + Knowledge Graph)")
        print("=" * 80)
        print("\nJust type your question to search")
        print("\nSpecial commands:")
        print("  help       - Show this help")
        print("  threshold  - Change minimum similarity threshold")
        print("  weight     - Change vector/BM25 α and KG β weights")
        print("  stats      - Show statistics (DB + KG)")
        print("  quit       - Exit")
        print(f"\nCurrent settings:\n  Threshold: {self.similarity_threshold:.2f}\n"
              f"  Hybrid α: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25\n"
              f"  KG β: {self.beta*100:.0f}%")
        print("=" * 80)

    def show_stats(self):
        print("\n" + "=" * 80)
        print("DATABASE & GRAPH STATISTICS")
        print("=" * 80)
        # stats do híbrido (chroma)
        try:
            sample = self.hybrid_engine.collection.peek(limit=100)
            unique_papers = set()
            years = []
            for meta in sample["metadatas"]:
                if meta.get("paper_id"):
                    unique_papers.add(meta["paper_id"])
                if meta.get("year"):
                    try:
                        years.append(int(meta["year"]))
                    except Exception:
                        pass
            print(f"\nChroma:")
            print(f"  Total chunks: {self.hybrid_engine.total_chunks}")
            print(f"  Unique papers (sample): ~{len(unique_papers)}")
            if years:
                print(f"  Year range (sample): {min(years)}-{max(years)}")
        except Exception as e:
            print(f"\nChroma stats error: {e}")

        # stats do KG (neo4j)
        try:
            with self.graph_store._driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
                # Neo4j 5: checa FULLTEXT via SHOW INDEXES
                has_fulltext = session.run(f"""
                  SHOW INDEXES
                  YIELD name, type
                  RETURN any(x IN collect({name:name, type:type})
                             WHERE x.name='kg_fulltext' AND toUpper(x.type)='FULLTEXT') AS ok
                """).single()["ok"]
            print(f"\nNeo4j:")
            print(f"  Nodes: {node_count}")
            print(f"  Relationships: {edge_count}")
            print(f"  Fulltext index 'kg_fulltext': {'YES' if has_fulltext else 'NO'}")
        except Exception as e:
            print(f"\nNeo4j stats error: {e}")

        print(f"\nCurrent settings:")
        print(f"  Threshold: {self.similarity_threshold:.2f}")
        print(f"  Hybrid α: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
        print(f"  KG β: {self.beta*100:.0f}%")
        print("=" * 80)


def main():
    print("=" * 80)
    print("NASA BIOSCIENCE - FUSION SEARCH (Hybrid + KG)")
    print("=" * 80)
    print()

    engine = FusionQueryEngine(alpha=0.6, similarity_threshold=0.3, beta=0)

    print("Type 'help' for commands\n")
    while True:
        try:
            question = input("Your question: ").strip()
            if not question:
                continue

            cmd = question.lower()
            if cmd in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break
            if cmd in ("help", "?"):
                engine.show_help()
                continue
            if cmd == "threshold":
                engine.change_threshold()
                continue
            if cmd == "weight":
                engine.change_weights()
                continue
            if cmd in ("stats", "status"):
                engine.show_stats()
                continue

            print("\nSearching...")
            results = engine.query(question, top_k=20)
            engine.display_results(question, results)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            import traceback
            print(f"\nERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()


