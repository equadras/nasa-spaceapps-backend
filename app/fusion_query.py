"""
Fusion Query Engine - Fuse Hybrid (vector+BM25) results with Neo4j Knowledge Graph hits.
"""

from typing import Dict, List, Tuple
import math
import numpy as np
from sentence_transformers import SentenceTransformer

# Import your hybrid engine and the Neo4j loader
from hybrid_query import HybridPaperQueryEngine
from ..my_neo4j.connect import load_existing_graphindex

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _normalize_dict_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize scores to range [0,1]. If all equal, give them same normalized value."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    minv, maxv = float(vals.min()), float(vals.max())
    if math.isclose(maxv, minv):
        # all same -> return 1.0 for all (or 0.0 if zero)
        if maxv == 0.0:
            return {k: 0.0 for k in scores}
        else:
            return {k: 1.0 for k in scores}
    return {k: float((v - minv) / (maxv - minv)) for k, v in scores.items()}


class FusionQueryEngine:
    """
    Fusion engine combining HybridPaperQueryEngine (vector+BM25) and the Neo4j KG.

    Usage:
        fusion = FusionQueryEngine(beta=0.35)  # beta = weight for KG (0..1)
        resp = fusion.query("my question", top_k=20)
        fusion.display_results(resp)
    """

    def __init__(
        self,
        alpha=0.4,
        similarity_threshold=0.3,
        beta=0.25,
        sentence_model_name="sentence-transformers/all-mpnet-base-v2",
    ):
        """
        Args:
            alpha: forwarded to HybridPaperQueryEngine initialization (vector weight).
            similarity_threshold: forwarded to HybridPaperQueryEngine
            beta: weight for KnowledgeGraph scores when fusing (0 = ignore KG, 1 = only KG)
            sentence_model_name: model used to embed the query for KG embedding comparisons
        """
        print("Initializing FusionQueryEngine...")
        self.hybrid_engine = HybridPaperQueryEngine(alpha=alpha, similarity_threshold=similarity_threshold)
        self.beta = float(beta)
        self.sentence_model = SentenceTransformer(sentence_model_name)

        # Load KG index and graph store (Neo4j driver)
        self.kg_index, self.graph_store = load_existing_graphindex()
        # property names on nodes (change these if your Neo4j nodes use other names)
        self.node_text_property = "text"  # fallback: where node's textual content lives
        self.node_title_property = "title"
        self.node_embedding_property = "embedding"  # if your nodes store embeddings as list/array

        print(f"Fusion ready. Beta (KG weight) = {self.beta:.2f}")

    def _kg_candidates_fulltext(self, query: str, top_k: int = 50) -> Dict[str, Tuple[float, dict]]:
        """
        Try to fetch candidate KG nodes using a Neo4j fulltext index named 'kg_fulltext'.
        Returns dict[node_id_str] = (score, metadata_dict)
        """
        results = {}
        driver = self.graph_store._driver
        # change 'kg_fulltext' to whatever index you use in Neo4j
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $q) YIELD node, score
        RETURN id(node) AS nid, score AS score, node AS properties
        LIMIT $limit
        """
        try:
            with driver.session() as session:
                records = session.run(cypher, index_name="kg_fulltext", q=query, limit=top_k)
                for rec in records:
                    nid = str(rec["nid"])
                    score = float(rec["score"])
                    props = dict(rec["properties"])
                    # convert neo4j types if necessary
                    results[nid] = (score, props)
        except Exception as e:
            # fulltext index not present or query failed
            # caller will handle fallback
            results = {}
        return results

    def _kg_candidates_by_embedding(self, query: str, top_k: int = 200) -> Dict[str, Tuple[float, dict]]:
        """
        Fallback: if nodes have stored embeddings, compute cosine similarity of query embedding
        against node embeddings stored as a property named self.node_embedding_property.
        Returns dict[node_id_str] = (similarity, metadata_dict)
        """
        results = {}
        driver = self.graph_store._driver
        q_emb = self.sentence_model.encode(query).tolist()

        # Query nodes that have an embedding property set. This Cypher assumes embeddings stored as a list property.
        cypher = f"""
        MATCH (n)
        WHERE exists(n.{self.node_embedding_property})
        RETURN id(n) AS nid, n.{self.node_embedding_property} AS emb, properties(n) AS properties
        """
        try:
            with driver.session() as session:
                records = session.run(cypher)
                for rec in records:
                    nid = str(rec["nid"])
                    emb = rec["emb"]
                    # sometimes Neo4j returns a string or other type; ensure it's a list of floats
                    try:
                        node_emb = list(map(float, emb))
                    except Exception:
                        # skip malformed embeddings
                        continue
                    sim = _cosine_similarity(q_emb, node_emb)
                    props = dict(rec["properties"])
                    results[nid] = (sim, props)
        except Exception as e:
            # If something goes wrong, return empty, caller will fall back to text containment
            results = {}
        # keep only top_k by similarity
        if results:
            sorted_items = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
            return {k: v for k, v in sorted_items}
        return {}

    def _kg_candidates_text_containment(self, query: str, top_k: int = 200) -> Dict[str, Tuple[float, dict]]:
        """
        Last-resort fallback: scan nodes and score by simple token overlap proportion.
        (Slow for very large graphs - but safe fallback.)
        """
        results = {}
        q_tokens = set(query.lower().split())
        driver = self.graph_store._driver
        cypher = f"""
        MATCH (n)
        RETURN id(n) AS nid, n.{self.node_text_property} AS text, properties(n) AS properties
        """
        try:
            with driver.session() as session:
                records = session.run(cypher)
                for rec in records:
                    nid = str(rec["nid"])
                    text = rec["text"] or ""
                    tokens = set(str(text).lower().split())
                    overlap = len(q_tokens.intersection(tokens))
                    score = overlap / max(1, len(q_tokens))
                    if score > 0:
                        props = dict(rec["properties"])
                        results[nid] = (score, props)
        except Exception:
            results = {}
        if results:
            sorted_items = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
            return {k: v for k, v in sorted_items}
        return {}

    def _get_kg_candidates(self, query: str, top_k: int = 50) -> Dict[str, Tuple[float, dict]]:
        """
        Get candidates from KG using (in order): fulltext -> embeddings -> containment.
        Returns dict[node_id_str] = (raw_score, props)
        """
        # Try fulltext search first
        candidates = self._kg_candidates_fulltext(query, top_k=top_k)
        if candidates:
            return candidates

        # Try embedding-based retrieval
        candidates = self._kg_candidates_by_embedding(query, top_k=top_k)
        if candidates:
            return candidates

        # Fallback text containment
        candidates = self._kg_candidates_text_containment(query, top_k=top_k)
        return candidates

    def query(self, question: str, top_k: int = 20) -> dict:
        """
        Run hybrid vector+BM25 search, get KG candidates, fuse scores and return merged results.
        Returned structure mirrors your HybridPaperQueryEngine.query format:
        { 'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]] }
        Distances = 1 - final_score (so higher score -> smaller distance).
        """

        # 1) Get hybrid (vector+BM25) results
        hybrid_res = self.hybrid_engine.query(question, top_k=top_k * 2)  # ask for more to allow fusion/re-rank
        # hybrid_res format: ids/documents/metadatas/distances (all inside a single list)
        hybrid_ids = hybrid_res['ids'][0]
        hybrid_scores = {}
        # distances were returned as 1 - score in your engine
        for doc_id, dist in zip(hybrid_res['ids'][0], hybrid_res['distances'][0]):
            hybrid_scores[doc_id] = 1.0 - dist

        # 2) Get KG candidates (raw scores)
        kg_candidates = self._get_kg_candidates(question, top_k=200)
        kg_raw_scores = {nid: s for nid, (s, props) in kg_candidates.items()}

        # 3) If KG embedding-based or fulltext returned results, normalize both score maps to [0,1]
        hybrid_norm = _normalize_dict_scores(hybrid_scores)
        kg_norm = _normalize_dict_scores(kg_raw_scores)

        # 4) Merge doc-level results: we map KG node results to paper-level if possible
        # For simplicity, treat KG nodes as documents keyed by node_id (they may or may not map to paper_ids).
        merged_scores = {}

        # Add hybrid candidates
        for doc_id, s in hybrid_norm.items():
            merged_scores[doc_id] = (1 - self.beta) * s  # weighted by (1 - beta)

        # Add KG candidates (we give them distinct ids prefixed to avoid collision with chunk ids)
        for nid, s in kg_norm.items():
            # optionally try to map KG node -> paper_id if the node metadata contains 'paper_id'
            props = kg_candidates[nid][1]
            paper_id = props.get("paper_id") or props.get("paperId") or None
            if paper_id:
                key = f"paper::{paper_id}"
            else:
                key = f"kgnode::{nid}"

            existing = merged_scores.get(key, 0.0)
            merged_scores[key] = max(existing, self.beta * s)  # take max contribution from KG for this paper/node

        # 5) Sort merged results and select top_k
        sorted_merged = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 6) Build result payload similar to HybridPaperQueryEngine.query
        final_ids = []
        final_documents = []
        final_metadatas = []
        final_distances = []

        for key, score in sorted_merged:
            final_ids.append(key)
            # If key originates from hybrid chunk (no 'paper::' prefix), try to fetch its doc & metadata from hybrid engine
            if key in self.hybrid_engine.doc_map:
                idx = self.hybrid_engine.doc_map[key]
                final_documents.append(self.hybrid_engine.all_documents[idx])
                final_metadatas.append(self.hybrid_engine.all_metadatas[idx])
            elif key.startswith("paper::"):
                # try to find a representative chunk for that paper from hybrid data
                paper_id = key.split("::", 1)[1]
                # find first chunk with that paper_id (if any)
                found = False
                for didx, md in enumerate(self.hybrid_engine.all_metadatas):
                    if md.get("paper_id") == paper_id:
                        final_documents.append(self.hybrid_engine.all_documents[didx])
                        final_metadatas.append(md)
                        found = True
                        break
                if not found:
                    # fallback minimal metadata using KG node props if available
                    # find any KG node properties for this paper
                    meta = None
                    for nid, (_, props) in kg_candidates.items():
                        if props.get("paper_id") == paper_id:
                            meta = props
                            break
                    final_documents.append(meta.get(self.node_text_property, "") if meta else "")
                    final_metadatas.append(meta or {"paper_id": paper_id})
            else:
                # kgnode::<nid> -> use KG node props
                nid = key.split("::", 1)[1]
                props = kg_candidates.get(nid, (0.0, {}))[1]
                final_documents.append(props.get(self.node_text_property, ""))
                final_metadatas.append(props)

            final_distances.append(1.0 - float(score))  # distances = 1 - score

        # Keep same nested structure as Hybrid engine
        return {
            "ids": [final_ids],
            "documents": [final_documents],
            "metadatas": [final_metadatas],
            "distances": [final_distances],
        }

    def display_results(self, results):
        """Pretty-print fused results (similar to HybridPaperQueryEngine.display_results)."""
        print("\n" + "=" * 80)
        if not results['ids'][0]:
            print("No fused results.")
            return

        for i, (doc_id, meta, dist) in enumerate(zip(results['ids'][0], results['metadatas'][0], results['distances'][0]), 1):
            score = 1.0 - dist
            print(f"{i}. id: {doc_id} | score: {score:.4f}")
            title = meta.get("title") or meta.get("paper_title") or meta.get("paper_id") or ""
            if title:
                print(f"   {title}")
            if meta.get("authors"):
                print(f"   Authors: {meta['authors']}")
            if meta.get("journal"):
                print(f"   Journal: {meta['journal']}")
            snippet = (meta.get("abstract") or meta.get("text") or "")[:300]
            if snippet:
                print(f"   {snippet}...")
            print("-" * 80)

