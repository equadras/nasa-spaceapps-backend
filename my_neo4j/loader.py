import json
import os
import traceback
import time
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    KnowledgeGraphIndex,
    PromptTemplate,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from rank_bm25 import BM25Okapi
from pyvis.network import Network
from dotenv import load_dotenv
load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

import hashlib
from typing import List, Tuple

def ensure_schema(graph_store):
    # Neo4j 5 schema DDL (not procedures)
    with graph_store._driver.session() as session:
        # 1) Unique ids
        session.run("""
        CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
        FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE
        """)
        session.run("""
        CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)
        session.run("""
        CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS
        FOR (a:Assertion) REQUIRE a.assertion_id IS UNIQUE
        """)
        # 2) Full-text index on Paper.search
        session.run("""
        CREATE FULLTEXT INDEX kg_fulltext IF NOT EXISTS
        FOR (p:Paper) ON EACH [p.search]
        """)
        # (Optional) vector index if you add p.embedding later:
        # session.run("""
        # CREATE VECTOR INDEX kg_vector IF NOT EXISTS
        # FOR (p:Paper) ON (p.embedding)
        # OPTIONS {indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' }}
        # """)

        # 3) Quick check
        rec = session.run("""
        SHOW INDEXES
        YIELD name, type, state, entityType, labelsOrTypes, properties
        WHERE name = 'kg_fulltext'
        RETURN name, type, state, labelsOrTypes, properties
        """).single()
        if rec is None:
            raise RuntimeError("FULLTEXT index 'kg_fulltext' was not created.")
    print("Schema ensured (constraints + FULLTEXT index).")

def _assertion_id(paper_id: str, s: str, r: str, o: str) -> str:
    # deterministic id per (paper_id, s, r, o)
    raw = f"{paper_id}||{s}||{r}||{o}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def extract_triplets_with_llm(llm, text: str, max_triplets: int = 50) -> List[Tuple[str, str, str]]:
    """
    Use your existing LLM (Gemini) with the same prompt format you already set
    to extract (subject, relation, object) lines and parse them.
    """
    prompt = (
        "Some text is provided below. Given the text, extract up to "
        f"{max_triplets} directed knowledge triplets in the form of (subject, relation, object). "
        "Avoid stopwords.\n"
        "---------------------\n"
        f"Text: {text}\n"
        "---------------------\n"
        "Triplets:\n"
    )
    # Call the model via LlamaIndex LLM wrapper
    resp = llm.complete(prompt)
    lines = str(resp).strip().splitlines()
    triplets = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # Expect format: (subject, relation, object)
        if ln.startswith("(") and ln.endswith(")") and "," in ln:
            try:
                body = ln[1:-1]
                parts = [p.strip() for p in body.split(",", 2)]
                if len(parts) == 3:
                    s, r, o = parts
                    if s and r and o:
                        triplets.append((s, r, o))
            except Exception:
                continue
    return triplets

def upsert_paper_node(graph_store, md: dict, fallback_text: str = ""):
    # Minimal paper metadata + search text
    paper_id = md.get("paper_id") or md.get("id")
    title = md.get("title", "")
    abstract = md.get("abstract", "")
    keywords = md.get("keywords", "")
    if not abstract and fallback_text:
        abstract = fallback_text[:4000]
    cypher = """
    MERGE (p:Paper {paper_id: $paper_id})
    SET p.title    = $title,
        p.abstract = $abstract,
        p.keywords = $keywords,
        p.journal  = $journal,
        p.year     = $year,
        p.pmc_link = $pmc_link,
        p.search   = coalesce($title,'') + ' ' + coalesce($abstract,'') + ' ' + coalesce($keywords,'')
    """
    params = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "journal": md.get("journal", ""),
        "year": md.get("year", ""),
        "pmc_link": md.get("pmc_link", ""),
    }
    with graph_store._driver.session() as session:
        session.run(cypher, **params)
    return paper_id

def upsert_assertions(graph_store, paper_id: str, triplets: List[Tuple[str, str, str]], snippet: str = ""):
    """
    Upsert Entities, Assertion (with paper_id) and provenance edges:
      (s:Entity)-[:SUBJECT_OF]->(a:Assertion {assertion_id, paper_id, predicate, snippet?})
      (o:Entity)-[:OBJECT_OF]->(a)
      (a)-[:ASSERTED_IN]->(p:Paper {paper_id})
    """
    rows = []
    for (s, r, o) in triplets:
        aid = _assertion_id(paper_id, s, r, o)
        rows.append({
            "aid": aid, "paper_id": paper_id,
            "s": s, "r": r, "o": o,
            "snippet": snippet[:500] if snippet else None,
        })
    if not rows:
        return 0
    cypher = """
    UNWIND $rows AS row
    MERGE (p:Paper {paper_id: row.paper_id})
    MERGE (s:Entity {name: row.s})
    MERGE (o:Entity {name: row.o})
    MERGE (a:Assertion {assertion_id: row.aid})
      ON CREATE SET a.paper_id = row.paper_id,
                    a.predicate = row.r,
                    a.snippet = row.snippet
      ON MATCH  SET a.paper_id = row.paper_id,
                    a.predicate = row.r
    MERGE (s)-[:SUBJECT_OF]->(a)
    MERGE (o)-[:OBJECT_OF]->(a)
    MERGE (a)-[:ASSERTED_IN]->(p)
    """
    with graph_store._driver.session() as session:
        session.run(cypher, rows=rows)
    return len(rows)


def load_processed_papers():
    """Carrega papers processados"""
    papers_file = Path('../data/processed/all_papers.json')
    
    if not papers_file.exists():
        raise FileNotFoundError("ERROR: Arquivo all_papers.json não encontrado!")
    
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    return papers


def create_llamaindex_documents(papers):
    """Converte papers em Documents do LlamaIndex"""
    
    documents = []
    
    print("Creating LlamaIndex documents...")
    
    for paper in tqdm(papers):
        text_parts = []
        if paper.get('title'):
            text_parts.append(f"Title: {paper['title']}\n")
        if paper.get('abstract'):
            text_parts.append(f"Abstract: {paper['abstract']}\n")
        if paper.get('introduction'):
            text_parts.append(f"Introduction: {paper['introduction']}\n")
        if paper.get('methods'):
            text_parts.append(f"Methods: {paper['methods']}\n")
        if paper.get('results'):
            text_parts.append(f"Results: {paper['results']}\n")
        if paper.get('discussion'):
            text_parts.append(f"Discussion: {paper['discussion']}\n")
        if paper.get('conclusion'):
            text_parts.append(f"Conclusion: {paper['conclusion']}\n")
        if not any([paper.get('abstract'), paper.get('results'), paper.get('conclusion')]):
            if paper.get('full_text'):
                text_parts.append(paper['full_text'])
        main_text = '\n'.join(text_parts).strip()
        
        if not main_text or len(main_text) < 100:
            print(f"WARNING: Skipping paper without sufficient text: {paper.get('id')}")
            continue

        metadata = {
            'paper_id': paper.get('id', ''),
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', ''),  # <-- add this
            'authors': paper.get('authors', '')[:500],
            'year': paper.get('year', ''),
            'journal': paper.get('journal', '')[:200],
            'keywords': paper.get('keywords', '')[:300],
            'pmc_link': paper.get('pmc_link', ''),
            'has_abstract': bool(paper.get('abstract')),
            'has_results': bool(paper.get('results')),
            'has_conclusion': bool(paper.get('conclusion'))
        }

        doc = Document(
            text=main_text,
            metadata=metadata,
            id_=paper.get('id', f"doc_{len(documents)}")
        )
        
        documents.append(doc)
    
    print(f"SUCCESS: {len(documents)} documents created")
    return documents



def setup_llamaindex(documents):
    """Configura e carrega documentos em um KnowledgeGraphIndex via Neo4j"""

    print("\nConfiguring Neo4j GraphStore...")

    graph_store = Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL, 
        database=NEO4J_DATABASE
    )

    print("\nEnsuring Schema...")
    ensure_schema(graph_store)

    print("\nConfiguring embedding model (optional, for semantic search)...")

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    Settings.llm = GoogleGenAI(model="gemini-2.5-pro")
    Settings.chunk_size = 512

    llm = GoogleGenAI(
        model="gemini-2.5-pro"
    )

    # Define settings globais
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512 
    Settings.chunk_overlap = 10

    print("\nCreating Document chunks...");

    # This is the step that from_documents() usually does automatically
    parser = SimpleNodeParser.from_defaults(chunk_size=16384, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Document split into {len(nodes)} chunks.")


    # Custom prompt template string
    prompt_str = (
        "Some text is provided below. Given the text, extract up to "
        "{max_knowledge_triplets} directed knowledge triplets in the form of (subject, relation, object). "
        "Avoid stopwords.\n"
        "---------------------\n"
        "Text: {text}\n"
        "---------------------\n"
        "Triplets:\n"
    )
    my_custom_prompt = PromptTemplate(prompt_str)

    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    # Criar o índice do grafo
    kg_index = KnowledgeGraphIndex(
        nodes=[],  # Start with no nodes
        kg_triplet_extract_template=my_custom_prompt,
        storage_context=storage_context,
        max_triplets_per_chunk=100,
        include_embeddings=True,
    )

    print("Building provenance per paper...")
    papers_written = 0
    triples_written = 0

    # --- replace your block (308..338) with this ---

    from collections import defaultdict

# Map paper_id -> document metadata (for title/abstract/keywords)
    doc_meta = {}
    for d in documents:
        pid = (d.metadata or {}).get("paper_id") or d.doc_id
        doc_meta[pid] = d.metadata or {}

# Group chunk nodes by paper_id
    by_paper_nodes = defaultdict(list)
    for idx, n in enumerate(nodes):
        # Try to carry paper_id on chunk; fallback to ref_doc_id / document_id
        pid = None
        if hasattr(n, "metadata") and n.metadata:
            pid = n.metadata.get("paper_id") or n.metadata.get("document_id")
        if not pid:
            pid = getattr(n, "ref_doc_id", None)
        if not pid:
            # last resort: try to read from doc_meta by any hint
            pid = (n.metadata or {}).get("id") or f"unknown_{idx}"
        # persist pid on chunk metadata (helps later)
        if n.metadata is None:
            n.metadata = {}
        n.metadata["paper_id"] = pid
        by_paper_nodes[pid].append(n)

    print(f"Chunk grouping complete: {len(by_paper_nodes)} papers with chunks.")

    papers_written = 0
    triples_written = 0

    for i, (paper_id, chunk_list) in enumerate(by_paper_nodes.items(), start=1):
        # Upsert the Paper node once using document-level metadata, with fallback text
        md0 = doc_meta.get(paper_id, {})
        title = md0.get("title", "")
        abstract = md0.get("abstract", "")
        keywords = md0.get("keywords", "")
        fallback_text = (title + "\n" + abstract + "\n" + keywords).strip()
        upsert_paper_node(graph_store, md0, fallback_text=fallback_text)

        # Accumulate unique (s,r,o) for this paper (dedupe across chunks)
        seen = set()
        batch_triplets = []
        CHUNK_MAX_TRIPLETS = 100  # per chunk cap to keep costs under control

        for cidx, node in enumerate(chunk_list, start=1):
            # get chunk text; LlamaIndex Node supports get_content(); fall back to .text
            try:
                chunk_text = node.get_content()  # preferred
            except Exception:
                chunk_text = getattr(node, "text", "") or ""

            if not chunk_text:
                continue

            # extract triplets from *this chunk*
            triplets = extract_triplets_with_llm(llm, chunk_text, max_triplets=CHUNK_MAX_TRIPLETS)

            # collect unique triplets for this paper
            for (s, r, o) in triplets:
                key = (s, r, o)
                if key in seen:
                    continue
                seen.add(key)
                batch_triplets.append((s, r, o, chunk_text[:500]))  # keep a short snippet
            time.sleep(31)

        # Upsert assertions for this paper in manageable batches
        BATCH = 200
        wrote_for_paper = 0
        for start in range(0, len(batch_triplets), BATCH):
            slice_triplets = batch_triplets[start:start+BATCH]
            # adapt to upsert_assertions API (expects List[Tuple[str,str,str]] + snippet param)
            # We want per-triple snippets; quick adaptation: call per triple (safe) or extend API.
            # To keep it simple & fast: single snippet per batch from the paper abstract/title.
            # If you want strict per-triple snippet, loop per item and call upsert_assertions one by one.
            batch_snippet = (abstract or title)[:500]
            wrote_for_paper += upsert_assertions(
                graph_store,
                paper_id,
                [(s, r, o) for (s, r, o, _) in slice_triplets],
                snippet=batch_snippet
            )

        papers_written += 1
        triples_written += wrote_for_paper
        print(f"  ... processed paper {i}/{len(by_paper_nodes)} | chunks={len(chunk_list)} | assertions+provenance={wrote_for_paper} | total={triples_written}")

    print(f"Provenance build complete. Papers: {papers_written}, Assertions: {triples_written}")


    logging.info("Knowledge graph built successfully from all chunks!")

    print("\nKnowledge Graph Index created successfully in Neo4j!")

    return kg_index, graph_store


def visualize(kg_index):
    # networkx object
    g = kg_index.get_networkx_graph()

    # pyvis network object
    net = Network(cdn_resources="in_line", directed=True, height="750px", width="100%")

    # Load the networkx graph into pyvis
    net.from_nx(g)

    # Physics-based stabilization for a better layout
    net.show_buttons(filter_=['physics'])

    # Interactive HTML file
    net.show("graph.html")
    print("Successfully generated interactive graph visualization: graph.html")


def test_queries(index, documents):
    """Executes test queries on Neo4j"""
    query_engine = index.as_query_engine()

    queries = [
        "Which NASA projects studied plant growth?",
        "What experiments were done in space about microorganisms?",
        "Summarize research about radiation effects on plants.",
    ]

    print("\nRunning test queries...\n")
    for q in queries:
        print(f"Q: {q}")
        response = query_engine.query(q)
        print(f"A: {response}\n{'-'*50}\n")


def main():
    print("=" * 70)
    print("NASA BIOSCIENCE - GRAPH SEARCH (Neo4j + LlamaIndex)")
    print("=" * 70)

    try:
        # 1. Load papers
        print("\nLoading processed papers...")
        papers = load_processed_papers()
        print(f"SUCCESS: {len(papers)} papers loaded")

        # 2. Create Documents
        documents = create_llamaindex_documents(papers)

        if not documents:
            print("ERROR: No valid documents found!")
            return

        # 3. Setup LlamaIndex + Neo4j
        index, graph_store = setup_llamaindex(documents)

        # 4. Verify (count nodes and edges in Neo4j)
        driver = graph_store._driver
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

        print(f"\nTotal nodes in Neo4j: {node_count}")
        print(f"Total relationships in Neo4j: {edge_count}")

        # 5. Test queries
        test_queries(index, documents)

        print("\n" + "=" * 70)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Papers: {len(papers)}")
        print(f"Documents: {len(documents)}")
        print(f"Nodes: {node_count}")
        print(f"Relationships: {edge_count}")
        print("\nGraph search system ready!")
        print("=" * 70)

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
