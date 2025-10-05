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

    print("Processing chunks and building knowledge graph...")
    for i, node in enumerate(nodes):
        try:
            print(f"Loading node #{i}...")
            kg_index.insert_nodes([node])
            print(f"OK node #{i}. Nodes left: {len(nodes)-i-1}")
            time.sleep(31)
        except Exception as e:
            logging.error(f"Failed to process node #{i}. Aborting. Error: {e}")
            break

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
#test_queries(index, documents)

        print("\n" + "=" * 70)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Papers: {len(papers)}")
        print(f"Documents: {len(documents)}")
        print(f"Nodes: {node_count}")
        print(f"Relationships: {edge_count}")
        print("\nGraph search system ready!")
        print("=" * 70)

        visualize(index)

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
