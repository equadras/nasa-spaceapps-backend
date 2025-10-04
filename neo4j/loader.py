import os
from dotenv import load_dotenv

import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from rank_bm25 import BM25Okapi

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

from pathlib import Path
from llama_index.core import Settings
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_llamaindex(documents):
    """Configura e carrega documentos em um KnowledgeGraphIndex via Neo4j"""

    print("\nConfiguring Neo4j GraphStore...")

    # ⚡ configure sua conexão Neo4j
    graph_store = Neo4jGraphStore(
        username="neo4j",
        password="your_password",   # altere para sua senha real
        url="bolt://localhost:7687",  # ou bolt+s:// para Aura
        database="neo4j",            # padrão
    )

    print("\nConfiguring embedding model (optional, for semantic search)...")

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    Settings.llm = GoogleGenAI(model="gemini-2.5-pro")
    Settings.chunk_size = 512

    llm = GoogleGenAI(
        model="gemini-2.5-pro",
        # Tenacity is used by default for retries
        max_retries=5,
        timeout=60.0,
    )

    # Define settings globais
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 16382
    Settings.chunk_overlap = 10

    print("\nLoading documents into KnowledgeGraphIndex (Neo4j)...")
    print("This may take several minutes depending on hardware...\n")

    # Cria grafo e index
    from llama_index.core import (
        Document,
        KnowledgeGraphIndex,
        PromptTemplate
    )
    from llama_index.core.node_parser import SimpleNodeParser
    from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
    import time
    import logging

    # --- 2. Manually Create Chunks (Nodes) ---
    # This is the step that from_documents() usually does automatically
    parser = SimpleNodeParser.from_defaults(chunk_size=16384, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Document split into {len(nodes)} chunks.")


    # Define your custom prompt template string
    prompt_str = (
        "Some text is provided below. Given the text, extract up to "
        "{max_knowledge_triplets} directed knowledge triplets in the form of (subject, relation, object). "
        "Avoid stopwords.\n"
        "---------------------\n"
        "Text: {text}\n"
        "---------------------\n"
        "Triplets:\n"
    )

    # Create a LlamaIndex PromptTemplate object
    my_custom_prompt = PromptTemplate(prompt_str)

    # --- 3. Create an Empty Knowledge Graph Index ---
    kg_index = KnowledgeGraphIndex(
        nodes=[],  # Start with no nodes
        kg_triplet_extract_template=my_custom_prompt,
        max_triplets_per_chunk=100,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
    )

    # --- 4. Loop and Insert Each Chunk Individually ---
    print("Processing chunks and building knowledge graph...")
    for i, node in enumerate(nodes):
        try:
            # Call the retry-enabled function for each node
            kg_index.insert_nodes([node])
            print(f"Loaded node #{i}. Nodes left: {len(nodes)-i-1}")
            time.sleep(31)
        except Exception as e:
            # This will only be hit if all retry attempts for a specific node fail
            logging.error(f"Failed to process node #{i}. Aborting. Error: {e}")
            break

    logging.info("Knowledge graph built successfully from all chunks!")

    # Criar o índice do grafo
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=None,        # usa o default
        graph_store=graph_store,     # 👉 salva no Neo4j em vez de memória
        max_triplets_per_chunk=10,
        include_embeddings=True,     # se quiser semantic query
        show_progress=True,
    )

    print("\nKnowledge Graph Index created successfully in Neo4j!")

    return index, graph_store


def visualize():
    from pyvis.network import Network
    import logging
    import sys

    # --- Visualization Code ---
    # 1. Get the graph as a networkx object
    g = kg_index.get_networkx_graph()

    # 2. Create a pyvis network object
    #    `notebook=True` is great for Jupyter/Colab environments.
    #    `cdn_resources='in_line'` makes the HTML file self-contained.
    net = Network(notebook=True, cdn_resources="in_line", directed=True, height="750px", width="100%")

    # 3. Load the networkx graph into pyvis
    net.from_nx(g)

    # 4. Add physics-based stabilization for a better layout
    net.show_buttons(filter_=['physics'])

    # 5. Generate the interactive HTML file
    net.show("graph.html")
    print("Successfully generated interactive graph visualization: graph.html")


def test_queries(index, documents):
    """Executa algumas queries de teste no KnowledgeGraphIndex (Neo4j)"""
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
        driver = graph_store.driver
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

        visualize()  # se você já tiver uma função para desenhar

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
