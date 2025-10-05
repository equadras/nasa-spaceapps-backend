# build_graph_from_existing.py
import json
from pathlib import Path
import chromadb
from tqdm import tqdm
import numpy as np
import networkx as nx

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever


def load_existing_chromadb():
    """Load your existing ChromaDB without rebuilding"""
    
    print("Loading existing ChromaDB...")
    
    # Load papers for metadata
    papers_file = Path('../data/processed/all_papers.json')
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Create paper lookup
    papers_dict = {p['id']: p for p in papers}
    
    # Connect to existing ChromaDB
    db_path = Path('../database/chroma_db')
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_collection("nasa_bioscience")
    
    print(f"Loaded collection with {collection.count()} chunks")
    
    # Setup same embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    Settings.embed_model = embed_model
    
    # Create index from existing vector store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    return index, papers_dict


def build_paper_graph(index, papers_dict, max_neighbors=10, similarity_threshold=0.3):
    """
    Build graph by querying each paper against the database
    
    Args:
        index: Existing VectorStoreIndex
        papers_dict: Dict of paper_id -> paper data
        max_neighbors: Max connections per paper
        similarity_threshold: Min similarity to create edge (0-1)
    """
    
    print(f"\nBuilding similarity graph...")
    print(f"Papers: {len(papers_dict)}")
    print(f"Threshold: {similarity_threshold}")
    print(f"Max neighbors: {max_neighbors}")
    
    # Initialize graph structure
    graph = nx.Graph()
    
    # Add all papers as nodes
    for paper_id, paper in papers_dict.items():
        graph.add_node(
            paper_id,
            title=paper.get('title', ''),
            year=paper.get('year', ''),
            authors=paper.get('authors', ''),
            keywords=paper.get('keywords', '')
        )
    
    # Create retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=max_neighbors + 5)
    
    # Query each paper
    for paper_id, paper in tqdm(papers_dict.items(), desc="Querying papers"):
        # Build query from paper content
        query_parts = []
        if paper.get('title'):
            query_parts.append(paper['title'])
        if paper.get('abstract'):
            query_parts.append(paper['abstract'][:500])
        
        query_text = ' '.join(query_parts)
        
        if not query_text:
            continue
        
        # Query for similar papers
        try:
            results = retriever.retrieve(query_text)
            
            # Add edges for similar papers
            for node in results:
                similar_paper_id = node.node.metadata.get('paper_id')
                
                if not similar_paper_id or similar_paper_id == paper_id:
                    continue
                
                score = node.score
                
                # Only add edge if above threshold
                if score >= similarity_threshold:
                    if not graph.has_edge(paper_id, similar_paper_id):
                        graph.add_edge(
                            paper_id,
                            similar_paper_id,
                            weight=float(score),
                            similarity=float(score)
                        )
        
        except Exception as e:
            print(f"Error processing {paper_id}: {e}")
            continue
    
    print(f"\nGraph complete!")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Avg connections per paper: {2 * graph.number_of_edges() / graph.number_of_nodes():.1f}")
    
    return graph


def save_graph_custom_format(graph, papers_dict, filename='paper_graph.json'):
    """
    Save graph with complete metadata in nodes
    """

    nodes = []
    for node_id in graph.nodes():
        paper = papers_dict.get(node_id, {})

        nodes.append({
            'id': node_id,
            'title': paper.get('title', ''),
            'authors': paper.get('authors', ''),
            'year': paper.get('year', ''),
            'keywords': paper.get('keywords', ''),
            'pmc_link': paper.get('pmc_link', ''),
            'abstract': paper.get('abstract', ''),
            })

    # Build edges dict
    edges = {}
    for node_id in graph.nodes():
        neighbors = list(graph.neighbors(node_id))
        edges[node_id] = {
                'edges': neighbors
                }

    # Combine
    graph_data = {
            'nodes': nodes,
            'edges': edges
            }

    # Save
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"Saved custom format with full metadata: {filename}")

    return graph_data


def analyze_graph(graph):
    """Print graph statistics"""
    
    print("\n" + "="*70)
    print("GRAPH STATISTICS")
    print("="*70)
    
    
    degrees = [d for n, d in graph.degree()]
    print(f"\nDegree distribution:")
    print(f"  Min: {min(degrees)}")
    print(f"  Max: {max(degrees)}")
    print(f"  Mean: {np.mean(degrees):.1f}")
    print(f"  Median: {np.median(degrees):.1f}")
    
    # Most connected papers
    print(f"\nTop 10 most connected papers:")
    top_papers = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:10]
    for i, (paper_id, degree) in enumerate(top_papers, 1):
        title = graph.nodes[paper_id].get('title', 'N/A')[:60]
        year = graph.nodes[paper_id].get('year', 'N/A')
        print(f"  {i}. {title}... ({year}) - {degree} connections")


def main():
    print("="*70)
    print("BUILD PAPER SIMILARITY GRAPH FROM EXISTING DATABASE")
    print("="*70)
    
    # Load existing ChromaDB
    index, papers_dict = load_existing_chromadb()
    
    # Build graph
    graph = build_paper_graph(
        index,
        papers_dict,
        max_neighbors=7,
        similarity_threshold=0.5
    )
    
    # Analyze
    analyze_graph(graph)
    
    # Save in your custom format
    graph_data = save_graph_custom_format(graph, papers_dict, 'paper_graph.json')
    
    # Also save as GraphML for visualization tools (optional)
    nx.write_graphml(graph, 'paper_similarity.graphml')
    print("Saved GraphML: paper_similarity.graphml")
    
    print("\n" + "="*70)
    print("DONE! Graph file created: paper_graph.json")
    print("="*70)


if __name__ == "__main__":
    main()
