import json
from pathlib import Path
import chromadb
from tqdm import tqdm
import numpy as np

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


class HybridRetriever:
    """Combines BM25 keyword search with vector semantic search"""
    
    def __init__(self, index, documents, alpha=0.5):
        """
        Args:
            index: LlamaIndex VectorStoreIndex
            documents: List of Document objects used to build the index
            alpha: Weight for vector search (0-1). Higher = more semantic, lower = more keyword-focused
        """
        self.vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=20)
        self.alpha = alpha
        
        print(f"Building BM25 index from {len(documents)} documents...")
        self.documents = documents
        self.doc_map = {doc.id_: doc for doc in documents}
        
        # Tokenize documents for BM25
        tokenized_corpus = [doc.text.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully")
    
    def retrieve(self, query_str, top_k=10):
        """Retrieve documents using hybrid BM25 + vector search"""
        
        # Get vector search results
        vector_nodes = self.vector_retriever.retrieve(query_str)
        
        # Get BM25 scores
        tokenized_query = query_str.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = bm25_scores / max_bm25
        
        # Combine scores
        combined_scores = {}
        
        # Add vector scores
        for node in vector_nodes:
            doc_id = node.node.id_
            combined_scores[doc_id] = self.alpha * node.score
        
        # Add BM25 scores
        for idx, doc in enumerate(self.documents):
            doc_id = doc.id_
            bm25_score = bm25_scores_norm[idx]
            
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - self.alpha) * bm25_score
            else:
                combined_scores[doc_id] = (1 - self.alpha) * bm25_score
        
        # Sort and get top_k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Convert to NodeWithScore format
        final_nodes = []
        for doc_id, score in sorted_results:
            if doc_id in self.doc_map:
                doc = self.doc_map[doc_id]
                node = NodeWithScore(node=doc, score=score)
                final_nodes.append(node)
        
        return final_nodes


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
    """Configura e carrega no ChromaDB via LlamaIndex"""
    
    print("\nConfiguring ChromaDB...")
    
    db_path = Path('../database/chroma_db')
    db_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    try:
        chroma_client.delete_collection("nasa_bioscience")
        print("Old collection removed")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="nasa_bioscience",
        metadata={
            "description": "NASA Space Bioscience Publications - 608 papers",
            "hnsw:space": "cosine"
        }
    )
    
    print("\nConfiguring embedding model...")
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 100
    
    print("Loading documents into ChromaDB...")
    print("This may take several minutes depending on hardware...\n")
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\nIndex created successfully!")
    
    return index, collection


def create_hybrid_query_engine(index, documents, alpha=0.5, use_reranking=True):
    """Creates query engine with BM25 + Vector hybrid retrieval and optional re-ranking"""
    
    print("\nConfiguring hybrid retriever (BM25 + Vector)...")
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        index=index,
        documents=documents,
        alpha=alpha  # Adjust: 0.5 = equal weight, <0.5 = more keyword-focused
    )
    
    # Optional re-ranking
    if use_reranking:
        print("Adding cross-encoder re-ranking...")
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=5
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            node_postprocessors=[rerank]
        )
    else:
        query_engine = RetrieverQueryEngine(retriever=hybrid_retriever)
    
    print("Hybrid query engine configured successfully")
    return query_engine


def test_queries(index, documents):
    """Testa o sistema com queries de exemplo"""
    
    print("\n" + "=" * 70)
    print("TESTING HYBRID RETRIEVAL (BM25 + Vector + Re-ranking)")
    print("=" * 70)
    
    # Create hybrid query engine with alpha=0.4 for better keyword matching
    query_engine = create_hybrid_query_engine(
        index, 
        documents, 
        alpha=0.4,  # 40% vector, 60% BM25 - better for keyword-specific queries
        use_reranking=True
    )
    
    test_queries_list = [
        "water deprived environments",  # Your problematic query
        "What are the main effects of microgravity on human health?",
        "How does space radiation affect biological systems?",
        "bone loss in space",
        "immune system changes during spaceflight"
    ]
    
    for i, query in enumerate(test_queries_list, 1):
        print(f"\n{'-' * 70}")
        print(f"Query {i}: {query}")
        print('-' * 70)
        
        response = query_engine.query(query)
        
        print(f"\nResponse:")
        print(str(response)[:400] + "..." if len(str(response)) > 400 else str(response))
        
        print(f"\nRelevant papers ({len(response.source_nodes)} results):")
        for j, node in enumerate(response.source_nodes, 1):
            meta = node.node.metadata
            score = node.score if hasattr(node, 'score') else 'N/A'
            print(f"  {j}. {meta.get('title', 'N/A')[:80]}...")
            print(f"     Score: {score:.4f} | Year: {meta.get('year', 'N/A')}")


def main():
    print("=" * 70)
    print("NASA BIOSCIENCE - HYBRID SEARCH (BM25 + Vector + Re-ranking)")
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
        
        # 3. Setup LlamaIndex + ChromaDB
        index, collection = setup_llamaindex(documents)
        
        # 4. Verify
        count = collection.count()
        print(f"\nTotal chunks in ChromaDB: {count}")
        
        # 5. Test with hybrid retrieval	
        test_queries(index, documents)
        
        print("\n" + "=" * 70)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Database: {Path('../database/chroma_db').absolute()}")
        print(f"Papers: {len(papers)}")
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {count}")
        print(f"Chunks per document: ~{count/len(documents):.1f}")
        print("\nHybrid search system ready!")
        print("Alpha=0.4 (40% semantic, 60% keyword matching)")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
