import json
from pathlib import Path
import chromadb
from tqdm import tqdm

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank


def load_processed_papers():
    """Carrega papers processados"""
    papers_file = Path('data/processed/all_papers.json')
    
    if not papers_file.exists():
        raise FileNotFoundError(
            "ERROR: Arquivo all_papers.json não encontrado!\n"
        )
    
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    return papers

def create_llamaindex_documents(papers):
    """Converte papers em Documents do LlamaIndex"""
    
    documents = []
    
    print("Creating LlamaIndex documents...")
    
    for paper in tqdm(papers):
        # Montar texto principal (priorizar conteúdo mais relevante)
        text_parts = []
        
        # Título
        if paper.get('title'):
            text_parts.append(f"Title: {paper['title']}\n")
        
        # Abstract (muito importante!)
        if paper.get('abstract'):
            text_parts.append(f"Abstract: {paper['abstract']}\n")
        
        # Introduction
        if paper.get('introduction'):
            text_parts.append(f"Introduction: {paper['introduction']}\n")
        
        # Methods (resumido)
        if paper.get('methods'):
            text_parts.append(f"Methods: {paper['methods']}\n")
        
        # Results (muito importante!)
        if paper.get('results'):
            text_parts.append(f"Results: {paper['results']}\n")
        
        # Discussion
        if paper.get('discussion'):
            text_parts.append(f"Discussion: {paper['discussion']}\n")
        
        # Conclusion (importante!)
        if paper.get('conclusion'):
            text_parts.append(f"Conclusion: {paper['conclusion']}\n")
        
        # Se não tem seções, usa texto completo
        if not any([paper.get('abstract'), paper.get('results'), paper.get('conclusion')]):
            if paper.get('full_text'):
                text_parts.append(paper['full_text'])
        
        main_text = '\n'.join(text_parts).strip()
        
        # Se mesmo assim não tem texto, pula
        if not main_text or len(main_text) < 100:
            print(f"WARNING: Skipping paper without sufficient text: {paper.get('id')}")
            continue
        
        # Metadados
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
        
        # Criar Document
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
    
    # ChromaDB
    db_path = Path('database/chroma_db')
    db_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # Deletar collection antiga se existir
    try:
        chroma_client.delete_collection("nasa_bioscience")
        print("Old collection removed")
    except:
        pass
    
    # Criar nova collection
    collection = chroma_client.create_collection(
        name="nasa_bioscience",
        metadata={
            "description": "NASA Space Bioscience Publications - 608 papers",
            "hnsw:space": "cosine"
        }
    )
    
    print("\nConfiguring embedding model...")
    
    # Embedding model (local, gratuito!)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Mais rápido!
        # model_name="sentence-transformers/all-mpnet-base-v2"  # Melhor, mas lento
        device="cpu"  # Ou "cuda" se tiver GPU
    )
    
    # Configurações globais
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 100
    
    print("Loading documents into ChromaDB...")
    print("This may take several minutes depending on hardware...\n")
    
    # Vector Store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Criar index (isso processa embeddings e salva no ChromaDB)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\nIndex created successfully!")
    
    return index, collection

def create_query_engine_with_reranking(index):
    """Creates query engine with cross-encoder re-ranking for better precision"""
    
    print("\nConfiguring query engine with re-ranking...")
    
    # Re-ranking using cross-encoder
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=5  # Final number of results after re-ranking
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=10,  # Get more candidates initially
        node_postprocessors=[rerank],
        response_mode="compact"
    )
    
    print("Query engine with re-ranking configured successfully")
    
    return query_engine

def test_queries(index):
    """Testa o sistema com queries de exemplo"""
    
    print("\n" + "=" * 70)
    print("TESTING QUERIES WITH RE-RANKING")
    print("=" * 70)
    
    # Use query engine with re-ranking
    query_engine = create_query_engine_with_reranking(index)
    
    test_queries_list = [
        "What are the main effects of microgravity on human health?",
        "How does space radiation affect biological systems?",
        "What challenges exist for growing plants in space?",
        "What are the key findings about bone loss in space?"
    ]
    
    for i, query in enumerate(test_queries_list, 1):
        print(f"\n{'-' * 70}")
        print(f"Query {i}: {query}")
        print('-' * 70)
        
        response = query_engine.query(query)
        
        print(f"\nResponse:")
        print(str(response)[:500] + "..." if len(str(response)) > 500 else str(response))
        
        print(f"\nRelevant papers ({len(response.source_nodes)} results):")
        for j, node in enumerate(response.source_nodes, 1):
            meta = node.node.metadata
            score = node.score if hasattr(node, 'score') else 'N/A'
            print(f"  {j}. {meta.get('title', 'N/A')[:80]}...")
            print(f"     Score: {score:.4f} | Year: {meta.get('year', 'N/A')}")

def main():
    print("=" * 70)
    print("NASA BIOSCIENCE - LOADING TO CHROMADB + LLAMAINDEX")
    print("=" * 70)
    
    try:
        # 1. Carregar papers processados
        print("\nLoading processed papers...")
        papers = load_processed_papers()
        print(f"SUCCESS: {len(papers)} papers loaded")
        
        # 2. Criar Documents
        documents = create_llamaindex_documents(papers)
        
        if not documents:
            print("ERROR: No valid documents found!")
            return
        
        # 3. Setup LlamaIndex + ChromaDB
        index, collection = setup_llamaindex(documents)
        
        # 4. Verificar
        count = collection.count()
        print(f"\nTotal chunks in ChromaDB: {count}")
        
        # 5. Testar com re-ranking
        test_queries(index)
        
        print("\n" + "=" * 70)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Database: {Path('database/chroma_db').absolute()}")
        print(f"Papers: {len(papers)}")
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {count}")
        print(f"Chunks per document: ~{count/len(documents):.1f}")
        print("\nNext step: Create interactive graph!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
