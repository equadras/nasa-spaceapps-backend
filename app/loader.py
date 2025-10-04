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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank


def load_processed_papers():
    """Carrega papers processados"""
    papers_file = Path('data/processed/all_papers.json')
    
    if not papers_file.exists():
        raise FileNotFoundError(
            "ERROR: Arquivo all_papers.json não encontrado!\n"
            "Execute primeiro: python scripts/1_download_and_process.py"
        )
    
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    return papers


def create_llamaindex_documents(papers):
    """
    IMPROVEMENT #1: Section-Based Document Creation
    Creates separate documents for each section to enable precise retrieval
    """
    
    documents = []
    
    print("Creating section-based documents...")
    
    # Define sections with importance weights for re-ranking
    section_configs = [
        ('abstract', 1.0, True),      # weight, is_required
        ('introduction', 0.7, False),
        ('methods', 0.5, False),
        ('results', 0.9, False),
        ('discussion', 0.8, False),
        ('conclusion', 0.95, False)
    ]
    
    for paper in tqdm(papers, desc="Processing papers"):
        paper_id = paper.get('id', f"paper_{len(documents)}")
        
        # Base metadata shared across all sections
        base_metadata = {
            'paper_id': paper_id,
            'title': paper.get('title', 'Unknown Title'),
            'authors': paper.get('authors', 'Unknown Authors'),
            'year': str(paper.get('year', 'N/A')),
            'journal': paper.get('journal', 'Unknown Journal'),
            'keywords': paper.get('keywords', ''),
            'pmc_link': paper.get('pmc_link', '')
        }
        
        # Track if paper has any valid content
        has_content = False
        
        # Create document for each section
        for section_name, weight, is_required in section_configs:
            section_text = paper.get(section_name, '').strip()
            
            if not section_text or len(section_text) < 50:
                continue
            
            has_content = True
            
            # Add section header for context
            formatted_text = f"[{section_name.upper()}]\n"
            formatted_text += f"Paper: {paper.get('title', 'Unknown')}\n\n"
            formatted_text += section_text
            
            # Section-specific metadata
            metadata = {
                **base_metadata,
                'section': section_name,
                'section_weight': weight,
                'text_length': len(section_text),
                'is_main_content': section_name in ['abstract', 'results', 'conclusion']
            }
            
            doc = Document(
                text=formatted_text,
                metadata=metadata,
                id_=f"{paper_id}_{section_name}"
            )
            
            documents.append(doc)
        
        # Fallback: if no sections found, use full_text
        if not has_content and paper.get('full_text'):
            full_text = paper['full_text'].strip()
            if len(full_text) >= 100:
                metadata = {
                    **base_metadata,
                    'section': 'full_text',
                    'section_weight': 0.6,
                    'text_length': len(full_text),
                    'is_main_content': True
                }
                
                doc = Document(
                    text=full_text,
                    metadata=metadata,
                    id_=f"{paper_id}_full"
                )
                documents.append(doc)
    
    print(f"SUCCESS: {len(documents)} section documents created from {len(papers)} papers")
    return documents


def setup_llamaindex(documents):
    """
    IMPROVEMENT #2: Improved Chunking Strategy
    Uses SentenceSplitter with optimized settings for scientific papers
    """
    
    print("\nConfiguring ChromaDB...")
    
    # ChromaDB setup
    db_path = Path('database/chroma_db')
    db_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # Delete old collection if exists
    try:
        chroma_client.delete_collection("nasa_bioscience")
        print("Old collection removed")
    except:
        pass
    
    # Create new collection
    collection = chroma_client.create_collection(
        name="nasa_bioscience",
        metadata={
            "description": "NASA Space Bioscience Publications - Section-based chunking",
            "hnsw:space": "cosine"
        }
    )
    
    print("\nConfiguring embedding model...")
    
    # Embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
         # model_name="sentence-transformers/all-mpnet-base-v2"
        device="cuda" #cuda for gpu or cpu for cpu
    )
    
    # IMPROVED CHUNKING: Sentence-aware splitting
    text_splitter = SentenceSplitter(
        chunk_size=512,           # Smaller chunks for precise retrieval
        chunk_overlap=128,        # Larger overlap to preserve context
        separator=" ",
        paragraph_separator="\n\n"
    )
    
    # Global settings
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.chunk_size = 512
    Settings.chunk_overlap = 128
    
    print("Loading documents into ChromaDB...")
    print("This may take several minutes...\n")
    
    # Vector Store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\nIndex created successfully!")
    
    return index, collection


def create_query_engine(index):
    """
    IMPROVEMENT #3: Custom Query Engine with Re-ranking
    Uses cross-encoder to re-rank results based on semantic relevance
    """
    
    print("\nConfiguring query engine with re-ranking...")
    
    # Re-ranking using cross-encoder for better precision
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_n=5  # Final number of results after re-ranking
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=15,  # Get more candidates initially
        node_postprocessors=[rerank],
        response_mode="compact"  # Efficient response generation
    )
    
    print("Query engine configured successfully")
    
    return query_engine


def query_with_sources(query_engine, query_text, show_chunks=True):
    """
    IMPROVEMENT #4: Enhanced Retrieval with Source Tracking
    Detailed source attribution showing which chunks contributed to response
    """
    
    print(f"\n{'='*80}")
    print(f"QUERY: {query_text}")
    print(f"{'='*80}\n")
    
    response = query_engine.query(query_text)
    
    print("RESPONSE:")
    print("-" * 80)
    print(response.response)
    print()
    
    print(f"{'='*80}")
    print(f"SOURCE CHUNKS ({len(response.source_nodes)} results):")
    print(f"{'='*80}\n")
    
    for i, node in enumerate(response.source_nodes, 1):
        meta = node.node.metadata
        score = node.score if hasattr(node, 'score') else 'N/A'
        
        # Header with metadata
        print(f"[{i}] {meta.get('title', 'Unknown')[:70]}...")
        print(f"    Section: {meta.get('section', 'unknown').upper()} | "
              f"Score: {score:.4f} | "
              f"Year: {meta.get('year', 'N/A')} | "
              f"Weight: {meta.get('section_weight', 'N/A')}")
        
        if meta.get('pmc_link'):
            print(f"    Link: {meta['pmc_link']}")
        
        # Show chunk text if requested
        if show_chunks:
            chunk_text = node.node.text.replace('\n', ' ')
            print(f"    Preview: {chunk_text}...")
        
        print()
    
    return response


def test_queries(query_engine):
    """Test the system with example queries"""
    
    print("\n" + "="*80)
    print("TESTING ENHANCED RETRIEVAL SYSTEM")
    print("="*80)
    
    test_queries_list = [
        "What are the main effects of microgravity on bone density in mice?",
        "How does space radiation affect immune system function?",
        "What challenges exist for long-duration spaceflight?",
        "Describe findings about muscle atrophy in space missions"
    ]
    
    for query in test_queries_list:
        query_with_sources(query_engine, query, show_chunks=True)
        print("\n" + "-"*80 + "\n")


def main():
    print("="*80)
    print("NASA BIOSCIENCE - ENHANCED EMBEDDING SYSTEM")
    print("="*80)
    
    try:
        # 1. Load papers
        print("\nLoading processed papers...")
        papers = load_processed_papers()
        print(f"SUCCESS: {len(papers)} papers loaded")
        
        # 2. Create section-based documents
        documents = create_llamaindex_documents(papers)
        
        if not documents:
            print("ERROR: No valid documents found!")
            return
        
        # 3. Setup with improved chunking
        index, collection = setup_llamaindex(documents)
        
        # 4. Create query engine with re-ranking
        query_engine = create_query_engine(index)
        
        # 5. Verify
        count = collection.count()
        print(f"\nTotal chunks in ChromaDB: {count}")
        
        # 6. Test with enhanced retrieval
        test_queries(query_engine)
        
        # Summary
        print("\n" + "="*80)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Database: {Path('database/chroma_db').absolute()}")
        print(f"Original papers: {len(papers)}")
        print(f"Section documents: {len(documents)}")
        print(f"Total chunks: {count}")
        print(f"Chunks per document: ~{count/len(documents):.1f}")
        print("\nSystem ready for queries!")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
