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


def load_processed_papers():
    """Carrega papers processados"""
    papers_file = Path('../data/processed/all_papers.json')
    
    if not papers_file.exists():
        raise FileNotFoundError(
            " Arquivo all_papers.json não encontrado!\n"
            "Execute primeiro: python scripts/1_download_and_process.py"
        )
    
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    return papers

def create_llamaindex_documents(papers):
    """Converte papers em Documents do LlamaIndex"""
    
    documents = []
    
    print(" Criando documentos LlamaIndex...")
    
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
            print(f"⚠️  Pulando paper sem texto suficiente: {paper.get('id')}")
            continue
        
        metadata = {
            'paper_id': paper.get('id', ''),
            'title': paper.get('title', ''),
            'authors': paper.get('authors', ''),
            'year': paper.get('year', ''),
            'journal': paper.get('journal', ''),
            'keywords': paper.get('keywords', ''),
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
    
    print(f"{len(documents)} documentos criados")
    return documents

def setup_llamaindex(documents):
    """Configura e carrega no ChromaDB via LlamaIndex"""
    
    print("\n🔧 Configurando ChromaDB...")
    
    # ChromaDB
    db_path = Path('../database/chroma_db')
    db_path.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # Deletar collection antiga se existir
    try:
        chroma_client.delete_collection("nasa_bioscience")
        print("Collection antiga removida")
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
    
    print("\nConfigurando modelo de embeddings...")
    
    # Embedding model (local, gratuito!)
    embed_model = HuggingFaceEmbedding(

        model_name="sentence-transformers/all-MiniLM-L6-v2", # Mais rápido!
        # model_name="sentence-transformers/all-mpnet-base-v2"  Melhor, mas lento
        device="cpu"  # Ou "cuda" se tiver GPU
    )
    
    # Configurações globais
    Settings.embed_model = embed_model
    #Settings.chunk_size = 512
    #Settings.chunk_overlap = 50
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 100
    
    print("Carregando documentos no ChromaDB...")
    print("Isso pode levar alguns minutos dependendo do hardware...\n")
    
    # Vector Store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Criar index (isso processa embeddings e salva no ChromaDB)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\nIndex criado com sucesso!")
    
    return index, collection


def main():
    try:
        # 1. Carregar papers processados
        print("\nCarregando papers processados...")
        papers = load_processed_papers()
        print(f"{len(papers)} papers carregados")
        
        # 2. Criar Documents
        documents = create_llamaindex_documents(papers)
        
        if not documents:
            print("Nenhum documento válido encontrado!")
            return
        
        # 3. Setup LlamaIndex + ChromaDB
        index, collection = setup_llamaindex(documents)
        
        # 4. Verificar
        count = collection.count()
        print(f"\nTotal de chunks no ChromaDB: {count}")
        
        
        print("\n" + "=" * 70)
        print(" PROCESSO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        print(f" Database: {Path('../database/chroma_db').absolute()}")
        print(f" Papers: {len(papers)}")
        print(f" Documents: {len(documents)}")
        print(f" Chunks: {count}")
        print("\n Próximo passo: Criar o grafo interativo!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
