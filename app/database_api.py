from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import os
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi

load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "../database/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

# Hybrid search parameters
ALPHA = float(os.getenv("ALPHA", "0.4"))  # 40% vector, 60% BM25
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

app = FastAPI(title="NASA Bioscience Database API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class ChunkResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict

class QueryResponse(BaseModel):
    query: str
    total_results: int
    chunks: List[ChunkResult]

# Global state
collection = None
embedding_model = None
bm25_index = None
all_ids = None
all_documents = None
all_metadatas = None
doc_map = None

@app.on_event("startup")
async def startup_event():
    global collection, embedding_model, bm25_index, all_ids, all_documents, all_metadatas, doc_map
    
    print("Initializing ChromaDB...")
    db_path = Path(DATABASE_PATH)
    
    if not db_path.exists():
        print(f"ERROR: ChromaDB not found at {DATABASE_PATH}")
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=str(db_path))
    
    try:
        collection = client.get_collection("nasa_bioscience")
        print(f"Collection loaded: {collection.count()} chunks")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Build BM25 index
    print("Building BM25 index for hybrid search...")
    all_data = collection.get(include=['documents', 'metadatas'])
    all_ids = all_data['ids']
    all_documents = all_data['documents']
    all_metadatas = all_data['metadatas']
    
    # Tokenize for BM25
    tokenized_corpus = [doc.lower().split() for doc in all_documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # Create lookup map
    doc_map = {doc_id: idx for idx, doc_id in enumerate(all_ids)}
    
    print(f"BM25 index built with {len(all_ids)} documents")
    print(f"Hybrid search ready: {ALPHA*100:.0f}% vector + {(1-ALPHA)*100:.0f}% BM25")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print("Database API ready!")

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "database_api",
        "total_chunks": collection.count() if collection else 0,
        "search_mode": "hybrid",
        "alpha": ALPHA,
        "threshold": SIMILARITY_THRESHOLD
    }

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Execute hybrid query (Vector + BM25) against ChromaDB"""
    if not collection or not embedding_model or bm25_index is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    
    try:
        top_k = min(request.top_k, 50)
        
        # 1. Vector search
        query_embedding = embedding_model.encode(request.query).tolist()
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=20,  
            include=['documents', 'metadatas', 'distances']
        )
        
        # Build vector scores dict
        vector_scores = {}
        for doc_id, distance in zip(vector_results['ids'][0], vector_results['distances'][0]):
            vector_scores[doc_id] = 1 - distance  
        
        # 2. BM25 search
        tokenized_query = request.query.lower().split()
        bm25_scores_raw = bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores_norm = bm25_scores_raw / max_bm25
        
        # 3. Combine scores with threshold filtering
        combined_scores = {}
        for doc_id in all_ids:
            idx = doc_map[doc_id]
            score = 0.0
            
            # Add vector component
            if doc_id in vector_scores:
                score += ALPHA * vector_scores[doc_id]
            
            # Add BM25 component
            score += (1 - ALPHA) * bm25_scores_norm[idx]
            
            # Only include if above threshold
            if score >= SIMILARITY_THRESHOLD:
                combined_scores[doc_id] = score
        
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
               # 5. Format response with clean metadata
        chunks = []
        for doc_id, score in sorted_results:
            idx = doc_map[doc_id]
            raw_metadata = all_metadatas[idx]

            # Extract only the fields you want
            clean_metadata = {
                'paper_id': raw_metadata.get('paper_id', ''),
                'title': raw_metadata.get('title', ''),
                'authors': raw_metadata.get('authors', ''),
                'year': raw_metadata.get('year', ''),
                'keywords': raw_metadata.get('keywords', ''),
                'pmc_link': raw_metadata.get('pmc_link', ''),
                'journal': raw_metadata.get('journal', ''),
            }

            chunks.append(ChunkResult(
                chunk_id=doc_id,
                score=float(score),
                text=all_documents[idx],
                metadata=clean_metadata  # Only clean metadata
            ))

        return QueryResponse(
            query=request.query,
            total_results=len(chunks),
            chunks=chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Database statistics"""
    if not collection:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        sample = collection.peek(limit=100)
        
        unique_papers = set()
        years = []
        
        for meta in sample['metadatas']:
            if meta.get('paper_id'):
                unique_papers.add(meta['paper_id'])
            if meta.get('year'):
                try:
                    years.append(int(meta['year']))
                except:
                    pass
        
        return {
            "total_chunks": collection.count(),
            "unique_papers_sample": len(unique_papers),
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
            "search_mode": "hybrid",
            "alpha": ALPHA,
            "threshold": SIMILARITY_THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get all chunks for a specific paper"""
    if not collection:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        results = collection.get(
            where={"paper_id": paper_id},
            include=['documents', 'metadatas']
        )
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail=f"Paper not found")
        
        chunks = []
        for chunk_id, doc_text, metadata in zip(
            results['ids'],
            results['documents'],
            results['metadatas']
        ):
            chunks.append({
                'chunk_id': chunk_id,
                'text': doc_text,
                'metadata': metadata
            })
        
        return {
            'paper_id': paper_id,
            'total_chunks': len(chunks),
            'chunks': chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
