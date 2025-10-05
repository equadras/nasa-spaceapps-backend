"""
Database API - Runs on the machine with ChromaDB
Handles direct database queries
"""
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

load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "database/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")

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

@app.on_event("startup")
async def startup_event():
    global collection, embedding_model
    
    print("Initializing ChromaDB...")
    db_path = Path('database/chroma_db')
    
    if not db_path.exists():
        print("ERROR: ChromaDB not found")
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=str(db_path))
    
    try:
        collection = client.get_collection("nasa_bioscience")
        print(f"Collection loaded: {collection.count()} chunks")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Database API ready!")

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "database_api",
        "total_chunks": collection.count() if collection else 0
    }

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Execute query against ChromaDB"""
    if not collection or not embedding_model:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")
    
    try:
        # Generate embedding
        query_embedding = embedding_model.encode(request.query).tolist()
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.top_k, 50),
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format response
        chunks = []
        if results['ids'][0]:
            for chunk_id, doc_text, metadata, distance in zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                chunks.append(ChunkResult(
                    chunk_id=chunk_id,
                    score=float(1 - distance),
                    text=doc_text,
                    metadata=metadata
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
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A"
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
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Port 8001 for database API
