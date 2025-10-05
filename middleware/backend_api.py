"""
Backend API - Public API with rate limiting and app identification
"""
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.generativeai as genai


load_dotenv()

DATABASE_API_URL = os.getenv("DATABASE_API_URL")
FRONTEND_KEY = os.getenv("FRONTEND_KEY", "nasa-bioscience-public-v1")
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

 
#for model in genai.list_models():
#print(model.name)

# Validate required config
if not DATABASE_API_URL:
    raise ValueError("DATABASE_API_URL environment variable is required")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NASA Bioscience Backend API",
    version="1.0.0",
    description="Public API for querying NASA space bioscience papers"
)

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - only allow your frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
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

class SummarizeRequest(BaseModel):
    query: str
    chunks: List[ChunkResult]

class PaperReference(BaseModel):
    paper_id: str
    title: str

class QueryResponse(BaseModel):
    content: str
    referenceNodes: List[PaperReference]


# Helper function for app identification
def verify_app_id(x_app_id: Optional[str] = Header(None)):
    """
    Simple app identifier check (not for security, just to identify legitimate traffic)
    This is visible in frontend code - it's not a secret
    """
    if x_app_id != FRONTEND_KEY:
        logger.warning(f"Request with invalid/missing app ID: {x_app_id}")
        raise HTTPException(
            status_code=403,
            detail="Missing or invalid app identifier"
        )
    return x_app_id

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "service": "NASA Bioscience Backend API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check - no auth required"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DATABASE_API_URL}/health",
                timeout=5.0
            )
            db_status = response.json()
        
        return {
            "backend": "healthy",
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Database API unreachable: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def query_papers(
    request: Request,
    query_request: QueryRequest,
    x_app_id: str = Header(None)
):
    """Query papers and generate AI summary"""
    verify_app_id(x_app_id)
    
    if not query_request.query or len(query_request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    
    if query_request.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k cannot exceed 50")
    
    if query_request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    
    logger.info(f"Query: '{query_request.query}' (top_k={query_request.top_k})")
    
    try:
        # Get results from database
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DATABASE_API_URL}/query",
                json=query_request.model_dump(),
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
        
        chunks = [ChunkResult(**chunk) for chunk in result['chunks']]
        
        content = ""
        reference_nodes = []
        tittle = ""

#want to add the paper tittle here
        tittle = ""
        
        if GEMINI_API_KEY and chunks:
            try:
                context_parts = []
                seen_papers = {}
                
                for chunk in chunks[:10]:
                    meta = chunk.metadata
                    paper_id = meta.get('paper_id', '')
                    
                    if paper_id and paper_id not in seen_papers:
                        seen_papers[paper_id] = {
                            'title': meta.get('title', 'Unknown')
                        }
                        
                        context_parts.append(
                            f"[{paper_id}] {meta.get('title', 'Unknown')}\n"
                            f"Authors: {meta.get('authors', 'Unknown')}\n"
                            f"Year: {meta.get('year', 'Unknown')}\n"
                            f"Journal: {meta.get('journal', 'Unknown')}\n"
                            f"Content: {chunk.text[:700]}\n"
                        )
                
                reference_nodes = [
                    PaperReference(
                        paper_id=paper_id,
                        title=paper_info['title']
                    )
                    for paper_id, paper_info in seen_papers.items()
                ]
                context = "\n---\n".join(context_parts)
                
                prompt = f"""You are a specialized space biology research assistant. Analyze the scientific papers below and provide a comprehensive summary about "{query_request.query}".

            INSTRUCTIONS:
            1. Write 2-3 clear paragraphs synthesizing the key findings
            2. Focus on: main discoveries, biological mechanisms, experimental methods, and implications
            3. When citing findings, reference the paper ID in brackets like [PMC1234567]
            4. Compare and contrast findings across different papers when relevant
            5. Use precise scientific language but remain accessible
            6. Only state what the papers explicitly show - do not speculate beyond their findings

            SCIENTIFIC PAPERS:
            {context}

            SUMMARY:"""
                        
                model = genai.GenerativeModel('models/gemma-3-12b-it')
                response = model.generate_content(prompt)
                content = response.text
                
                logger.info(f"Generated summary with {len(reference_nodes)} papers")
                
            except Exception as e:
                logger.error(f"AI summarization failed: {e}")
                content = f"Search completed successfully but AI summary generation failed. Found {len(chunks)} results."
        else:
            content = f"Found {len(chunks)} results. AI summarization not available."
        
        return QueryResponse(
            content=content,
            referenceNodes=reference_nodes
        )
    
    except httpx.TimeoutException:
        logger.error("Database API timeout")
        raise HTTPException(status_code=504, detail="Database query timeout")
    
    except httpx.HTTPStatusError as e:
        logger.error(f"Database API error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Database error")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats")
@limiter.limit("10/minute")  # Lower limit for stats
async def get_statistics(
    request: Request,
    x_app_id: str = Header(None)
):
    """Get database statistics - rate limited"""
    verify_app_id(x_app_id)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DATABASE_API_URL}/stats",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/papers/{paper_id}")
@limiter.limit("30/minute")  # Higher limit for individual paper lookups
async def get_paper(
    request: Request,
    paper_id: str,
    x_app_id: str = Header(None)
):
    """Get specific paper by ID"""
    verify_app_id(x_app_id)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DATABASE_API_URL}/papers/{paper_id}",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Paper not found")
        raise HTTPException(status_code=500, detail="Error retrieving paper")
    
    except Exception as e:
        logger.error(f"Get paper failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve paper")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
