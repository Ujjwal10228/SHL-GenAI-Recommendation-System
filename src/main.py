# src/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FAISS_INDEX_PATH, EMBEDDINGS_META_PATH, CATALOG_PATH,
    DEFAULT_TOP_K, MAX_RECOMMENDATIONS, DEBUG, API_HOST, API_PORT
)
from models.schemas import (
    HealthResponse, RecommendRequest, RecommendResponse, AssessmentRecommendation
)
from recommender.embedder import Embedder
from recommender.indexer import CatalogIndex
from recommender.retrieval import RecommenderService
from recommender.recommend import RecommendationEngine

# Logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Intelligent recommendation system for SHL assessment catalog",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons - lazy loaded
_embedder = None
_index = None
_service = None
_engine = None

def get_engine():
    """Lazy load engine on first use."""
    global _embedder, _index, _service, _engine
    
    if _engine is not None:
        return _engine
    
    logger.info("Initializing recommendation engine...")
    
    # Check if index files exist
    if not FAISS_INDEX_PATH.exists() or not EMBEDDINGS_META_PATH.exists():
        logger.error(
            f"Index files not found. Expected:\n"
            f"  - {FAISS_INDEX_PATH}\n"
            f"  - {EMBEDDINGS_META_PATH}\n"
            f"Please build the index first by running:\n"
            f"  python -m src.evaluation.build_index"
        )
        raise FileNotFoundError("Index files not built. See logs for instructions.")
    
    _embedder = Embedder()
    _index = CatalogIndex(
        faiss_path=FAISS_INDEX_PATH,
        meta_path=EMBEDDINGS_META_PATH,
        embedder=_embedder,
        data_dir=CATALOG_PATH.parent
    )
    _index.load()
    _service = RecommenderService(index=_index, embedder=_embedder)
    _engine = RecommendationEngine(service=_service)
    
    logger.info("Recommendation engine initialized successfully")
    return _engine

@app.on_event("startup")
async def startup_event():
    """Initialize engine on app startup."""
    try:
        get_engine()
        logger.info("✓ Engine loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load engine: {e}")

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health():
    """
    Check if API is running and index is loaded.
    
    Returns:
        HealthResponse with status
    """
    try:
        engine = get_engine()
        return HealthResponse(
            status="ok",
            message="API is running and index is loaded"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"API error: {str(e)}"
        )

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Recommendations"],
    summary="Get assessment recommendations"
)
async def recommend(payload: RecommendRequest):
    """
    Get recommended SHL assessments based on query or job description.
    
    Args:
        payload: RecommendRequest with query or jd_url
        
    Returns:
        RecommendResponse with list of recommended assessments
        
    Raises:
        HTTPException: If neither query nor jd_url provided or processing fails
    """
    try:
        # Validate input
        if not payload.query and not payload.jd_url:
            raise HTTPException(
                status_code=400,
                detail="Either 'query' or 'jd_url' must be provided"
            )
        
        logger.info(f"Recommend request: query={'Yes' if payload.query else 'No'}, jd_url={'Yes' if payload.jd_url else 'No'}")
        
        # Get recommendations
        engine = get_engine()
        results = engine.recommend(
            query=payload.query,
            jd_url=payload.jd_url,
            top_k=DEFAULT_TOP_K
        )
        
        # Ensure between 5-10 recommendations
        results = results[:MAX_RECOMMENDATIONS]
        if len(results) == 0:
            logger.warning("No recommendations found")
            return RecommendResponse(results=[], count=0)
        
        # Convert to response format
        recommendations = [
            AssessmentRecommendation(
                assessment_name=r["assessment_name"],
                assessment_url=r["assessment_url"],
                test_type=r.get("test_type"),
                duration_minutes=r.get("duration_minutes"),
                category=r.get("category"),
            )
            for r in results
        ]
        
        response = RecommendResponse(
            results=recommendations,
            query_processed=payload.query or "JD URL provided",
            count=len(recommendations)
        )
        
        logger.info(f"Returned {len(recommendations)} recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation processing failed: {str(e)}"
        )

@app.get("/", tags=["Root"])
async def root():
    """API information."""
    return {
        "api": "SHL Assessment Recommendation Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    )
