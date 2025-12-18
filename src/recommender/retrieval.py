# src/recommender/retrieval.py
from typing import List, Dict
import logging
from src.utils.jd_utils import fetch_jd_from_url

logger = logging.getLogger(__name__)

class RecommenderService:
    """Orchestrates retrieval from catalog index."""
    
    def __init__(self, index, embedder):
        """
        Args:
            index: CatalogIndex instance
            embedder: Embedder instance
        """
        self.index = index
        self.embedder = embedder
        logger.info("RecommenderService initialized")
    
    def normalize_input(
        self,
        query_text: str = None,
        jd_url: str = None
    ) -> str:
        """
        Normalize input: combine query and fetched JD if both provided.
        
        Args:
            query_text: Natural language query
            jd_url: URL to job description
            
        Returns:
            Combined text for embedding
            
        Raises:
            ValueError: If neither provided
        """
        if jd_url:
            logger.info(f"Fetching JD from {jd_url}")
            jd_text = fetch_jd_from_url(jd_url)
            
            if query_text:
                combined = query_text.strip() + "\n\n" + jd_text
                logger.info(f"Combined query ({len(query_text)} chars) + JD ({len(jd_text)} chars)")
                return combined
            
            logger.info(f"Using fetched JD only ({len(jd_text)} chars)")
            return jd_text
        
        if query_text:
            logger.info(f"Using query text ({len(query_text)} chars)")
            return query_text.strip()
        
        raise ValueError("Either 'query_text' or 'jd_url' must be provided")
    
    def retrieve_candidates(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve top-k candidates from index.
        
        Args:
            query: Normalized query text
            top_k: Number of candidates to retrieve
            
        Returns:
            List of candidates with retrieval_score
        """
        logger.info(f"Embedding query ({len(query)} chars)")
        q_vec = self.embedder.embed_text(query)
        
        logger.info(f"Searching for top {top_k} candidates")
        scores, candidates = self.index.search(q_vec, top_k=top_k)
        
        # Attach scores
        for score, candidate in zip(scores, candidates):
            candidate["retrieval_score"] = float(score)
        
        logger.info(f"Retrieved {len(candidates)} candidates")
        return candidates
