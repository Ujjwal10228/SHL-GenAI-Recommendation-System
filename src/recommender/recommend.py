# src/recommender/recommend.py
from typing import List, Dict
import logging
from unittest import result
from . import rerank
from . import retrieval

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """End-to-end recommendation pipeline."""
    
    def __init__(self, service: retrieval.RecommenderService):
        """
        Args:
            service: RecommenderService instance
        """
        self.service = service
        logger.info("RecommendationEngine initialized")
    
    def recommend(
        self,
        query: str = None,
        jd_url: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Full recommendation pipeline: normalize -> retrieve -> rerank.
        
        Args:
            query: Natural language query
            jd_url: URL to job description
            top_k: Number of final recommendations (5-10)
            
        Returns:
            List of recommended assessments with name and URL
        """
        logger.info(f"Recommendation request: query={bool(query)}, jd_url={bool(jd_url)}, top_k={top_k}")
        
        # Step 1: Normalize input
        text = self.service.normalize_input(query_text=query, jd_url=jd_url)
        
        # Step 2: Retrieve candidates (get more than final k to allow reranking)
        candidates = self.service.retrieve_candidates(text, top_k=50)
        
        if not candidates:
            logger.warning("No candidates retrieved")
            return []
        
        # Step 3: Rerank with heuristics
        logger.info("Applying reranking heuristics")
        ranked = rerank.rerank(text, candidates, k=top_k)
        
        # Step 4: Format output
        # result = [
        #     {
        #         "assessment_name": c.get("name", "Unknown"),
        #         "assessment_url": c.get("url", ""),
        #         "test_type": c.get("test_type"),
        #         "duration_minutes": c.get("duration_minutes"),
        #         "category": c.get("category"),
        #     }
        #     for c in ranked
        #     if c.get("url")  # Only include items with valid URLs
        # ]
        
        result = []

        for c in ranked:
            result.append(
                {
                    "assessment_name": c.get("name", "Unknown"),
                    "assessment_url": c.get("url", ""),  # empty for synthetic
                    "test_type": c.get("test_type"),
                    "duration_minutes": c.get("duration_minutes"),
                    "category": c.get("category"),
                    "synthetic": c.get("url") is None
                }
        )

        
        logger.info(f"Returning {len(result)} recommendations")
        return result
