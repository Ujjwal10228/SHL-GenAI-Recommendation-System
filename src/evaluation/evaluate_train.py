# src/evaluation/evaluate_train.py
"""
Evaluate recommendation system on train set.

Computes Mean Recall@10 and Mean Precision@10 on the provided labeled train data.
"""

import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config import TRAIN_DATA_PATH, FAISS_INDEX_PATH, EMBEDDINGS_META_PATH
from recommender.embedder import Embedder
from recommender.indexer import CatalogIndex
from recommender.retrieval import RecommenderService
from recommender.recommend import RecommendationEngine
from evaluation.metrics import mean_recall_at_k, mean_precision_at_k

def main():
    """Main evaluation logic."""
    logger.info("=" * 80)
    logger.info("TRAIN SET EVALUATION")
    logger.info("=" * 80)
    
    # Load train data
    logger.info(f"Loading train data from {TRAIN_DATA_PATH}")
    df = pd.read_excel(TRAIN_DATA_PATH, sheet_name="Train-Set")
    logger.info(f"Loaded {len(df)} rows")
    
    # Group by query
    rel_by_query = defaultdict(set)
    for _, row in df.iterrows():
        query = str(row["Query"]).strip()
        url = str(row["Assessment_url"]).strip()
        rel_by_query[query].add(url)
    
    logger.info(f"Found {len(rel_by_query)} unique queries")
    for query, urls in rel_by_query.items():
        logger.info(f"  Query: {query[:70]}...")
        logger.info(f"    Relevant assessments: {len(urls)}")
    
    # Initialize engine
    logger.info("\nInitializing recommendation engine...")
    embedder = Embedder()
    index = CatalogIndex(
        faiss_path=FAISS_INDEX_PATH,
        meta_path=EMBEDDINGS_META_PATH,
        embedder=embedder
    )
    index.load()
    service = RecommenderService(index=index, embedder=embedder)
    engine = RecommendationEngine(service=service)
    logger.info("✓ Engine ready")
    
    # Generate predictions
    logger.info("\nGenerating predictions on train set...")
    pred_by_query = {}
    
    for i, query in enumerate(rel_by_query.keys(), 1):
        logger.info(f"[{i}/{len(rel_by_query)}] Processing query: {query[:60]}...")
        
        try:
            recs = engine.recommend(query=query, jd_url=None, top_k=10)
            urls = [r["assessment_url"] for r in recs]
            pred_by_query[query] = urls
            logger.info(f"  → Generated {len(urls)} recommendations")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            pred_by_query[query] = []
    
    # Compute metrics
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    mr10 = mean_recall_at_k(rel_by_query, pred_by_query, k=10)
    mp10 = mean_precision_at_k(rel_by_query, pred_by_query, k=10)
    
    logger.info(f"\nMean Recall@10:    {mr10:.4f}")
    logger.info(f"Mean Precision@10: {mp10:.4f}")
    
    # Per-query breakdown
    logger.info("\nPer-query metrics:")
    for query in rel_by_query.keys():
        relevant = rel_by_query[query]
        predicted = pred_by_query.get(query, [])
        
        from evaluation.metrics import recall_at_k, precision_at_k
        r10 = recall_at_k(relevant, predicted, k=10)
        p10 = precision_at_k(relevant, predicted, k=10)
        
        logger.info(f"  {query[:50]}...")
        logger.info(f"    R@10: {r10:.4f}, P@10: {p10:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Evaluation complete. MR@10 = {mr10:.4f}")
    logger.info("=" * 80)
    
    return mr10, mp10

if __name__ == "__main__":
    main()
