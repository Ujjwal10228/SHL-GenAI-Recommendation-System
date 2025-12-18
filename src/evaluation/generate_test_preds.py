# src/evaluation/generate_test_preds.py
"""
Generate predictions on the test set and save as CSV in submission format.

Format:
  Query,Assessment_url
  Query 1,URL1
  Query 1,URL2
  ...
  Query 2,URL1
  ...
"""

import pandas as pd
import csv
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

def main():
    """Generate test predictions."""
    logger.info("=" * 80)
    logger.info("TEST SET PREDICTION GENERATION")
    logger.info("=" * 80)
    
    # Load test data
    logger.info(f"Loading test data from {TRAIN_DATA_PATH}")
    df = pd.read_excel(TRAIN_DATA_PATH, sheet_name="Test-Set")
    test_queries = df["Query"].tolist()
    logger.info(f"Loaded {len(test_queries)} test queries")
    
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
    logger.info("\nGenerating predictions...")
    rows = []
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"[{i}/{len(test_queries)}] Query: {query[:60]}...")
        
        try:
            recs = engine.recommend(query=query, jd_url=None, top_k=10)
            
            if not recs:
                logger.warning(f"  ⚠ No recommendations for this query")
            else:
                logger.info(f"  ✓ Generated {len(recs)} recommendations")
            
            for rec in recs:
                rows.append({
                    "Query": query,
                    "Assessment_url": rec["assessment_url"]
                })
                
        except Exception as e:
            logger.error(f"  ✗ Error processing query: {e}")
    
    # Save CSV in submission format
    output_path = Path(__file__).parent.parent.parent / "submission_predictions.csv"
    
    logger.info(f"\nSaving predictions to {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    logger.info(f"✓ Saved {len(rows)} prediction rows")
    logger.info("=" * 80)
    
    # Summary
    unique_queries = len(set(r["Query"] for r in rows))
    logger.info(f"\nSummary:")
    logger.info(f"  Total rows: {len(rows)}")
    logger.info(f"  Unique queries: {unique_queries}")
    logger.info(f"  Avg recommendations per query: {len(rows) / unique_queries if unique_queries > 0 else 0:.2f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
