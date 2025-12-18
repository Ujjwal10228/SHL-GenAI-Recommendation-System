# src/evaluation/build_index.py
"""
Build FAISS index from catalog.

This script:
1. Loads catalog_clean.csv
2. Embeds all assessment text blobs
3. Creates FAISS index
4. Saves index and metadata

Run this ONCE after crawling, before starting API or evaluation.
"""

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

from config import CATALOG_PATH, FAISS_INDEX_PATH, EMBEDDINGS_META_PATH
from recommender.embedder import Embedder
from recommender.indexer import CatalogIndex

def main():
    """Build index."""
    logger.info("=" * 80)
    logger.info("BUILDING FAISS INDEX")
    logger.info("=" * 80)
    
    # Check catalog exists
    if not CATALOG_PATH.exists():
        logger.error(f"Catalog file not found: {CATALOG_PATH}")
        logger.error("Run crawl_catalog.py first")
        return False
    
    logger.info(f"Catalog path: {CATALOG_PATH}")
    logger.info(f"Index path: {FAISS_INDEX_PATH}")
    logger.info(f"Meta path: {EMBEDDINGS_META_PATH}")
    
    # Initialize embedder
    logger.info("\nInitializing embedding model...")
    embedder = Embedder()
    logger.info(f"✓ Model loaded: {embedder.model_name}")
    logger.info(f"  Embedding dimension: {embedder.embedding_dim}")
    
    # Build index
    logger.info("\nBuilding index...")
    index = CatalogIndex(
        faiss_path=FAISS_INDEX_PATH,
        meta_path=EMBEDDINGS_META_PATH,
        embedder=embedder
    )
    
    index.build(catalog_path=CATALOG_PATH, force=True)
    
    # Verify
    size = index.get_size()
    logger.info(f"\n✓ Index built successfully")
    logger.info(f"  Total items: {size}")
    
    if size < 377:
        logger.warning(f"⚠ Only {size} items in index (requirement: 377+)")
        return False
    
    logger.info("=" * 80)
    logger.info("✓ Index ready for use")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
