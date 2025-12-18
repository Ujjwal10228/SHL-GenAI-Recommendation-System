# src/config.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "src" / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API Config
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Embedding Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIM = 384  # Dimension for all-mpnet-base-v2

# Recommendation Config
DEFAULT_TOP_K = 10
MIN_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 10

# Index Paths
FAISS_INDEX_PATH = DATA_DIR / "embeddings.faiss"
EMBEDDINGS_META_PATH = DATA_DIR / "embeddings_meta.json"
CATALOG_PATH = DATA_DIR / "catalog_clean.csv"
TRAIN_DATA_PATH = DATA_DIR / "Gen_AI Dataset.xlsx"

# SHL Crawling Config
SHL_BASE_URL = "https://www.shl.com"
SHL_CATALOG_URL = f"{SHL_BASE_URL}/solutions/products/product-catalog/"
CRAWL_TIMEOUT = 20
CRAWL_DELAY = 0.5  # seconds between requests

# Evaluation Config
RECALL_K = 10
