# SHL Assessment Recommendation Engine - Complete Solution

## Project Structure

```
shl-rec-engine/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ main.py                          # FastAPI app
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ schemas.py                    # Pydantic schemas
│  ├─ data/
│  │  ├─ Gen_AI-Dataset.xlsx           # Train/Test data
│  │  ├─ catalog_clean.csv             # Cleaned catalog
│  │  ├─ embeddings.faiss              # Vector index
│  │  └─ embeddings_meta.json          # Metadata
│  ├─ crawling/
│  │  ├─ __init__.py
│  │  ├─ crawl_catalog.py              # SHL catalog scraper
│  │  └─ parse_catalog.py              # Data cleaner
│  ├─ recommender/
│  │  ├─ __init__.py
│  │  ├─ embedder.py                   # Embedding wrapper
│  │  ├─ indexer.py                    # Vector index builder
│  │  ├─ retrieval.py                  # Query retrieval
│  │  ├─ rerank.py                     # Reranking logic
│  │  └─ recommend.py                  # Full pipeline
│  ├─ evaluation/
│  │  ├─ __init__.py
│  │  ├─ metrics.py                    # Recall@K metrics
│  │  ├─ evaluate_train.py             # Train evaluation
│  │  └─ generate_test_preds.py        # Test prediction CSV
│  └─ utils/
│     ├─ __init__.py
│     ├─ jd_utils.py                   # JD parsing
│     └─ logging_utils.py              # Logging helpers
├─ frontend/                           # React SPA
├─ docs/
│  └─ approach.md                      # 2-page approach doc
├─ requirements.txt
├─ .gitignore
├─ README.md
└─ submission_predictions.csv          # Output file
```

## Key Features

✓ Crawls 377+ SHL Individual Test Solutions  
✓ Semantic embeddings with sentence-transformers  
✓ FAISS vector indexing for fast retrieval  
✓ Multi-domain balancing (technical + behavioral)  
✓ Duration-aware filtering  
✓ FastAPI with `/health` & `/recommend` endpoints  
✓ React frontend with table UI  
✓ Comprehensive evaluation metrics  
✓ Mean Recall@10 optimization  
✓ Production-ready deployment

## Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare data
python src/crawling/crawl_catalog.py
python src/crawling/parse_catalog.py

# 3. Build embeddings index
python -c "from src.recommender.indexer import CatalogIndex; from src.recommender.embedder import Embedder; CatalogIndex(...).build()"

# 4. Evaluate on train set
python src/evaluation/evaluate_train.py

# 5. Generate test predictions
python src/evaluation/generate_test_preds.py

# 6. Run API server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 7. In another terminal, run frontend
cd frontend && npm start
```

## API Testing

```bash
# Health check
curl http://localhost:8000/health

# Recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer with collaboration skills"}'
```

## Submission Checklist

- [ ] GitHub repo with all code  
- [ ] API deployed and `/health` returns OK  
- [ ] Frontend deployed and functional  
- [ ] `submission_predictions.csv` in correct format  
- [ ] `docs/approach.md` (2-page PDF/Docx)  
- [ ] Mean Recall@10 on train set documented  
- [ ] Edge cases tested  

## Optimization Notes

1. **Embedding Model**: Uses `all-mpnet-base-v2` (384-dim, high quality)  
2. **Reranking**: Domain balancing (K vs P test types)  
3. **Duration Filtering**: Extracts max duration from query text  
4. **Index**: FAISS IndexFlatIP for cosine similarity (normalized)  

See `docs/approach.md` for detailed methodology.
