# src/recommender/indexer.py
import faiss
import numpy as np
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class CatalogIndex:
    """FAISS vector index for assessment catalog."""
    
    def __init__(self, faiss_path: Path, meta_path: Path, embedder, data_dir: Path = None):
        """
        Initialize index.
        
        Args:
            faiss_path: Path to save/load FAISS index
            meta_path: Path to save/load metadata JSON
            embedder: Embedder instance
            data_dir: Directory containing catalog_clean.csv
        """
        self.faiss_path = faiss_path
        self.meta_path = meta_path
        self.embedder = embedder
        self.data_dir = data_dir or Path("src/data")
        self.index = None
        self.meta = None
        logger.info("CatalogIndex initialized")
    
    def build(self, catalog_path: Path = None, force: bool = False):
        """
        Build index from catalog CSV.
        
        Args:
            catalog_path: Path to catalog_clean.csv
            force: Rebuild even if index exists
        """
        if not force and self.faiss_path.exists() and self.meta_path.exists():
            logger.info("Index already exists. Use force=True to rebuild.")
            return
        
        if catalog_path is None:
            catalog_path = self.data_dir / "catalog_clean.csv"
        
        logger.info(f"Building index from {catalog_path}")
        df = pd.read_csv(catalog_path)
        
        texts = df["text_blob"].fillna("").tolist()
        logger.info(f"Embedding {len(texts)} assessments...")
        embeddings = self.embedder.embed_texts(texts, show_progress=True)
        embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index with cosine similarity (IP on normalized vectors)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings)
        
        logger.info(f"Index created with {index.ntotal} items")
        faiss.write_index(index, str(self.faiss_path))
        logger.info(f"Index saved to {self.faiss_path}")
        
        # Save metadata
        meta = df.to_dict(orient="records")
        self.meta = meta
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata saved to {self.meta_path}")
        
        self.index = index
    
    def load(self):
        """Load index and metadata from disk."""
        if not self.faiss_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"Index files not found. Build index first.\n"
                f"Expected: {self.faiss_path}, {self.meta_path}"
            )
        
        logger.info(f"Loading index from {self.faiss_path}")
        self.index = faiss.read_index(str(self.faiss_path))
        
        with open(self.meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        
        logger.info(f"Index loaded. Total items: {self.index.ntotal}")
    
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 20
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Search index with query vector.
        
        Args:
            query_vec: Query embedding (1D array)
            top_k: Number of results to return
            
        Returns:
            Tuple of (scores, metadata_list)
        """
        if self.index is None:
            self.load()
        
        # Normalize query for cosine similarity
        q = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        
        scores, idxs = self.index.search(q, min(top_k, self.index.ntotal))
        
        results = [self.meta[i] for i in idxs[0] if i < len(self.meta)]
        return scores[0], results
    
    def get_size(self) -> int:
        """Get number of items in index."""
        if self.index is None:
            self.load()
        return self.index.ntotal
