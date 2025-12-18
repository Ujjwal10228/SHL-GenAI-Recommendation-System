# src/evaluation/metrics.py
"""
Evaluation metrics: Recall@K and Mean Recall@K
"""

from collections import defaultdict
from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)

def recall_at_k(relevant: Set[str], predicted: List[str], k: int = 10) -> float:
    """
    Calculate Recall@K for a single query.
    
    Recall@K = (# of relevant items in top K) / (total # of relevant items)
    
    Args:
        relevant: Set of relevant assessment URLs
        predicted: List of predicted URLs (ordered by rank)
        k: Cutoff point
        
    Returns:
        Recall value between 0 and 1
    """
    if not relevant:
        return 0.0
    
    top_k = set(predicted[:k])
    hits = len(relevant & top_k)
    return hits / len(relevant)

def mean_recall_at_k(
    relevant_by_query: Dict[str, Set[str]],
    predicted_by_query: Dict[str, List[str]],
    k: int = 10
) -> float:
    """
    Calculate Mean Recall@K across all queries.
    
    Args:
        relevant_by_query: Dict mapping query -> set of relevant URLs
        predicted_by_query: Dict mapping query -> list of predicted URLs
        k: Cutoff point
        
    Returns:
        Mean recall value
    """
    scores = []
    
    for query, relevant_urls in relevant_by_query.items():
        predicted_urls = predicted_by_query.get(query, [])
        score = recall_at_k(relevant_urls, predicted_urls, k)
        scores.append(score)
        logger.debug(f"Query: {query[:50]}... | Recall@{k}: {score:.4f}")
    
    if not scores:
        return 0.0
    
    mean_score = sum(scores) / len(scores)
    return mean_score

def precision_at_k(relevant: Set[str], predicted: List[str], k: int = 10) -> float:
    """
    Calculate Precision@K for a single query.
    
    Precision@K = (# of relevant items in top K) / K
    
    Args:
        relevant: Set of relevant assessment URLs
        predicted: List of predicted URLs
        k: Cutoff point
        
    Returns:
        Precision value between 0 and 1
    """
    if k == 0:
        return 0.0
    
    top_k = set(predicted[:k])
    hits = len(relevant & top_k)
    return hits / k

def mean_precision_at_k(
    relevant_by_query: Dict[str, Set[str]],
    predicted_by_query: Dict[str, List[str]],
    k: int = 10
) -> float:
    """
    Calculate Mean Precision@K across all queries.
    
    Args:
        relevant_by_query: Dict mapping query -> set of relevant URLs
        predicted_by_query: Dict mapping query -> list of predicted URLs
        k: Cutoff point
        
    Returns:
        Mean precision value
    """
    scores = []
    
    for query, relevant_urls in relevant_by_query.items():
        predicted_urls = predicted_by_query.get(query, [])
        score = precision_at_k(relevant_urls, predicted_urls, k)
        scores.append(score)
    
    if not scores:
        return 0.0
    
    return sum(scores) / len(scores)
