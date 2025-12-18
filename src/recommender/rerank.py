# src/recommender/rerank.py
import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def infer_max_duration(query: str) -> Optional[int]:
    """
    Extract maximum duration constraint from query text.
    
    Returns:
        Duration in minutes or None
    """
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*hours?', 60),
        (r'(\d+(?:\.\d+)?)\s*mins?', 1),
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return int(value * multiplier)
    
    return None

def infer_desired_domains(query: str) -> dict:
    """
    Infer which assessment domains are desired.
    
    Returns:
        Dict with keys: wants_technical, wants_behavioral, wants_cognitive
    """
    query_lower = query.lower()
    
    technical_kw = [
        'java', 'python', 'sql', 'javascript', 'react', 'developer', 'engineer',
        'frontend', 'backend', 'coding', 'automation', 'testing', 'qa', 'technical',
        'programming', 'knowledge', 'skill', 'proficient', 'expertise', 'expertise'
    ]
    
    behavioral_kw = [
        'collaborate', 'communication', 'personality', 'leadership', 'team',
        'interpersonal', 'behavioral', 'soft skill', 'culture', 'fit', 'manager',
        'management'
    ]
    
    cognitive_kw = [
        'cognitive', 'reasoning', 'analytical', 'logical', 'problem', 'iq',
        'aptitude', 'numeracy', 'verbal', 'ability'
    ]
    
    wants_technical = any(kw in query_lower for kw in technical_kw)
    wants_behavioral = any(kw in query_lower for kw in behavioral_kw)
    wants_cognitive = any(kw in query_lower for kw in cognitive_kw)
    
    return {
        'wants_technical': wants_technical,
        'wants_behavioral': wants_behavioral,
        'wants_cognitive': wants_cognitive
    }

def categorize_test_type(test_type: str) -> str:
    """Map test type code to domain."""
    if not test_type:
        return 'other'
    
    test_type = test_type.upper().strip()
    mapping = {
        'K': 'technical',
        'P': 'behavioral',
        'C': 'cognitive',
        'L': 'behavioral',
        'V': 'technical',
        'N': 'cognitive',
        'R': 'cognitive',
    }
    return mapping.get(test_type, 'other')

def apply_duration_filter(
    candidates: List[Dict],
    max_minutes: Optional[int]
) -> List[Dict]:
    """Filter candidates by duration constraint."""
    if max_minutes is None:
        return candidates
    
    filtered = []
    for c in candidates:
        dur = c.get("duration_minutes")
        if dur is None or dur <= max_minutes:
            filtered.append(c)
    
    return filtered

def balance_by_domains(
    candidates: List[Dict],
    desired_domains: dict,
    k: int
) -> List[Dict]:
    """
    Balance recommendations across desired domains.
    
    If query mentions both technical and behavioral, try to split k evenly.
    """
    wants_tech = desired_domains.get('wants_technical', False)
    wants_behav = desired_domains.get('wants_behavioral', False)
    wants_cog = desired_domains.get('wants_cognitive', False)
    
    if not any([wants_tech, wants_behav, wants_cog]):
        return candidates[:k]
    
    # Categorize candidates
    tech_candidates = []
    behav_candidates = []
    cog_candidates = []
    other_candidates = []
    
    for c in candidates:
        test_type = c.get("test_type", "")
        domain = categorize_test_type(test_type)
        
        if domain == 'technical':
            tech_candidates.append(c)
        elif domain == 'behavioral':
            behav_candidates.append(c)
        elif domain == 'cognitive':
            cog_candidates.append(c)
        else:
            other_candidates.append(c)
    
    # Allocate slots
    num_domains = sum([wants_tech, wants_behav, wants_cog])
    
    if num_domains == 0:
        return candidates[:k]
    
    slots_per_domain = max(1, k // num_domains)
    result = []
    
    if wants_tech:
        result.extend(tech_candidates[:slots_per_domain])
    
    if wants_behav:
        result.extend(behav_candidates[:slots_per_domain])
    
    if wants_cog:
        result.extend(cog_candidates[:slots_per_domain])
    
    # Fill remaining slots with any candidates
    remaining = k - len(result)
    if remaining > 0:
        all_remaining = [
            c for c in candidates if c not in result
        ]
        result.extend(all_remaining[:remaining])
    
    return result[:k]

def rerank(query: str, candidates: List[Dict], k: int = 10) -> List[Dict]:
    """
    Rerank candidates applying heuristics:
    - Duration filtering
    - Domain balancing
    
    Args:
        query: Original query text
        candidates: List of candidate assessments with scores
        k: Target number of results
        
    Returns:
        Reranked list of top k assessments
    """
    # Extract constraints
    max_duration = infer_max_duration(query)
    domains = infer_desired_domains(query)
    
    # Apply filters
    logger.info(f"Max duration constraint: {max_duration} minutes")
    logger.info(f"Desired domains: {domains}")
    
    filtered = apply_duration_filter(candidates, max_duration)
    logger.info(f"After duration filter: {len(filtered)} candidates")
    
    balanced = balance_by_domains(filtered, domains, k)
    logger.info(f"After domain balancing: {len(balanced)} results")
    
    return balanced[:k]
