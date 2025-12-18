# src/utils/jd_utils.py
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def fetch_jd_from_url(url: str, timeout: int = 20) -> str:
    """
    Fetch job description from a URL and extract main text content.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text from the page
        
    Raises:
        requests.RequestException: If fetch fails
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=" ", strip=True)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        logger.error(f"Failed to fetch JD from {url}: {e}")
        raise

def extract_duration_constraint(text: str) -> Optional[int]:
    """
    Extract maximum duration constraint from query text.
    
    Examples:
        "40 minutes" -> 40
        "90 mins" -> 90
        "1 hour" -> 60
        "1.5 hours" -> 90
        
    Args:
        text: Query text
        
    Returns:
        Duration in minutes, or None if not found
    """
    # Pattern: "X minutes/mins" or "X hours"
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*hours?', 60),  # hours to minutes
        (r'(\d+(?:\.\d+)?)\s*mins?', 1),    # minutes as-is
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return int(value * multiplier)
    
    return None

def extract_keywords(text: str) -> dict:
    """
    Extract key signals from query for domain balancing.
    
    Returns dict with:
        - wants_technical: bool
        - wants_behavioral: bool
        - wants_cognitive: bool
    """
    text_lower = text.lower()
    
    technical_keywords = [
        'java', 'python', 'sql', 'javascript', 'react', 'developer', 'engineer',
        'frontend', 'backend', 'coding', 'automation', 'selenium', 'testing',
        'qa', 'technical', 'programming', 'knowledge', 'skill'
    ]
    
    behavioral_keywords = [
        'collaborate', 'communication', 'personality', 'leadership', 'team',
        'interpersonal', 'behavioral', 'soft skills', 'culture fit', 'profile',
        'leadership', 'management'
    ]
    
    cognitive_keywords = [
        'cognitive', 'reasoning', 'analytical', 'logical', 'problem solving',
        'iq', 'aptitude', 'numeracy', 'verbal'
    ]
    
    wants_technical = any(kw in text_lower for kw in technical_keywords)
    wants_behavioral = any(kw in text_lower for kw in behavioral_keywords)
    wants_cognitive = any(kw in text_lower for kw in cognitive_keywords)
    
    return {
        'wants_technical': wants_technical,
        'wants_behavioral': wants_behavioral,
        'wants_cognitive': wants_cognitive
    }

def categorize_test_type(test_type: str) -> str:
    """Map test type codes to categories."""
    mapping = {
        'K': 'technical',      # Knowledge & Skills
        'P': 'behavioral',     # Personality & Behavior
        'C': 'cognitive',      # Cognitive ability
        'L': 'behavioral',     # Leadership
        'V': 'technical',      # Verbal ability
        'N': 'cognitive',      # Numerical
        'R': 'cognitive',      # Reasoning
    }
    return mapping.get(test_type, 'other')
