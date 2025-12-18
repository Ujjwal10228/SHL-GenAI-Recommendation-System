# src/crawling/crawl_catalog.py
"""
Crawl SHL assessment catalog to build product database.

This script scrapes https://www.shl.com/solutions/products/product-catalog/
to extract 377+ individual test solutions (excluding pre-packaged job solutions).

Output: src/data/catalog_clean.csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import logging
from pathlib import Path
from typing import List, Set, Dict
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

time.sleep(2)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

session = requests.Session()
session.headers.update(HEADERS)

retries = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[403, 429, 500, 502, 503],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
DATA_DIR = Path(__file__).parent.parent / "data"

def get_catalog_pages() -> List[str]:
    """
    Get all catalog page URLs.
    
    Returns:
        List of page URLs
    """
    pages = [CATALOG_URL]
    
    # Try to find paginated links
    try:
        # resp = requests.get(CATALOG_URL, timeout=20)
        resp = session.get(CATALOG_URL, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Look for pagination links (this might vary by SHL site structure)
        pagination = soup.find("nav", class_=re.compile("pagination", re.I))
        if pagination:
            links = pagination.find_all("a")
            for link in links:
                href = link.get("href")
                if href and href not in pages:
                    if not href.startswith("http"):
                        href = BASE_URL + href
                    pages.append(href)
    except Exception as e:
        logger.warning(f"Could not find pagination: {e}")
    
    return pages

def parse_product_list_page(url: str) -> Set[str]:
    """
    Extract product URLs from a catalog listing page.
    
    Returns:
        Set of product detail URLs
    """
    product_urls = set()
    
    try:
        # resp = requests.get(url, timeout=20)
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find product cards/links
        # This selector needs to match the actual SHL site structure
        product_links = soup.find_all("a", href=re.compile(r"/solutions/products/product-catalog/view/"))
        
        for link in product_links:
            href = link.get("href")
            if href:
                if not href.startswith("http"):
                    href = BASE_URL + href
                product_urls.add(href)
        
        logger.info(f"Found {len(product_urls)} products on {url}")
        
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
    
    return product_urls

def parse_product_detail(url: str) -> Dict:
    """
    Parse product detail page to extract metadata.
    
    Returns:
        Dict with: name, url, test_type, duration_minutes, category, description, tags, text_blob
    """
    try:
        # resp = requests.get(url, timeout=20)
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract product name
        name_elem = soup.find("h1")
        name = name_elem.get_text(strip=True) if name_elem else "Unknown"
        
        # Extract duration (look for patterns like "40 minutes", "1 hour", etc.)
        duration_minutes = None
        text = soup.get_text()
        duration_match = re.search(r'(\d+)\s*(?:minute|min|hour)s?', text, re.IGNORECASE)
        if duration_match:
            val = int(duration_match.group(1))
            # Assume it's minutes if < 100, else convert from hours
            duration_minutes = val if val <= 100 else val * 60
        
        # Extract test type from URL or metadata (K, P, C, etc.)
        test_type = None
        # Try to infer from page content
        if 'personality' in url.lower() or 'opq' in url.lower():
            test_type = 'P'
        elif 'knowledge' in url.lower() or 'skill' in url.lower():
            test_type = 'K'
        elif 'cognitive' in url.lower():
            test_type = 'C'
        
        # Extract category and description
        category_elem = soup.find("div", class_=re.compile("category", re.I))
        category = category_elem.get_text(strip=True) if category_elem else ""
        
        desc_elem = soup.find("p", class_=re.compile("description", re.I))
        description = desc_elem.get_text(strip=True) if desc_elem else ""
        
        # Extract tags
        tags = []
        tag_elems = soup.find_all("span", class_=re.compile("tag", re.I))
        for tag_elem in tag_elems:
            tag_text = tag_elem.get_text(strip=True)
            if tag_text:
                tags.append(tag_text)
        
        # Build text_blob for embedding
        text_blob = f"{name} {category} {' '.join(tags)} {description}".strip()
        
        return {
            'name': name,
            'url': url,
            'test_type': test_type,
            'duration_minutes': duration_minutes,
            'category': category,
            'description': description,
            'tags': ' | '.join(tags),
            'text_blob': text_blob
        }
        
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None

def main():
    """Main crawl logic."""
    logger.info("Starting SHL catalog crawl...")
    
    # Get all page URLs
    page_urls = get_catalog_pages()
    logger.info(f"Found {len(page_urls)} pages to crawl")
    
    # Collect all product URLs
    all_product_urls = set()
    for page_url in page_urls:
        urls = parse_product_list_page(page_url)
        all_product_urls.update(urls)
        time.sleep(0.5)
    
    logger.info(f"Total unique products found: {len(all_product_urls)}")
    
    # Parse each product
    rows = []
    for i, product_url in enumerate(all_product_urls, 1):
        logger.info(f"[{i}/{len(all_product_urls)}] Parsing {product_url}")
        data = parse_product_detail(product_url)
        if data:
            rows.append(data)
        time.sleep(0.5)
    
    # Save to CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "catalog_clean.csv"
    
    fieldnames = [
        'name', 'url', 'test_type', 'duration_minutes',
        'category', 'description', 'tags', 'text_blob'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    logger.info(f"✓ Crawl complete. Saved {len(rows)} products to {output_path}")
    
    if len(rows) < 377:
        logger.warning(f"⚠ Only found {len(rows)} products (requirement: 377+)")
    else:
        logger.info(f"✓ Met requirement of 377+ products")

if __name__ == "__main__":
    main()
