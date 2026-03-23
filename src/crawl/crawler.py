"""
Web Crawler for AI Researchers Knowledge Graph Project
======================================================
Crawls Wikipedia, arXiv author pages, and research lab websites
to collect text about AI researchers.

Usage:
    python src/crawl/crawler.py
    python src/crawl/crawler.py --output data/my_crawl.jsonl --max-pages 100 --delay 2.0
"""

import argparse
import json
import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default output path (relative to project root)
DEFAULT_OUTPUT = Path("data/crawled_pages.jsonl")
MIN_WORD_COUNT = 500
DEFAULT_DELAY = 1.5          # seconds between requests
REQUEST_TIMEOUT = 20         # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0          # exponential backoff multiplier

# ---------------------------------------------------------------------------
# Seed URLs — Wikipedia pages and research profile pages for AI researchers
# ---------------------------------------------------------------------------

SEED_URLS = [
    # Wikipedia — individual researchers
    "https://en.wikipedia.org/wiki/Yann_LeCun",
    "https://en.wikipedia.org/wiki/Geoffrey_Hinton",
    "https://en.wikipedia.org/wiki/Yoshua_Bengio",
    "https://en.wikipedia.org/wiki/Andrew_Ng",
    "https://en.wikipedia.org/wiki/Fei-Fei_Li",
    "https://en.wikipedia.org/wiki/J%C3%BCrgen_Schmidhuber",
    "https://en.wikipedia.org/wiki/Ian_Goodfellow",
    "https://en.wikipedia.org/wiki/Ilya_Sutskever",
    "https://en.wikipedia.org/wiki/Demis_Hassabis",
    "https://en.wikipedia.org/wiki/Sam_Altman",
    "https://en.wikipedia.org/wiki/Pieter_Abbeel",
    "https://en.wikipedia.org/wiki/Judea_Pearl",
    "https://en.wikipedia.org/wiki/Stuart_Russell",
    "https://en.wikipedia.org/wiki/Peter_Norvig",
    "https://en.wikipedia.org/wiki/Michael_I._Jordan",
    "https://en.wikipedia.org/wiki/Bernhard_Sch%C3%B6lkopf",
    "https://en.wikipedia.org/wiki/Zoubin_Ghahramani",
    "https://en.wikipedia.org/wiki/Nando_de_Freitas",
    "https://en.wikipedia.org/wiki/David_Silver_(computer_scientist)",
    "https://en.wikipedia.org/wiki/Oriol_Vinyals",
    "https://en.wikipedia.org/wiki/Alex_Krizhevsky",
    "https://en.wikipedia.org/wiki/Samy_Bengio",
    "https://en.wikipedia.org/wiki/Hugo_Larochelle",
    "https://en.wikipedia.org/wiki/Kyunghyun_Cho",
    "https://en.wikipedia.org/wiki/Alec_Radford",
    # Wikipedia — organizations / topics
    "https://en.wikipedia.org/wiki/DeepMind",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/Meta_AI",
    "https://en.wikipedia.org/wiki/Montreal_Institute_for_Learning_Algorithms",
    "https://en.wikipedia.org/wiki/Google_Brain",
    "https://en.wikipedia.org/wiki/Turing_Award",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Generative_adversarial_network",
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "https://en.wikipedia.org/wiki/AlphaGo",
    "https://en.wikipedia.org/wiki/ImageNet",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    # arXiv author pages (abstracts only — light)
    "https://arxiv.org/search/?searchtype=author&query=LeCun%2C+Y",
    "https://arxiv.org/search/?searchtype=author&query=Bengio%2C+Y",
    "https://arxiv.org/search/?searchtype=author&query=Hinton%2C+G",
    # Research lab pages
    "https://ai.meta.com/research/",
    "https://deepmind.google/research/",
    "https://openai.com/research/",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CrawledPage:
    url: str
    text: str
    word_count: int
    timestamp: str
    title: str = ""
    domain: str = ""


# ---------------------------------------------------------------------------
# Robots.txt cache
# ---------------------------------------------------------------------------

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def can_fetch(url: str, user_agent: str = "*") -> bool:
    """Check robots.txt to see if we are allowed to fetch this URL."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{base}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
            _robots_cache[base] = rp
        except Exception:
            # If we cannot read robots.txt, assume allowed
            logger.debug(f"Could not read robots.txt for {base}, assuming allowed.")
            _robots_cache[base] = None
    rp = _robots_cache[base]
    if rp is None:
        return True
    return rp.can_fetch(user_agent, url)


# ---------------------------------------------------------------------------
# HTTP fetching with retry logic
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "AcademicCrawlerBot/1.0 (AI Researchers KG project; educational use; "
        "contact: student@university.edu)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
}


def fetch_url(client: httpx.Client, url: str) -> Optional[str]:
    """
    Fetch a URL with retry logic and exponential backoff.
    Returns raw HTML string or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.debug(f"Skipping non-HTML content at {url}: {content_type}")
                return None
            return response.text
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} for {url} (attempt {attempt}/{MAX_RETRIES})")
            if e.response.status_code in (403, 404, 410):
                return None  # No point retrying
        except httpx.TimeoutException:
            logger.warning(f"Timeout for {url} (attempt {attempt}/{MAX_RETRIES})")
        except httpx.RequestError as e:
            logger.warning(f"Request error for {url}: {e} (attempt {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            sleep_time = RETRY_BACKOFF ** attempt
            logger.debug(f"Waiting {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)
    return None


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def extract_text(html: str, url: str) -> tuple[str, str]:
    """
    Extract main text content from HTML using trafilatura.
    Returns (title, text) tuple.
    """
    # trafilatura extraction — returns None if extraction fails
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
    )
    if text is None:
        return "", ""

    # Extract title separately
    title = ""
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        # Clean Wikipedia-style titles
        title = re.sub(r"\s*[-–|]\s*Wikipedia.*$", "", title).strip()

    return title, text


def count_words(text: str) -> int:
    """Count approximate word count of a text string."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------


def crawl(
    seed_urls: list[str],
    output_path: Path,
    max_pages: int = 200,
    delay: float = DEFAULT_DELAY,
    min_words: int = MIN_WORD_COUNT,
) -> list[CrawledPage]:
    """
    Crawl the given seed URLs (no recursive following in this version).
    Saves results as JSONL to output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    failed = 0
    results: list[CrawledPage] = []

    logger.info(f"Starting crawl: {len(seed_urls)} seed URLs, max_pages={max_pages}, delay={delay}s")

    with httpx.Client(http2=False) as client:
        with open(output_path, "w", encoding="utf-8") as out_f:
            for url in tqdm(seed_urls[:max_pages], desc="Crawling", unit="page"):
                # Robots.txt check
                if not can_fetch(url):
                    logger.info(f"Robots.txt disallows: {url}")
                    skipped += 1
                    continue

                # Rate limiting
                time.sleep(delay)

                # Fetch page
                html = fetch_url(client, url)
                if html is None:
                    logger.warning(f"Failed to fetch: {url}")
                    failed += 1
                    continue

                # Extract text
                title, text = extract_text(html, url)
                if not text:
                    logger.debug(f"No text extracted from: {url}")
                    skipped += 1
                    continue

                word_count = count_words(text)
                if word_count < min_words:
                    logger.debug(f"Too short ({word_count} words): {url}")
                    skipped += 1
                    continue

                domain = urlparse(url).netloc
                page = CrawledPage(
                    url=url,
                    text=text,
                    word_count=word_count,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    title=title,
                    domain=domain,
                )
                results.append(page)

                # Write JSONL
                out_f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                out_f.flush()
                saved += 1

                logger.info(f"[{saved}] Saved: {url} ({word_count} words)")

    logger.info(
        f"\nCrawl complete: {saved} saved, {skipped} skipped, {failed} failed."
        f"\nOutput: {output_path}"
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Web Crawler for AI Researchers KG")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=len(SEED_URLS),
        help="Maximum number of pages to crawl",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORD_COUNT,
        help=f"Minimum word count to keep a page (default: {MIN_WORD_COUNT})",
    )
    args = parser.parse_args()

    results = crawl(
        seed_urls=SEED_URLS,
        output_path=args.output,
        max_pages=args.max_pages,
        delay=args.delay,
        min_words=args.min_words,
    )

    # Print a quick summary
    if results:
        word_counts = [p.word_count for p in results]
        print(f"\n=== Crawl Summary ===")
        print(f"Pages saved   : {len(results)}")
        print(f"Avg word count: {sum(word_counts)//len(word_counts)}")
        print(f"Max word count: {max(word_counts)}")
        print(f"Domains covered:")
        domains = set(p.domain for p in results)
        for d in sorted(domains):
            count = sum(1 for p in results if p.domain == d)
            print(f"  {d}: {count} page(s)")


if __name__ == "__main__":
    main()
