#!/usr/bin/env python3
"""
Semantic Scholar-Only Paper Download Script

For conferences NOT on OpenReview: AAAI, IJCAI, AAMAS
(NeurIPS already downloaded via OpenReview)

These conferences use CMT/EasyChair, so papers are indexed via Semantic Scholar instead.

Expected results:
- AAAI 2023: ~1,900 papers
- AAAI 2024: ~2,300 papers
- IJCAI 2023: ~650 papers
- IJCAI 2024: ~900 papers
- AAMAS 2024: ~900 papers
- AAMAS 2025: ~200 papers (may not be published yet)
Total: ~6,850 papers with 30-50% PDF coverage (~2,500-3,400 PDFs)

Usage:
    python download_papers_semantic_scholar.py

Requires:
    - SEMANTIC_SCHOLAR_API_KEY in .env file
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
EMAIL = os.getenv("EMAIL", "openevolvetesting.worrier295@passmail.net")
BASE_DIR = Path(__file__).parent / "data"
MAX_PDF_WORKERS = 5

# Conference configurations - using better search strategies
CONFERENCES = {
    "aaai": {
        "years": [2023, 2024],
        "full_name": "AAAI Conference on Artificial Intelligence",
        "venue_variations": [
            "AAAI",
            "Proceedings of the AAAI Conference on Artificial Intelligence",
        ],
    },
    "ijcai": {
        "years": [2023, 2024],
        "full_name": "International Joint Conference on Artificial Intelligence",
        "venue_variations": [
            "IJCAI",
            "International Joint Conference on Artificial Intelligence",
        ],
    },
    "aamas": {
        "years": [2024, 2025],
        "full_name": "International Conference on Autonomous Agents and Multiagent Systems",
        "venue_variations": [
            "AAMAS",
            "International Conference on Autonomous Agents and Multiagent Systems",
            "Proceedings of the International Conference on Autonomous Agents and Multiagent Systems",
        ],
    },
}

# Setup logging
BASE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR.parent / "download_semantic_scholar.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Paper metadata"""

    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    downloaded: bool = False


class RateLimiter:
    """Rate limiter with exponential backoff for 429 errors"""

    def __init__(self, base_delay: float = 1.1):
        self.base_delay = base_delay
        self.last_request = 0
        self.backoff_multiplier = 1.0
        self.consecutive_429s = 0

    def wait(self):
        """Wait with current backoff applied"""
        delay = self.base_delay * self.backoff_multiplier
        elapsed = time.time() - self.last_request

        if elapsed < delay:
            wait_time = delay - elapsed
            if wait_time > 2:
                logger.info(f"Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        self.last_request = time.time()

    def record_429(self):
        """Record a 429 error and increase backoff"""
        self.consecutive_429s += 1
        self.backoff_multiplier = min(2.0 ** self.consecutive_429s, 16.0)
        logger.warning(
            f"Rate limit hit (429). Backoff: {self.backoff_multiplier}x. "
            f"Waiting {60 * self.backoff_multiplier:.0f}s..."
        )
        time.sleep(60 * self.backoff_multiplier)

    def record_success(self):
        """Record successful request and gradually reduce backoff"""
        if self.consecutive_429s > 0:
            self.consecutive_429s = max(0, self.consecutive_429s - 1)
            self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.75)


class SemanticScholarClient:
    """Semantic Scholar API client with smart retry logic"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str]):
        if not api_key or api_key == "your_semantic_scholar_api_key_here":
            logger.error("SEMANTIC_SCHOLAR_API_KEY not set in .env file!")
            logger.error("Get one at: https://api.semanticscholar.org")
            sys.exit(1)

        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update(
            {"x-api-key": api_key, "User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"}
        )
        self.request_count = 0

    def search_papers(
        self, conference: str, venue_variations: List[str], year: int, max_results: int = 3000
    ) -> List[Paper]:
        """
        Search for papers using multiple strategies:
        1. Direct venue search
        2. Bulk fetch by year + post-filter
        """
        logger.info(f"Searching Semantic Scholar for {conference} {year}...")

        all_papers = []
        seen_titles = set()

        # Strategy: Search by conference name + year, then filter by venue
        search_queries = [
            f"{conference} {year}",
            f"{conference.upper()} {year}",
        ]

        for query in search_queries:
            papers = self._search_with_query(query, year, max_results // len(search_queries))

            # Filter by venue
            for paper in papers:
                if not paper.venue:
                    continue

                # Check if venue matches any variation
                venue_lower = paper.venue.lower()
                if any(v.lower() in venue_lower for v in venue_variations):
                    title_key = paper.title.lower().strip()
                    if title_key not in seen_titles and title_key:
                        seen_titles.add(title_key)
                        all_papers.append(paper)

            if len(all_papers) > 100:  # Got enough results
                break

        logger.info(f"Found {len(all_papers)} papers for {conference} {year}")
        return all_papers

    def _search_with_query(self, query: str, year: int, limit: int) -> List[Paper]:
        """Execute a single search query with pagination"""
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "year": year,
            "fields": "paperId,title,authors,year,venue,abstract,openAccessPdf,externalIds",
            "limit": 100,
            "offset": 0,
        }

        papers = []
        max_attempts = 3

        while params["offset"] < limit:
            attempt = 0
            while attempt < max_attempts:
                self.rate_limiter.wait()
                self.request_count += 1

                try:
                    response = self.session.get(url, params=params, timeout=30)

                    if response.status_code == 429:
                        self.rate_limiter.record_429()
                        attempt += 1
                        continue

                    response.raise_for_status()
                    self.rate_limiter.record_success()

                    data = response.json()
                    results = data.get("data", [])

                    if not results:
                        return papers

                    for item in results:
                        paper = self._parse_paper(item)
                        if paper:
                            papers.append(paper)

                    if len(results) < params["limit"]:
                        return papers

                    params["offset"] += len(results)
                    break

                except requests.RequestException as e:
                    logger.debug(f"Request error: {e}")
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts")
                        return papers
                    time.sleep(2)

        return papers

    def _parse_paper(self, data: dict) -> Optional[Paper]:
        """Parse API response into Paper object"""
        try:
            external_ids = data.get("externalIds", {})
            open_access_pdf = data.get("openAccessPdf")

            return Paper(
                paper_id=f"s2_{data.get('paperId', '')}",
                title=data.get("title", ""),
                authors=[a.get("name", "") for a in data.get("authors", [])],
                year=data.get("year", 0),
                venue=data.get("venue", ""),
                abstract=data.get("abstract", ""),
                pdf_url=open_access_pdf.get("url") if open_access_pdf else None,
                arxiv_id=external_ids.get("ArXiv"),
                doi=external_ids.get("DOI"),
            )
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None


class ArxivClient:
    """ArXiv client for fallback PDFs"""

    def get_pdf_url(self, arxiv_id: str) -> Optional[str]:
        if not arxiv_id:
            return None
        arxiv_id = arxiv_id.split("v")[0]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


class UnpaywallClient:
    """Unpaywall client for open access PDFs"""

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self):
        self.session = requests.Session()
        self.last_request = 0

    def get_pdf_url(self, doi: str) -> Optional[str]:
        if not doi:
            return None

        # Rate limit
        elapsed = time.time() - self.last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self.last_request = time.time()

        try:
            url = f"{self.BASE_URL}/{doi}"
            response = self.session.get(url, params={"email": EMAIL}, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("is_oa") and data.get("best_oa_location"):
                return data["best_oa_location"].get("url_for_pdf")
        except Exception as e:
            logger.debug(f"Unpaywall error for {doi}: {e}")

        return None


class PaperDownloader:
    """Main downloader"""

    def __init__(self):
        self.semantic_scholar = SemanticScholarClient(SEMANTIC_SCHOLAR_API_KEY)
        self.arxiv = ArxivClient()
        self.unpaywall = UnpaywallClient()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

        self.stats = {
            "total_papers": 0,
            "pdfs_downloaded": 0,
            "pdfs_failed": 0,
            "by_conference": {},
        }

    def run(self):
        """Main execution"""
        logger.info("=" * 70)
        logger.info("Semantic Scholar Paper Download (AAAI, IJCAI, AAMAS)")
        logger.info("=" * 70)
        logger.info(f"Output: {BASE_DIR}")
        logger.info(f"PDF workers: {MAX_PDF_WORKERS}\n")

        start_time = time.time()

        for conf_name, conf_config in CONFERENCES.items():
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing {conf_name.upper()}")
            logger.info(f"{'=' * 70}")

            self.stats["by_conference"][conf_name] = {"papers": 0, "pdfs": 0, "failed": 0}

            for year in conf_config["years"]:
                self.process_conference_year(conf_name, conf_config, year)

        elapsed = time.time() - start_time
        self.print_summary(elapsed)

    def process_conference_year(self, conf_name: str, conf_config: dict, year: int):
        """Process one conference/year"""
        logger.info(f"\nProcessing {conf_name.upper()} {year}...")

        output_dir = BASE_DIR / conf_name / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Search Semantic Scholar
        papers = self.semantic_scholar.search_papers(
            conf_name, conf_config["venue_variations"], year
        )

        if not papers:
            logger.warning(f"No papers found for {conf_name} {year}")
            return

        logger.info(f"Found {len(papers)} papers for {conf_name} {year}")

        # Download PDFs
        successful = self.download_pdfs_parallel(papers, output_dir)
        failed = len(papers) - successful

        # Save metadata
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "conference": conf_name,
                    "year": year,
                    "total_papers": len(papers),
                    "downloaded": successful,
                    "failed": failed,
                    "papers": [asdict(p) for p in papers],
                },
                f,
                indent=2,
            )

        # Update stats
        self.stats["total_papers"] += len(papers)
        self.stats["pdfs_downloaded"] += successful
        self.stats["pdfs_failed"] += failed
        self.stats["by_conference"][conf_name]["papers"] += len(papers)
        self.stats["by_conference"][conf_name]["pdfs"] += successful
        self.stats["by_conference"][conf_name]["failed"] += failed

        logger.info(f"✓ {conf_name.upper()} {year}: {successful}/{len(papers)} PDFs")

    def download_pdfs_parallel(self, papers: List[Paper], output_dir: Path) -> int:
        """Download PDFs in parallel"""
        successful = 0

        def download_one(paper: Paper) -> Tuple[bool, Paper]:
            pdf_path = output_dir / f"{self._sanitize(paper.paper_id)}.pdf"

            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                paper.downloaded = True
                return True, paper

            if self.download_pdf(paper, pdf_path):
                paper.downloaded = True
                return True, paper
            return False, paper

        with ThreadPoolExecutor(max_workers=MAX_PDF_WORKERS) as executor:
            futures = {executor.submit(download_one, p): p for p in papers}

            with tqdm(total=len(papers), desc="Downloading PDFs") as pbar:
                for future in as_completed(futures):
                    success, _ = future.result()
                    if success:
                        successful += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful})

        return successful

    def download_pdf(self, paper: Paper, output_path: Path) -> bool:
        """Download single PDF with fallbacks"""
        urls = []

        if paper.pdf_url:
            urls.append(paper.pdf_url)
        if paper.arxiv_id:
            urls.append(self.arxiv.get_pdf_url(paper.arxiv_id))
        if paper.doi:
            unpaywall_url = self.unpaywall.get_pdf_url(paper.doi)
            if unpaywall_url:
                urls.append(unpaywall_url)

        for url in urls:
            if not url:
                continue

            try:
                response = self.session.get(url, timeout=60, stream=True)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                    continue

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if output_path.stat().st_size > 1000:
                    with open(output_path, "rb") as f:
                        if f.read(4) == b"%PDF":
                            return True

                output_path.unlink()

            except Exception as e:
                logger.debug(f"Download failed: {e}")
                if output_path.exists():
                    output_path.unlink()

        return False

    def _sanitize(self, filename: str) -> str:
        for char in '<>:"/\\|?*':
            filename = filename.replace(char, "_")
        return filename[:200]

    def print_summary(self, elapsed: float):
        """Print summary"""
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total papers: {self.stats['total_papers']}")
        logger.info(f"PDFs downloaded: {self.stats['pdfs_downloaded']}")
        logger.info(f"PDFs failed: {self.stats['pdfs_failed']}")

        if self.stats["total_papers"] > 0:
            coverage = (self.stats["pdfs_downloaded"] / self.stats["total_papers"]) * 100
            logger.info(f"Coverage: {coverage:.1f}%")

        logger.info("\nBy Conference:")
        for conf, stats in self.stats["by_conference"].items():
            if stats["papers"] > 0:
                coverage = (stats["pdfs"] / stats["papers"]) * 100
                logger.info(f"  {conf.upper()}: {stats['pdfs']}/{stats['papers']} ({coverage:.1f}%)")

        logger.info(f"\nAPI requests made: {self.semantic_scholar.request_count}")
        logger.info(f"Time: {elapsed / 60:.1f} minutes")
        logger.info(f"Output: {BASE_DIR}")
        logger.info("=" * 70)


def main():
    try:
        downloader = PaperDownloader()
        downloader.run()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
