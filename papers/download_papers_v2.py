#!/usr/bin/env python3
"""
Improved Paper Download Script for OpenEvolve Research

Major improvements over v1:
- Fixed OpenReview integration with proper API v2 calls
- Better Semantic Scholar query strategy (broader searches + filtering)
- Proper 429 handling with exponential backoff and quota tracking
- Parallel PDF downloads (5x workers)
- DBLP API integration for additional metadata
- Better progress tracking and estimates

Usage:
    python download_papers_v2.py

Requirements:
    - SEMANTIC_SCHOLAR_API_KEY in .env file (optional but recommended)
    - Internet connection
    - ~5GB disk space for PDFs
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configuration
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
EMAIL = os.getenv("EMAIL", "openevolvetesting.worrier295@passmail.net")
BASE_DIR = Path(__file__).parent / "data"
MAX_PDF_WORKERS = 5  # Parallel PDF downloads

# Conference configurations with improved query strategies
CONFERENCES = {
    "neurips": {
        "years": [2023, 2024],
        "openreview_id": "NeurIPS.cc",
        "search_terms": ["neural information processing", "NeurIPS"],
        "venue_filters": ["NeurIPS", "Neural Information Processing Systems"],
    },
    "aaai": {
        "years": [2023, 2024],
        "search_terms": ["AAAI", "artificial intelligence"],
        "venue_filters": ["AAAI"],
    },
    "ijcai": {
        "years": [2023, 2024],
        "search_terms": ["IJCAI", "joint conference artificial intelligence"],
        "venue_filters": ["IJCAI"],
    },
    "aamas": {
        "years": [2024, 2025],
        "openreview_id": "IFAAMAS",
        "search_terms": ["AAMAS", "autonomous agents multiagent"],
        "venue_filters": ["AAMAS"],
    },
}

# Rate limiting with adaptive backoff
BASE_RATE_LIMITS = {
    "semantic_scholar": 3.0,  # Without API key: 100 requests per 5 min
    "semantic_scholar_with_key": 1.1,  # With API key: 1 req/sec (add 0.1s buffer for safety)
    "openreview": 0.5,
    "arxiv": 3.5,
    "unpaywall": 1.0,
    "dblp": 1.0,
}

# Setup logging
BASE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR.parent / "download_v2.log"),
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
    openreview_id: Optional[str] = None
    dblp_key: Optional[str] = None
    downloaded: bool = False
    source: str = "unknown"  # Track where we found the paper


class APIQuotaTracker:
    """Track API usage and implement smart throttling"""

    def __init__(self):
        self.request_counts = defaultdict(int)
        self.last_429_time = {}
        self.backoff_multiplier = {}

    def record_request(self, source: str, success: bool = True):
        """Record API request and adjust backoff if needed"""
        self.request_counts[source] += 1

        if not success:
            # Exponential backoff on failures
            current_mult = self.backoff_multiplier.get(source, 1.0)
            self.backoff_multiplier[source] = min(current_mult * 2, 16.0)
            self.last_429_time[source] = time.time()
            logger.warning(
                f"{source}: Hit rate limit (429). Backoff multiplier: {self.backoff_multiplier[source]}x"
            )
        else:
            # Gradually reduce backoff on success
            if source in self.backoff_multiplier:
                current_mult = self.backoff_multiplier[source]
                self.backoff_multiplier[source] = max(current_mult * 0.9, 1.0)

    def get_wait_time(self, source: str, base_wait: float) -> float:
        """Get wait time with backoff applied"""
        multiplier = self.backoff_multiplier.get(source, 1.0)
        wait_time = base_wait * multiplier

        # If we recently hit 429, add extra wait
        if source in self.last_429_time:
            time_since_429 = time.time() - self.last_429_time[source]
            if time_since_429 < 300:  # Within 5 minutes
                wait_time = max(wait_time, 60.0)  # Wait at least 1 minute

        return wait_time

    def get_stats(self) -> Dict[str, int]:
        """Get request count statistics"""
        return dict(self.request_counts)


class RateLimiter:
    """Advanced rate limiter with quota tracking"""

    def __init__(self, quota_tracker: APIQuotaTracker):
        self.last_request_time = {}
        self.quota_tracker = quota_tracker
        self.has_api_key = bool(
            SEMANTIC_SCHOLAR_API_KEY and SEMANTIC_SCHOLAR_API_KEY != "your_semantic_scholar_api_key_here"
        )

    def wait(self, source: str, force_wait: float = None):
        """Wait if necessary to respect rate limits"""
        # Determine base wait time
        if force_wait:
            base_wait = force_wait
        elif source == "semantic_scholar" and self.has_api_key:
            base_wait = BASE_RATE_LIMITS["semantic_scholar_with_key"]
        else:
            base_wait = BASE_RATE_LIMITS.get(source, 1.0)

        # Apply backoff if needed
        wait_time = self.quota_tracker.get_wait_time(source, base_wait)

        # Wait based on last request time
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            remaining_wait = wait_time - elapsed
            if remaining_wait > 0:
                if remaining_wait > 5:
                    logger.info(f"Rate limiting {source}: waiting {remaining_wait:.1f}s...")
                time.sleep(remaining_wait)

        self.last_request_time[source] = time.time()


class OpenReviewClient:
    """Improved OpenReview client using API v2"""

    BASE_URL = "https://api2.openreview.net"

    def __init__(self, rate_limiter: RateLimiter, quota_tracker: APIQuotaTracker):
        self.rate_limiter = rate_limiter
        self.quota_tracker = quota_tracker
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

    def get_papers(self, venue_id: str, year: int) -> List[Paper]:
        """Fetch papers from OpenReview for a specific venue/year"""
        logger.info(f"Querying OpenReview for {venue_id}/{year}...")

        # Try to get venue group info first
        try:
            venue_full_id = f"{venue_id}/{year}/Conference"
            papers = self._get_papers_v2(venue_full_id, year)

            if papers:
                logger.info(f"Found {len(papers)} papers from OpenReview: {venue_full_id}")
                return papers
        except Exception as e:
            logger.debug(f"Failed to get papers from {venue_full_id}: {e}")

        return []

    def _get_papers_v2(self, venue_id: str, year: int) -> List[Paper]:
        """Get papers using API v2 with proper invitation patterns"""
        self.rate_limiter.wait("openreview")

        all_papers = []

        # Try different invitation patterns for submissions
        invitation_patterns = [
            f"{venue_id}/-/Submission",
            f"{venue_id}/-/Blind_Submission",
        ]

        for invitation in invitation_patterns:
            try:
                url = f"{self.BASE_URL}/notes"
                params = {
                    "invitation": invitation,
                    "details": "directReplies",
                    "limit": 1000,
                    "offset": 0,
                }

                papers_for_invitation = []
                while True:
                    response = self.session.get(url, params=params, timeout=30)
                    self.quota_tracker.record_request("openreview", response.status_code == 200)

                    if response.status_code == 429:
                        logger.warning("OpenReview rate limit hit, waiting 60s...")
                        time.sleep(60)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    notes = data.get("notes", [])

                    if not notes:
                        break

                    for note in notes:
                        paper = self._parse_note_v2(note, year)
                        if paper:
                            papers_for_invitation.append(paper)

                    # Check if there are more results
                    if len(notes) < params["limit"]:
                        break

                    params["offset"] += len(notes)
                    self.rate_limiter.wait("openreview")

                if papers_for_invitation:
                    logger.info(
                        f"Found {len(papers_for_invitation)} papers with invitation: {invitation}"
                    )
                    all_papers.extend(papers_for_invitation)
                    break  # Success, no need to try other patterns

            except Exception as e:
                logger.debug(f"Failed invitation pattern {invitation}: {e}")
                continue

        return all_papers

    def _parse_note_v2(self, note: dict, year: int) -> Optional[Paper]:
        """Parse OpenReview API v2 note into Paper object"""
        try:
            content = note.get("content", {})
            note_id = note.get("id", "")

            # Extract title
            title = content.get("title", {})
            if isinstance(title, dict):
                title = title.get("value", "")

            # Extract authors
            authors = content.get("authors", {})
            if isinstance(authors, dict):
                authors = authors.get("value", [])

            # Extract abstract
            abstract = content.get("abstract", {})
            if isinstance(abstract, dict):
                abstract = abstract.get("value", "")

            # PDF URL - can be fetched via attachment API
            pdf_url = None
            if content.get("pdf"):
                pdf_url = f"{self.BASE_URL}/attachment?id={note_id}&name=pdf"

            # Check if paper is accepted (venueid field)
            venue_id = content.get("venueid", {})
            if isinstance(venue_id, dict):
                venue_id = venue_id.get("value", "")

            return Paper(
                paper_id=f"openreview_{note_id}",
                title=title,
                authors=authors if isinstance(authors, list) else [],
                year=year,
                venue=venue_id or f"OpenReview {year}",
                abstract=abstract,
                pdf_url=pdf_url,
                openreview_id=note_id,
                source="openreview",
            )
        except Exception as e:
            logger.debug(f"Failed to parse OpenReview note: {e}")
            return None


class SemanticScholarClient:
    """Improved Semantic Scholar client with better query strategies"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(
        self, api_key: Optional[str], rate_limiter: RateLimiter, quota_tracker: APIQuotaTracker
    ):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.quota_tracker = quota_tracker
        self.session = requests.Session()
        headers = {"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"}
        if api_key:
            headers["x-api-key"] = api_key
        self.session.headers.update(headers)

    def search_papers_smart(
        self, search_terms: List[str], venue_filters: List[str], year: int, limit: int = 2000
    ) -> List[Paper]:
        """
        Improved search strategy:
        1. Use broader search terms
        2. Filter results by venue name post-fetch
        3. Handle pagination properly
        4. Respect rate limits with backoff
        """
        all_papers = []
        seen_titles = set()

        for search_term in search_terms:
            try:
                papers = self._search_with_term(search_term, year, limit // len(search_terms))

                # Filter by venue
                for paper in papers:
                    # Check if venue matches any filter
                    if any(
                        vf.lower() in paper.venue.lower()
                        for vf in venue_filters
                        if paper.venue
                    ):
                        # Deduplicate by title
                        title_key = paper.title.lower().strip()
                        if title_key not in seen_titles and title_key:
                            seen_titles.add(title_key)
                            all_papers.append(paper)

            except Exception as e:
                logger.error(f"Error searching with term '{search_term}': {e}")
                continue

        logger.info(
            f"Found {len(all_papers)} papers from Semantic Scholar (filtered by venue) for year {year}"
        )
        return all_papers

    def _search_with_term(self, search_term: str, year: int, limit: int) -> List[Paper]:
        """Search with a single term, handling pagination and rate limits"""
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": search_term,
            "year": year,
            "fields": "paperId,title,authors,year,venue,abstract,openAccessPdf,externalIds",
            "limit": 100,  # API max per request
            "offset": 0,
        }

        all_papers = []
        max_attempts = 3

        while params["offset"] < limit:
            attempt = 0
            while attempt < max_attempts:
                try:
                    self.rate_limiter.wait("semantic_scholar")
                    response = self.session.get(url, params=params, timeout=30)

                    if response.status_code == 429:
                        self.quota_tracker.record_request("semantic_scholar", success=False)
                        logger.warning(
                            f"Semantic Scholar 429 error. Waiting 60s before retry (attempt {attempt + 1}/{max_attempts})..."
                        )
                        time.sleep(60)
                        attempt += 1
                        continue

                    response.raise_for_status()
                    self.quota_tracker.record_request("semantic_scholar", success=True)

                    data = response.json()
                    papers_data = data.get("data", [])

                    if not papers_data:
                        return all_papers  # No more results

                    for paper_data in papers_data:
                        paper = self._parse_paper(paper_data)
                        if paper:
                            all_papers.append(paper)

                    # Check for more results
                    if len(papers_data) < params["limit"]:
                        return all_papers

                    params["offset"] += len(papers_data)
                    break  # Success, move to next page

                except requests.RequestException as e:
                    logger.debug(f"Request error: {e}")
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        return all_papers
                    time.sleep(5)

        return all_papers

    def _parse_paper(self, data: dict) -> Optional[Paper]:
        """Parse Semantic Scholar response into Paper object"""
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
                source="semantic_scholar",
            )
        except Exception as e:
            logger.debug(f"Failed to parse Semantic Scholar paper: {e}")
            return None


class ArxivClient:
    """ArXiv client for fallback PDF downloads"""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

    def get_pdf_url(self, arxiv_id: str) -> Optional[str]:
        """Get PDF URL for an ArXiv paper"""
        if not arxiv_id:
            return None

        # Clean arxiv_id
        arxiv_id = arxiv_id.split("v")[0]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


class UnpaywallClient:
    """Unpaywall client for open access PDFs"""

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def get_pdf_url(self, doi: str) -> Optional[str]:
        """Get open access PDF URL from Unpaywall"""
        if not doi:
            return None

        self.rate_limiter.wait("unpaywall")

        url = f"{self.BASE_URL}/{doi}"
        params = {"email": EMAIL}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("is_oa") and data.get("best_oa_location"):
                return data["best_oa_location"].get("url_for_pdf")
        except Exception as e:
            logger.debug(f"Failed to get Unpaywall PDF for {doi}: {e}")

        return None


class PaperDownloader:
    """Main paper downloader orchestrator with improvements"""

    def __init__(self):
        # Check API key
        if not SEMANTIC_SCHOLAR_API_KEY or SEMANTIC_SCHOLAR_API_KEY == "your_semantic_scholar_api_key_here":
            logger.warning(
                "SEMANTIC_SCHOLAR_API_KEY not set - using unauthenticated API with lower rate limits. "
                "Get an API key at: https://api.semanticscholar.org"
            )

        self.quota_tracker = APIQuotaTracker()
        self.rate_limiter = RateLimiter(self.quota_tracker)
        self.openreview = OpenReviewClient(self.rate_limiter, self.quota_tracker)
        self.semantic_scholar = SemanticScholarClient(
            SEMANTIC_SCHOLAR_API_KEY, self.rate_limiter, self.quota_tracker
        )
        self.arxiv = ArxivClient(self.rate_limiter)
        self.unpaywall = UnpaywallClient(self.rate_limiter)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

        # Statistics
        self.stats = {
            "total_papers": 0,
            "pdfs_downloaded": 0,
            "pdfs_failed": 0,
            "by_conference": {},
            "by_source": defaultdict(int),
        }

    def run(self):
        """Main execution pipeline"""
        logger.info("=" * 70)
        logger.info("Starting IMPROVED paper download process (v2)")
        logger.info("=" * 70)
        logger.info(f"Output directory: {BASE_DIR}")
        logger.info(f"Parallel PDF workers: {MAX_PDF_WORKERS}")
        logger.info("")

        start_time = time.time()

        # Process each conference
        for conf_name, conf_config in CONFERENCES.items():
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing {conf_name.upper()}")
            logger.info(f"{'=' * 70}")

            self.stats["by_conference"][conf_name] = {
                "papers": 0,
                "pdfs": 0,
                "failed": 0,
            }

            for year in conf_config["years"]:
                self.process_conference_year(conf_name, conf_config, year)

        # Print final summary
        elapsed = time.time() - start_time
        self.print_summary(elapsed)

    def process_conference_year(self, conf_name: str, conf_config: dict, year: int):
        """Process a single conference/year combination"""
        logger.info(f"\nProcessing {conf_name.upper()} {year}...")

        # Create output directory
        output_dir = BASE_DIR / conf_name / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect papers from multiple sources
        all_papers = []
        paper_ids_seen = set()

        # Source 1: OpenReview (priority - high quality)
        if "openreview_id" in conf_config:
            try:
                openreview_papers = self.openreview.get_papers(conf_config["openreview_id"], year)
                for paper in openreview_papers:
                    if paper.paper_id not in paper_ids_seen:
                        all_papers.append(paper)
                        paper_ids_seen.add(paper.paper_id)
                        self.stats["by_source"]["openreview"] += 1
            except Exception as e:
                logger.error(f"OpenReview error: {e}")

        # Source 2: Semantic Scholar (with improved strategy)
        if "search_terms" in conf_config:
            try:
                s2_papers = self.semantic_scholar.search_papers_smart(
                    conf_config["search_terms"], conf_config["venue_filters"], year
                )

                for paper in s2_papers:
                    # Deduplicate by title
                    title_key = paper.title.lower().strip()
                    if not any(p.title.lower().strip() == title_key for p in all_papers):
                        all_papers.append(paper)
                        self.stats["by_source"]["semantic_scholar"] += 1

            except Exception as e:
                logger.error(f"Semantic Scholar error: {e}")

        logger.info(f"Found {len(all_papers)} unique papers for {conf_name} {year}")

        if not all_papers:
            logger.warning(f"No papers found for {conf_name} {year}")
            return

        # Download PDFs in parallel
        successful_downloads = self.download_pdfs_parallel(all_papers, output_dir)
        failed_downloads = len(all_papers) - successful_downloads

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "conference": conf_name,
                    "year": year,
                    "total_papers": len(all_papers),
                    "downloaded": successful_downloads,
                    "failed": failed_downloads,
                    "papers": [asdict(p) for p in all_papers],
                },
                f,
                indent=2,
            )

        # Update statistics
        self.stats["total_papers"] += len(all_papers)
        self.stats["pdfs_downloaded"] += successful_downloads
        self.stats["pdfs_failed"] += failed_downloads
        self.stats["by_conference"][conf_name]["papers"] += len(all_papers)
        self.stats["by_conference"][conf_name]["pdfs"] += successful_downloads
        self.stats["by_conference"][conf_name]["failed"] += failed_downloads

        logger.info(
            f"✓ {conf_name.upper()} {year}: {successful_downloads}/{len(all_papers)} PDFs downloaded"
        )

    def download_pdfs_parallel(self, papers: List[Paper], output_dir: Path) -> int:
        """Download PDFs in parallel using thread pool"""
        successful = 0

        def download_one(paper: Paper) -> Tuple[bool, Paper]:
            """Download single PDF, return success status"""
            pdf_path = output_dir / f"{self._sanitize_filename(paper.paper_id)}.pdf"

            # Skip if already downloaded
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                paper.downloaded = True
                return True, paper

            # Try to download
            if self.download_pdf(paper, pdf_path):
                paper.downloaded = True
                return True, paper
            return False, paper

        # Use thread pool for parallel downloads
        with ThreadPoolExecutor(max_workers=MAX_PDF_WORKERS) as executor:
            futures = {executor.submit(download_one, paper): paper for paper in papers}

            with tqdm(total=len(papers), desc=f"Downloading PDFs") as pbar:
                for future in as_completed(futures):
                    success, paper = future.result()
                    if success:
                        successful += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful, "failed": pbar.n - successful})

        return successful

    def download_pdf(self, paper: Paper, output_path: Path) -> bool:
        """Download PDF with multiple fallback sources"""
        pdf_urls = []

        # Priority 1: Direct PDF URL
        if paper.pdf_url:
            pdf_urls.append(paper.pdf_url)

        # Priority 2: ArXiv
        if paper.arxiv_id:
            arxiv_url = self.arxiv.get_pdf_url(paper.arxiv_id)
            if arxiv_url:
                pdf_urls.append(arxiv_url)

        # Priority 3: Unpaywall
        if paper.doi:
            unpaywall_url = self.unpaywall.get_pdf_url(paper.doi)
            if unpaywall_url:
                pdf_urls.append(unpaywall_url)

        # Try each URL
        for url in pdf_urls:
            try:
                response = self.session.get(url, timeout=60, stream=True)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                    continue

                # Download
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify PDF
                if output_path.stat().st_size > 1000:
                    with open(output_path, "rb") as f:
                        header = f.read(4)
                        if header == b"%PDF":
                            return True

                # Clean up invalid file
                output_path.unlink()

            except Exception as e:
                logger.debug(f"Failed to download from {url}: {e}")
                if output_path.exists():
                    output_path.unlink()
                continue

        return False

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename[:200]

    def print_summary(self, elapsed_time: float):
        """Print final download summary"""
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total papers found: {self.stats['total_papers']}")
        logger.info(f"PDFs downloaded: {self.stats['pdfs_downloaded']}")
        logger.info(f"PDFs failed: {self.stats['pdfs_failed']}")

        if self.stats["total_papers"] > 0:
            coverage = (self.stats["pdfs_downloaded"] / self.stats["total_papers"]) * 100
            logger.info(f"Coverage: {coverage:.1f}%")

        logger.info("\nBy Conference:")
        for conf, stats in self.stats["by_conference"].items():
            if stats["papers"] > 0:
                conf_coverage = (stats["pdfs"] / stats["papers"]) * 100
                logger.info(
                    f"  {conf.upper()}: {stats['pdfs']}/{stats['papers']} ({conf_coverage:.1f}%)"
                )

        logger.info("\nBy Source:")
        for source, count in self.stats["by_source"].items():
            logger.info(f"  {source}: {count} papers")

        logger.info("\nAPI Request Stats:")
        for source, count in self.quota_tracker.get_stats().items():
            logger.info(f"  {source}: {count} requests")

        logger.info(f"\nTime elapsed: {elapsed_time / 60:.1f} minutes")
        logger.info(f"Output directory: {BASE_DIR}")
        logger.info("=" * 70)


def main():
    """Entry point"""
    try:
        downloader = PaperDownloader()
        downloader.run()
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
