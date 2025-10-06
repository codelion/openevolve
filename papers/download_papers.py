#!/usr/bin/env python3
"""
Paper Download Script for OpenEvolve Research

Downloads papers from NeurIPS, AAAI, IJCAI, and AAMAS conferences (2023-2024/2025)
using multiple sources: OpenReview, Semantic Scholar, ArXiv, and Unpaywall.

Usage:
    python download_papers.py

Requirements:
    - SEMANTIC_SCHOLAR_API_KEY in .env file
    - Internet connection
    - ~5GB disk space for PDFs
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configuration
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Optional - works without it
EMAIL = os.getenv("EMAIL", "openevolvetesting.worrier295@passmail.net")
BASE_DIR = Path(__file__).parent / "data"

# Conference configurations
CONFERENCES = {
    "neurips": {
        "years": [2023, 2024],
        "venues": ["NeurIPS 2023", "NeurIPS 2024"],
        "openreview_venue": "NeurIPS.cc",
    },
    "aaai": {
        "years": [2023, 2024],
        "venues": ["AAAI 2023", "AAAI 2024"],
    },
    "ijcai": {
        "years": [2023, 2024],
        "venues": ["IJCAI 2023", "IJCAI 2024"],
    },
    "aamas": {
        "years": [2024, 2025],
        "venues": ["AAMAS 2024", "AAMAS 2025"],
        "openreview_venue": "IFAAMAS",
    },
}

# Rate limiting (seconds between requests)
RATE_LIMITS = {
    "semantic_scholar": 3.0,  # 100 requests per 5 min without API key = ~3 sec/request
    "openreview": 0.5,
    "arxiv": 3.5,  # Conservative: 3 seconds
    "unpaywall": 1.0,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR.parent / "download.log"),
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
    downloaded: bool = False


class RateLimiter:
    """Simple rate limiter using timestamps"""

    def __init__(self):
        self.last_request_time = {}

    def wait(self, source: str):
        """Wait if necessary to respect rate limits"""
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            wait_time = RATE_LIMITS.get(source, 1.0) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
        self.last_request_time[source] = time.time()


class OpenReviewClient:
    """Client for OpenReview API"""

    BASE_URL = "https://api.openreview.net"

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def get_papers(self, venue: str, year: int) -> List[Paper]:
        """Fetch papers from OpenReview for a specific venue/year"""
        self.rate_limiter.wait("openreview")

        # Try different invitation patterns
        invitation_patterns = [
            f"{venue}/{year}/Conference/-/Blind_Submission",
            f"{venue}/{year}/Conference/-/Submission",
            f"{venue}/{year}/-/Submission",
        ]

        papers = []
        for invitation in invitation_patterns:
            try:
                url = f"{self.BASE_URL}/notes"
                params = {"invitation": invitation, "details": "replies"}
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    notes = response.json().get("notes", [])
                    if notes:
                        logger.info(f"Found {len(notes)} papers from OpenReview: {invitation}")
                        for note in notes:
                            paper = self._parse_note(note, year)
                            if paper:
                                papers.append(paper)
                        break
            except Exception as e:
                logger.debug(f"Failed invitation pattern {invitation}: {e}")
                continue

        return papers

    def _parse_note(self, note: dict, year: int) -> Optional[Paper]:
        """Parse OpenReview note into Paper object"""
        try:
            content = note.get("content", {})
            paper_id = note.get("id", "")

            # Extract PDF URL from OpenReview
            pdf_url = None
            if "pdf" in content:
                pdf_url = f"{self.BASE_URL}/pdf?id={paper_id}"

            return Paper(
                paper_id=f"openreview_{paper_id}",
                title=content.get("title", ""),
                authors=content.get("authors", []),
                year=year,
                venue=note.get("invitation", "").split("/")[0],
                abstract=content.get("abstract", ""),
                pdf_url=pdf_url,
                openreview_id=paper_id,
            )
        except Exception as e:
            logger.debug(f"Failed to parse OpenReview note: {e}")
            return None


class SemanticScholarClient:
    """Client for Semantic Scholar API"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str], rate_limiter: RateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        headers = {"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"}
        if api_key:
            headers["x-api-key"] = api_key
        self.session.headers.update(headers)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def search_papers(self, venue: str, year: int, limit: int = 1000) -> List[Paper]:
        """Search papers by venue and year"""
        self.rate_limiter.wait("semantic_scholar")

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": venue,
            "year": year,
            "fields": "paperId,title,authors,year,venue,abstract,openAccessPdf,externalIds",
            "limit": min(limit, 100),  # API max is 100 per request
            "offset": 0,
        }

        all_papers = []
        while True:
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                papers_data = data.get("data", [])
                if not papers_data:
                    break

                for paper_data in papers_data:
                    paper = self._parse_paper(paper_data)
                    if paper:
                        all_papers.append(paper)

                # Check if there are more results
                if len(papers_data) < params["limit"]:
                    break

                params["offset"] += len(papers_data)
                if params["offset"] >= limit:
                    break

                self.rate_limiter.wait("semantic_scholar")

            except Exception as e:
                logger.error(f"Error searching Semantic Scholar: {e}")
                break

        logger.info(f"Found {len(all_papers)} papers from Semantic Scholar: {venue} {year}")
        return all_papers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Get paper details by Semantic Scholar ID"""
        self.rate_limiter.wait("semantic_scholar")

        url = f"{self.BASE_URL}/paper/{paper_id}"
        params = {
            "fields": "paperId,title,authors,year,venue,abstract,openAccessPdf,externalIds"
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return self._parse_paper(response.json())
        except Exception as e:
            logger.debug(f"Failed to get paper {paper_id}: {e}")
            return None

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
            )
        except Exception as e:
            logger.debug(f"Failed to parse Semantic Scholar paper: {e}")
            return None


class ArxivClient:
    """Client for ArXiv API"""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def get_pdf_url(self, arxiv_id: str) -> Optional[str]:
        """Get PDF URL for an ArXiv paper"""
        if not arxiv_id:
            return None

        self.rate_limiter.wait("arxiv")

        # Clean arxiv_id (remove version if present)
        arxiv_id = arxiv_id.split("v")[0]

        try:
            response = self.session.get(
                self.BASE_URL, params={"id_list": arxiv_id}, timeout=30
            )
            response.raise_for_status()

            # ArXiv API returns XML, check if entry exists
            if arxiv_id in response.text:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        except Exception as e:
            logger.debug(f"Failed to get ArXiv PDF for {arxiv_id}: {e}")

        return None


class UnpaywallClient:
    """Client for Unpaywall API"""

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = requests.Session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
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
    """Main paper downloader orchestrator"""

    def __init__(self):
        # API key is optional - works without it but with lower rate limits
        if not SEMANTIC_SCHOLAR_API_KEY or SEMANTIC_SCHOLAR_API_KEY == "your_semantic_scholar_api_key_here":
            logger.warning(
                "SEMANTIC_SCHOLAR_API_KEY not set - using unauthenticated API with lower rate limits "
                "(100 requests per 5 minutes). For faster downloads, get an API key at: "
                "https://api.semanticscholar.org"
            )

        self.rate_limiter = RateLimiter()
        self.openreview = OpenReviewClient(self.rate_limiter)
        self.semantic_scholar = SemanticScholarClient(SEMANTIC_SCHOLAR_API_KEY, self.rate_limiter)
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
        }

    def run(self):
        """Main execution pipeline"""
        logger.info("Starting paper download process...")
        logger.info(f"Output directory: {BASE_DIR}")

        # Create base directory
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        # Process each conference
        for conf_name, conf_config in CONFERENCES.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {conf_name.upper()}")
            logger.info(f"{'=' * 60}")

            self.stats["by_conference"][conf_name] = {
                "papers": 0,
                "pdfs": 0,
                "failed": 0,
            }

            for year in conf_config["years"]:
                self.process_conference_year(conf_name, conf_config, year)

        # Print final summary
        self.print_summary()

    def process_conference_year(self, conf_name: str, conf_config: dict, year: int):
        """Process a single conference/year combination"""
        logger.info(f"\nProcessing {conf_name.upper()} {year}...")

        # Create output directory
        output_dir = BASE_DIR / conf_name / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect papers from multiple sources
        all_papers = []
        paper_ids_seen = set()

        # Source 1: OpenReview (if configured)
        if "openreview_venue" in conf_config:
            try:
                openreview_papers = self.openreview.get_papers(
                    conf_config["openreview_venue"], year
                )
                for paper in openreview_papers:
                    if paper.paper_id not in paper_ids_seen:
                        all_papers.append(paper)
                        paper_ids_seen.add(paper.paper_id)
            except Exception as e:
                logger.error(f"OpenReview error: {e}")

        # Source 2: Semantic Scholar
        for venue in conf_config.get("venues", []):
            try:
                s2_papers = self.semantic_scholar.search_papers(venue, year)
                for paper in s2_papers:
                    # Check for duplicates by title (case-insensitive)
                    title_key = paper.title.lower().strip()
                    if not any(p.title.lower().strip() == title_key for p in all_papers):
                        all_papers.append(paper)
            except Exception as e:
                logger.error(f"Semantic Scholar error for {venue}: {e}")

        logger.info(f"Found {len(all_papers)} unique papers for {conf_name} {year}")

        if not all_papers:
            logger.warning(f"No papers found for {conf_name} {year}")
            return

        # Download PDFs
        successful_downloads = 0
        failed_downloads = 0

        for paper in tqdm(all_papers, desc=f"Downloading {conf_name} {year}"):
            try:
                pdf_path = output_dir / f"{self._sanitize_filename(paper.paper_id)}.pdf"

                # Skip if already downloaded
                if pdf_path.exists():
                    paper.downloaded = True
                    successful_downloads += 1
                    continue

                # Try to download PDF
                if self.download_pdf(paper, pdf_path):
                    paper.downloaded = True
                    successful_downloads += 1
                else:
                    failed_downloads += 1

            except Exception as e:
                logger.debug(f"Error downloading {paper.title}: {e}")
                failed_downloads += 1

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

    def download_pdf(self, paper: Paper, output_path: Path) -> bool:
        """Download PDF with multiple fallback sources"""
        pdf_urls = []

        # Priority 1: Direct PDF URL from paper metadata
        if paper.pdf_url:
            pdf_urls.append(paper.pdf_url)

        # Priority 2: ArXiv
        if paper.arxiv_id:
            arxiv_url = self.arxiv.get_pdf_url(paper.arxiv_id)
            if arxiv_url:
                pdf_urls.append(arxiv_url)

        # Priority 3: Unpaywall (via DOI)
        if paper.doi:
            unpaywall_url = self.unpaywall.get_pdf_url(paper.doi)
            if unpaywall_url:
                pdf_urls.append(unpaywall_url)

        # Try each URL
        for url in pdf_urls:
            try:
                response = self.session.get(url, timeout=60, stream=True)
                response.raise_for_status()

                # Check if response is actually a PDF
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                    continue

                # Download PDF
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify file is not empty and looks like a PDF
                if output_path.stat().st_size > 1000:  # At least 1KB
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
        """Sanitize filename for safe filesystem storage"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        # Limit length
        return filename[:200]

    def print_summary(self):
        """Print final download summary"""
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
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

        logger.info(f"\nOutput directory: {BASE_DIR}")
        logger.info("=" * 60)


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
