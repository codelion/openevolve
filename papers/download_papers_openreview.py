#!/usr/bin/env python3
"""
OpenReview-Only Paper Download Script

ALL major AI conferences are on OpenReview!
No Semantic Scholar = no rate limit issues!

Expected results:
- NeurIPS 2023: ~3,400 papers
- NeurIPS 2024: ~4,200 papers
- AAAI 2023: ~1,700 papers
- AAAI 2024: ~2,300 papers
- IJCAI 2023: ~600 papers
- IJCAI 2024: ~800 papers
- AAMAS 2024: ~800 papers
- AAMAS 2025: ~800 papers
Total: ~14,600 papers with 70-90% PDF coverage!

Usage:
    python download_papers_openreview.py
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

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL", "openevolvetesting.worrier295@passmail.net")
BASE_DIR = Path(__file__).parent / "data"
MAX_PDF_WORKERS = 5

# OpenReview conferences - ALL conferences are on OpenReview!
CONFERENCES = {
    "neurips": {
        "years": [2023, 2024],
        "venue_ids": {
            2023: "NeurIPS.cc/2023/Conference",
            2024: "NeurIPS.cc/2024/Conference",
        },
    },
    "aaai": {
        "years": [2023, 2024],
        "venue_ids": {
            2023: "AAAI.org/2023/Conference",
            2024: "AAAI.org/2024/Conference",
        },
    },
    "ijcai": {
        "years": [2023, 2024],
        "venue_ids": {
            2023: "ijcai.org/IJCAI/2023/Conference",
            2024: "ijcai.org/IJCAI/2024/Conference",
        },
    },
    "aamas": {
        "years": [2024, 2025],
        "venue_ids": {
            2024: "ifaamas.org/AAMAS/2024/Conference",
            2025: "ifaamas.org/AAMAS/2025/Conference",
        },
    },
}

# Setup logging
BASE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR.parent / "download_openreview.log"),
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
    openreview_id: Optional[str] = None
    downloaded: bool = False


class OpenReviewClient:
    """OpenReview API v2 client"""

    BASE_URL = "https://api2.openreview.net"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})
        self.last_request = 0

    def _rate_limit(self):
        """Simple rate limiting: 0.5s between requests"""
        elapsed = time.time() - self.last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self.last_request = time.time()

    def get_papers(self, venue_id: str, year: int) -> List[Paper]:
        """Fetch all papers from OpenReview for a venue/year"""
        logger.info(f"Querying OpenReview for {venue_id}...")
        return self._get_papers_v2(venue_id, year)

    def _get_papers_v2(self, venue_id: str, year: int) -> List[Paper]:
        """Get papers using API v2"""
        all_papers = []

        # Try submission invitation patterns
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

                papers_found = []
                while True:
                    self._rate_limit()
                    response = self.session.get(url, params=params, timeout=30)

                    if response.status_code == 429:
                        logger.warning("Rate limit hit, waiting 30s...")
                        time.sleep(30)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    notes = data.get("notes", [])

                    if not notes:
                        break

                    for note in notes:
                        paper = self._parse_note(note, year)
                        if paper:
                            papers_found.append(paper)

                    logger.info(
                        f"  Fetched {len(papers_found)} papers so far from {invitation}..."
                    )

                    # Check for more results
                    if len(notes) < params["limit"]:
                        break

                    params["offset"] += len(notes)

                if papers_found:
                    logger.info(f"Found {len(papers_found)} papers from {invitation}")
                    all_papers = papers_found
                    break  # Success

            except Exception as e:
                logger.error(f"Error with {invitation}: {e}")
                continue

        return all_papers

    def _parse_note(self, note: dict, year: int) -> Optional[Paper]:
        """Parse OpenReview note into Paper"""
        try:
            content = note.get("content", {})
            note_id = note.get("id", "")

            # Extract fields (handle both dict and direct values)
            def get_value(field):
                if isinstance(field, dict):
                    return field.get("value", "")
                return field

            title = get_value(content.get("title", ""))
            authors = get_value(content.get("authors", []))
            abstract = get_value(content.get("abstract", ""))
            venue_id = get_value(content.get("venueid", ""))

            # PDF URL
            pdf_url = None
            if content.get("pdf"):
                pdf_url = f"{self.BASE_URL}/pdf?id={note_id}"

            return Paper(
                paper_id=f"openreview_{note_id}",
                title=title,
                authors=authors if isinstance(authors, list) else [],
                year=year,
                venue=venue_id or f"OpenReview {year}",
                abstract=abstract,
                pdf_url=pdf_url,
                openreview_id=note_id,
            )
        except Exception as e:
            logger.debug(f"Failed to parse note: {e}")
            return None


class PaperDownloader:
    """Main downloader for OpenReview papers"""

    def __init__(self):
        self.openreview = OpenReviewClient()
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
        logger.info("OpenReview-Only Paper Download")
        logger.info("=" * 70)
        logger.info(f"Output directory: {BASE_DIR}")
        logger.info(f"Parallel PDF workers: {MAX_PDF_WORKERS}\n")

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

        # Get papers from OpenReview using year-specific venue ID
        venue_id = conf_config["venue_ids"][year]
        papers = self.openreview.get_papers(venue_id, year)

        if not papers:
            logger.warning(f"No papers found for {conf_name} {year}")
            return

        logger.info(f"Found {len(papers)} papers for {conf_name} {year}")

        # Download PDFs in parallel
        successful = self.download_pdfs_parallel(papers, output_dir)
        failed = len(papers) - successful

        # Save metadata
        metadata = {
            "conference": conf_name,
            "year": year,
            "total_papers": len(papers),
            "downloaded": successful,
            "failed": failed,
            "papers": [asdict(p) for p in papers],
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update stats
        self.stats["total_papers"] += len(papers)
        self.stats["pdfs_downloaded"] += successful
        self.stats["pdfs_failed"] += failed
        self.stats["by_conference"][conf_name]["papers"] += len(papers)
        self.stats["by_conference"][conf_name]["pdfs"] += successful
        self.stats["by_conference"][conf_name]["failed"] += failed

        logger.info(f"✓ {conf_name.upper()} {year}: {successful}/{len(papers)} PDFs downloaded")

    def download_pdfs_parallel(self, papers: List[Paper], output_dir: Path) -> int:
        """Download PDFs in parallel"""
        successful = 0

        def download_one(paper: Paper) -> Tuple[bool, Paper]:
            pdf_path = output_dir / f"{self._sanitize(paper.paper_id)}.pdf"

            # Skip existing
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                paper.downloaded = True
                return True, paper

            # Download
            if self.download_pdf(paper, pdf_path):
                paper.downloaded = True
                return True, paper
            return False, paper

        with ThreadPoolExecutor(max_workers=MAX_PDF_WORKERS) as executor:
            futures = {executor.submit(download_one, p): p for p in papers}

            with tqdm(total=len(papers), desc=f"Downloading PDFs") as pbar:
                for future in as_completed(futures):
                    success, paper = future.result()
                    if success:
                        successful += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successful})

        return successful

    def download_pdf(self, paper: Paper, output_path: Path) -> bool:
        """Download single PDF"""
        if not paper.pdf_url:
            return False

        try:
            response = self.session.get(paper.pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                return False

            # Download
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify
            if output_path.stat().st_size > 1000:
                with open(output_path, "rb") as f:
                    if f.read(4) == b"%PDF":
                        return True

            output_path.unlink()
            return False

        except Exception as e:
            logger.debug(f"Download failed for {paper.title}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def _sanitize(self, filename: str) -> str:
        """Sanitize filename"""
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

        logger.info(f"\nTime: {elapsed / 60:.1f} minutes")
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
