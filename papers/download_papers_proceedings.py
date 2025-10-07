#!/usr/bin/env python3
"""
Conference Proceedings Scraper

Downloads papers directly from official conference proceedings websites.
NO API needed - scrapes public proceedings pages!

Sources:
- AAAI: ojs.aaai.org (Open Journal Systems)
- IJCAI: ijcai.org/proceedings
- AAMAS: To be determined

Expected results:
- AAAI 2023: ~1,700 papers with PDFs
- AAAI 2024: ~2,400 papers with PDFs
- IJCAI 2023: ~650 papers with PDFs
- IJCAI 2024: ~1,000 papers with PDFs
Total: ~5,750 papers with 90%+ PDF coverage!

Usage:
    python download_papers_proceedings.py
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

EMAIL = os.getenv("EMAIL", "openevolvetesting.worrier295@passmail.net")
BASE_DIR = Path(__file__).parent / "data"
MAX_PDF_WORKERS = 5

# Conference proceedings URLs
CONFERENCES = {
    "aaai": {
        "years": {
            2023: {
                "base_url": "https://ojs.aaai.org/index.php/AAAI/issue/view",
                "issue_ids": list(range(555, 575)),  # Vol 37, issues 1-20
            },
            2024: {
                "base_url": "https://ojs.aaai.org/index.php/AAAI/issue/view",
                "issue_ids": list(range(576, 597)),  # Vol 38, issues 1-21
            },
        },
    },
    "ijcai": {
        "years": {
            2023: {
                "proceedings_url": "https://www.ijcai.org/proceedings/2023",
            },
            2024: {
                "proceedings_url": "https://www.ijcai.org/proceedings/2024",
            },
        },
    },
}

# Setup logging
BASE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR.parent / "download_proceedings.log"),
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
    pdf_url: Optional[str] = None
    page_url: Optional[str] = None
    doi: Optional[str] = None
    downloaded: bool = False


class AAAProceedingsScraper:
    """Scraper for AAAI OJS proceedings"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

    def get_papers(self, year: int, config: dict) -> List[Paper]:
        """Scrape all papers from AAAI proceedings"""
        logger.info(f"Scraping AAAI {year} proceedings from OJS...")

        all_papers = []
        base_url = config["base_url"]
        issue_ids = config["issue_ids"]

        for issue_id in tqdm(issue_ids, desc=f"AAAI {year} issues"):
            time.sleep(1)  # Be polite
            papers = self._scrape_issue(base_url, issue_id, year)
            all_papers.extend(papers)

        logger.info(f"Found {len(all_papers)} papers from AAAI {year}")
        return all_papers

    def _scrape_issue(self, base_url: str, issue_id: int, year: int) -> List[Paper]:
        """Scrape papers from a single issue"""
        try:
            url = f"{base_url}/{issue_id}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            papers = []

            # Find all article entries
            articles = soup.find_all("div", class_="obj_article_summary")

            for article in articles:
                try:
                    # Extract title
                    title_elem = article.find("h3", class_="title")
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)

                    # Extract article URL
                    title_link = title_elem.find("a")
                    article_url = title_link.get("href") if title_link else None

                    # Extract authors
                    authors_elem = article.find("div", class_="authors")
                    authors = []
                    if authors_elem:
                        author_text = authors_elem.get_text(strip=True)
                        authors = [a.strip() for a in author_text.split(",")]

                    # Find PDF link
                    pdf_url = None
                    pdf_link = article.find("a", class_="pdf")
                    if pdf_link:
                        pdf_url = pdf_link.get("href")

                    # Generate paper ID
                    paper_id = f"aaai_{year}_{len(papers) + 1}"

                    papers.append(
                        Paper(
                            paper_id=paper_id,
                            title=title,
                            authors=authors,
                            year=year,
                            venue=f"AAAI {year}",
                            pdf_url=pdf_url,
                            page_url=article_url,
                        )
                    )

                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")
                    continue

            return papers

        except Exception as e:
            logger.error(f"Failed to scrape issue {issue_id}: {e}")
            return []


class IJCAIProceedingsScraper:
    """Scraper for IJCAI proceedings"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"OpenEvolve-PaperDownloader ({EMAIL})"})

    def get_papers(self, year: int, config: dict) -> List[Paper]:
        """Scrape all papers from IJCAI proceedings"""
        logger.info(f"Scraping IJCAI {year} proceedings...")

        url = config["proceedings_url"]
        time.sleep(1)

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            papers = []

            # IJCAI proceedings structure varies, try multiple selectors
            # Look for paper entries
            paper_divs = soup.find_all("div", class_="paper_wrapper")

            if not paper_divs:
                # Try alternative structure
                paper_divs = soup.find_all("div", class_="paper")

            for idx, paper_div in enumerate(paper_divs):
                try:
                    # Extract title
                    title_elem = paper_div.find("div", class_="title")
                    if not title_elem:
                        title_elem = paper_div.find("span", class_="title")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    # Extract authors
                    authors_elem = paper_div.find("div", class_="authors")
                    if not authors_elem:
                        authors_elem = paper_div.find("span", class_="authors")

                    authors = []
                    if authors_elem:
                        author_text = authors_elem.get_text(strip=True)
                        authors = [a.strip() for a in re.split(r"[,;]", author_text)]

                    # Find PDF link
                    pdf_url = None
                    pdf_link = paper_div.find("a", href=re.compile(r"\.pdf$"))
                    if pdf_link:
                        pdf_url = pdf_link.get("href")
                        # Handle relative URLs - IJCAI uses paths like "0001.pdf"
                        if not pdf_url.startswith("http"):
                            # If it's just a filename, prepend the proceedings path
                            if not pdf_url.startswith("/"):
                                pdf_url = f"{url}/{pdf_url}"
                            else:
                                pdf_url = f"https://www.ijcai.org{pdf_url}"

                    paper_id = f"ijcai_{year}_{idx + 1}"

                    papers.append(
                        Paper(
                            paper_id=paper_id,
                            title=title,
                            authors=authors,
                            year=year,
                            venue=f"IJCAI {year}",
                            pdf_url=pdf_url,
                        )
                    )

                except Exception as e:
                    logger.debug(f"Failed to parse paper: {e}")
                    continue

            logger.info(f"Found {len(papers)} papers from IJCAI {year}")
            return papers

        except Exception as e:
            logger.error(f"Failed to scrape IJCAI {year}: {e}")
            return []


class PaperDownloader:
    """Main downloader"""

    def __init__(self):
        self.aaai_scraper = AAAProceedingsScraper()
        self.ijcai_scraper = IJCAIProceedingsScraper()
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
        logger.info("Conference Proceedings Scraper (AAAI, IJCAI)")
        logger.info("=" * 70)
        logger.info(f"Output: {BASE_DIR}\n")

        start_time = time.time()

        # Process AAAI
        self.process_aaai()

        # Process IJCAI
        self.process_ijcai()

        elapsed = time.time() - start_time
        self.print_summary(elapsed)

    def process_aaai(self):
        """Process all AAAI years"""
        logger.info(f"\n{'=' * 70}")
        logger.info("Processing AAAI")
        logger.info(f"{'=' * 70}")

        self.stats["by_conference"]["aaai"] = {"papers": 0, "pdfs": 0, "failed": 0}

        for year, config in CONFERENCES["aaai"]["years"].items():
            papers = self.aaai_scraper.get_papers(year, config)
            self.save_and_download("aaai", year, papers)

    def process_ijcai(self):
        """Process all IJCAI years"""
        logger.info(f"\n{'=' * 70}")
        logger.info("Processing IJCAI")
        logger.info(f"{'=' * 70}")

        self.stats["by_conference"]["ijcai"] = {"papers": 0, "pdfs": 0, "failed": 0}

        for year, config in CONFERENCES["ijcai"]["years"].items():
            papers = self.ijcai_scraper.get_papers(year, config)
            self.save_and_download("ijcai", year, papers)

    def save_and_download(self, conf_name: str, year: int, papers: List[Paper]):
        """Save metadata and download PDFs"""
        if not papers:
            logger.warning(f"No papers found for {conf_name} {year}")
            return

        output_dir = BASE_DIR / conf_name / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading PDFs for {len(papers)} papers...")
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
            if not paper.pdf_url:
                return False, paper

            pdf_path = output_dir / f"{self._sanitize(paper.paper_id)}.pdf"

            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                paper.downloaded = True
                return True, paper

            if self.download_pdf(paper.pdf_url, pdf_path):
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

    def download_pdf(self, url: str, output_path: Path) -> bool:
        """Download single PDF"""
        try:
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if output_path.stat().st_size > 1000:
                with open(output_path, "rb") as f:
                    if f.read(4) == b"%PDF":
                        return True

            output_path.unlink()
            return False

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

        logger.info(f"\nTime: {elapsed / 60:.1f} minutes")
        logger.info(f"Output: {BASE_DIR}")
        logger.info("=" * 70)


def main():
    try:
        # Check if beautifulsoup4 is installed
        try:
            import bs4
        except ImportError:
            logger.error("BeautifulSoup4 not installed!")
            logger.error("Run: pip install beautifulsoup4")
            sys.exit(1)

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
