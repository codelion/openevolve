# APIs and Tools for Downloading Conference Papers: A Practical Implementation Guide

**Bottom line**: You can technically download 4,000 papers from AAAI, IJCAI, NeurIPS, and AAMAS in one day using a combination of official APIs and third-party services. The best approach uses **OpenReview API for NeurIPS** (2021+), **Semantic Scholar API as the primary aggregator** for all conferences, **ArXiv API for preprints**, and **Unpaywall for open access PDFs**. However, several conferences lack official APIs, and legal/ethical considerations require using designated access methods rather than web scraping.

## Official conference APIs: limited availability requires workarounds

**NeurIPS stands alone with comprehensive API access** through OpenReview for papers from 2021 onward. The platform offers two REST APIs (v1 for 2021-2022, v2 for 2023+) with no authentication required for public papers, no explicit rate limits, and a maximum of 1,000 items per request requiring pagination for larger datasets. The OpenReview Python client (`openreview-py`) provides straightforward methods to download both PDFs and metadata programmatically. Pre-2021 NeurIPS papers reside on papers.nips.cc without an official API, requiring web scraping if needed.

**AAAI and IJCAI offer open access but no APIs**. Both conferences publish proceedings freely at ojs.aaai.org and ijcai.org/proceedings respectively, with individual papers available as PDFs alongside BibTeX metadata. While technically accessible, their Terms of Service explicitly prohibit bulk downloading without authorization. AAAI's policy states that "downloading significant portions of the Digital library for any purpose is prohibited," and IJCAI requires written permission for reproduction. Neither provides compressed archives or bulk download options.

**AAMAS presents the most restrictive access scenario**. Papers from 2007 onward appear on the IFAAMAS website (ifaamas.org/Proceedings) and are also indexed in ACM Digital Library for 2002-2006 and recent years. Critically, ACM explicitly prohibits automated downloading and scraping, warning that violations result in "temporary or permanent termination of download rights." Recent AAMAS conferences (2025-2026) appear on OpenReview with full API access, but historical papers require manual access or respectful web scraping from IFAAMAS with appropriate delays.

## Google Scholar API: officially nonexistent with risky alternatives

**No official Google Scholar API exists**, and Google has no announced plans to release one. The company explicitly prohibits automated scraping in its Terms of Service. Several commercial services (SerpAPI, ScrapingDog, Oxylabs) provide unofficial access by proxying requests, with prices ranging from $0.001 to $0.015 per request, but these violate Google's ToS and carry legal risks plus reliability concerns when Google changes page structures. The open-source Scholarly Python package offers free scraping but risks IP blocking. For production systems requiring legal compliance, Semantic Scholar or OpenAlex provide superior alternatives with official APIs and comparable coverage.

## ArXiv API: excellent for preprints but incomplete conference coverage

The ArXiv API provides comprehensive access with **no authentication required** and a conservative rate limit of **1 request per 3 seconds** (or up to 4 requests/second with bursting and 1-second sleeps). The API returns Atom XML format with metadata and PDF links accessible at `export.arxiv.org/pdf/[arxiv_id].pdf`. For bulk access, Amazon S3 buckets contain the complete repository (~9.2 TB total, with PDFs comprising ~2.7 TB) via requester-pays access.

**Conference coverage varies significantly by field and community practices**. Machine learning venues like NeurIPS, ICML, and ICLR achieve 80-95% ArXiv coverage as authors routinely post preprints before submission. AAAI and IJCAI show moderate coverage at approximately 60-80%, while AAMAS has lower adoption at 40-60% due to more diverse publication practices in the multi-agent systems community. The critical limitation: ArXiv has no native conference field, requiring searches by conference name in abstracts and comments using queries like `all:NeurIPS OR all:NIPS`.

## Third-party aggregators: Semantic Scholar emerges as the clear leader

**Semantic Scholar API provides the best comprehensive solution** for conference paper access. Developed by the Allen Institute for AI, it indexes 214+ million papers with explicit coverage of AAAI, IJCAI, NeurIPS, and AAMAS. The API offers rich metadata including abstracts, citations, author information, SPECTER2 embeddings, and `openAccessPdf` fields linking to freely available PDFs when they exist. Rate limits start at 100 requests per 5 minutes for unauthenticated users, improving to 1 request per second with a free API key obtainable through their website, with higher limits available upon request for specific research projects. The service provides JSON responses, supports advanced filtering, and offers bulk dataset downloads for large-scale analysis.

**OpenAlex serves as an excellent alternative**, particularly for researchers familiar with the retired Microsoft Academic API. This non-profit service covers 209+ million works with no authentication required and a recommended limit of 100,000 requests per day. Including an email parameter grants access to the "polite pool" with faster, more consistent response times. The platform provides open access indicators through Unpaywall integration and releases complete dataset snapshots biweekly under CC0 license, making it ideal for very large-scale projects.

**Papers with Code API** complements these services by linking papers to code implementations, though coverage limits to papers with associated GitHub repositories—valuable for reproducibility research but incomplete for general conference paper collections. The platform requires no authentication for read access and covers major ML conferences well, though with less comprehensive metadata than Semantic Scholar.

**Unpaywall API deserves special mention** for PDF acquisition. This service accepts DOIs and returns direct URLs to open access PDFs when available, supporting 100,000 API calls per day with just an email parameter. The workflow: obtain DOIs from Semantic Scholar or OpenAlex, then pipe them through Unpaywall to locate freely available PDFs. CORE API provides another option for open access papers, delivering actual full-text access to 37 million papers from 10,000+ repositories, though free tier rate limits prove very restrictive (1 batch or 5 single requests per 10 seconds).

**Microsoft Academic API retired on December 31, 2021**. Former users should migrate to OpenAlex, which inherited the Microsoft Academic Graph data and provides similar functionality with an open, community-driven model.

## Rate limits and one-day feasibility: technically possible but constrained by access

**Technical feasibility is straightforward**. With proper API implementation, downloading 4,000 papers in one day ranges from 7 minutes to 3.5 hours depending on source:

- **OpenReview**: 1,000 items per request, no explicit rate limit → Under 1 hour for 4,000 papers
- **Semantic Scholar**: 1 request/second with API key → 1.1 hours for 4,000 papers  
- **ArXiv**: 1 request/3 seconds conservatively → 3.3 hours, or 17 minutes with optimized bursting
- **Unpaywall**: 100,000 requests/day → 7 minutes for 4,000 papers

**The real constraint is PDF availability, not API speed**. Third-party APIs provide metadata readily but only link to PDFs—they don't host them. Open access coverage varies: approximately 30-50% of conference papers have freely available PDFs through repositories, arXiv, or publisher sites. The remaining papers require institutional subscriptions or direct author contact.

**Practical strategy for 1,000 papers per conference** (4,000 total):

1. **NeurIPS (1,000 papers)**: Use OpenReview API for 2021+ papers. Single API call retrieves all accepted papers for a year, then download PDFs individually. Estimated time: 30-45 minutes per year with rate limiting.

2. **AAAI (1,000 papers)**: Query Semantic Scholar API filtering by venue "AAAI". Retrieve metadata in batches of 100, then download PDFs from links provided or ArXiv. Estimated time: 1-2 hours with conservative rate limiting.

3. **IJCAI (1,000 papers)**: Similar Semantic Scholar workflow. Estimated time: 1-2 hours.

4. **AAMAS (1,000 papers)**: Semantic Scholar or OpenAlex API, supplemented by OpenReview for 2025-2026 papers. Estimated time: 1-2 hours.

**Total estimated time: 4-7 hours** with proper implementation, rate limiting, and error handling—comfortably achievable in one day.

## Legal and ethical considerations require designated access methods

**Fair use does not protect bulk downloading**. While U.S. copyright law permits using papers for research under fair use doctrine, this applies to reading and analyzing content, not systematic mass collection. Most academic papers are copyrighted by publishers or authors, and Terms of Service for digital libraries explicitly prohibit bulk downloading even with institutional subscriptions.

**Critical prohibitions across major platforms**:

- **ACM Digital Library**: "Using scripts or spiders to automatically download articles" constitutes a serious violation resulting in account termination
- **IEEE Xplore**: Explicitly prohibits systematic downloading, robots, or creating searchable archives  
- **PubMed Central**: Blocks IP ranges attempting bulk downloads via the main website
- **AAAI/IJCAI**: Prohibit downloading "significant portions" without authorization

**Violating robots.txt carries legal risk**. While robots.txt itself isn't legally binding, violating it can support legal claims under the Computer Fraud and Abuse Act (CFAA) for unauthorized access, Terms of Service violations, and trespass to chattels. Recent precedent shows services successfully prosecuting violators.

**The ethical framework demands respect for infrastructure**. Excessive scraping degrades service for legitimate users, increases costs for non-profit repositories, and can trigger institutional IP blocks. PubMed Central reports automatic blocking of bulk downloaders, and 90% of open access repositories report problems with AI bot scraping. Following best practices protects the scholarly infrastructure that enables research.

**Recommended ethical approach**:

1. **Use official APIs exclusively**—OpenReview, ArXiv, Semantic Scholar, OpenAlex
2. **Obtain API keys** where available and request rate limit increases with research justification
3. **Implement conservative rate limiting** that exceeds stated minimums
4. **Set descriptive User-Agent strings** including contact email
5. **Document methods thoroughly** for research transparency  
6. **Download only what you need** for your specific project
7. **Never redistribute** bulk collections without permission
8. **Contact repository administrators** for large-scale projects to request proper access

## Alternative approaches when direct APIs unavailable

**Official bulk data services provide the most legitimate path** for complete collections. ArXiv offers Amazon S3 access to the entire repository (~9.2 TB) via requester-pays buckets, with PDFs organized in ~500MB tar files. PubMed Central provides an FTP service for its Open Access subset. Semantic Scholar and OpenAlex release complete dataset snapshots biweekly, suitable for projects requiring tens of thousands of papers.

**Conference proceedings downloads vary by venue**. Some conferences historically provided USB drives to attendees with complete proceedings, though this practice has declined. OpenReview venues include batch download features for accepted papers. For other conferences, contact organizers directly with research justification—many will provide access for legitimate academic purposes.

**Web scraping frameworks exist but carry risks**. Scrapy provides enterprise-grade scraping with built-in politeness features, while Beautiful Soup + Requests offers simpler parsing for smaller projects. Selenium/Playwright handle JavaScript-heavy sites requiring browser automation. However, none of these tools make scraping legal if it violates Terms of Service—they're technical tools that must be deployed within legal constraints.

**The respectful scraping approach** when APIs don't exist: (1) Check robots.txt and respect directives, (2) implement delays exceeding stated minimums (5-10 seconds between requests), (3) use off-peak hours, (4) set proper User-Agent with contact info, (5) handle errors gracefully with exponential backoff, (6) cache results to avoid re-downloading, and (7) monitor server response times to detect if you're causing problems.

## Format availability: PDFs accessible but metadata more universally available

**All services provide JSON metadata as the primary format**. OpenReview API returns comprehensive JSON objects including titles, abstracts, authors, reviews, and decision information. Semantic Scholar, OpenAlex, Crossref, and CORE all use JSON as their native format. ArXiv returns Atom XML by default, requiring parsing with libraries like `feedparser` to convert to JSON.

**PDF access follows a hierarchy of availability**:

- **OpenReview**: Direct PDF downloads via `get_attachment()` method for all papers on platform
- **ArXiv**: Full PDF access for all papers at `export.arxiv.org/pdf/[id].pdf`  
- **Semantic Scholar/OpenAlex**: Provide URLs to PDFs hosted elsewhere when open access versions exist
- **CORE**: Actually serves full-text PDFs for 37 million papers, not just links
- **Unpaywall**: Returns direct URLs to open access PDFs at publisher sites, repositories, or preprint servers

**BibTeX and other citation formats** are universally available. OpenReview paper pages include BibTeX export, ArXiv provides BibTeX at specific URLs (`papers.nips.cc/paper/{year}/hash/{hash}-Abstract-Bibtex.bib`), and services like Crossref support content negotiation for multiple formats (JSON, BibTeX, RDF, CSL, XML).

**LaTeX source availability is limited**. ArXiv includes source files for many papers in separate S3 buckets (~2.9 TB total), accessible via the same requester-pays model as PDFs. OpenReview authors may include source in supplementary materials but this isn't standard. Publishers generally don't provide LaTeX source.

**Supplementary materials** follow paper-specific patterns. OpenReview allows downloading supplementary files (often ZIP archives) via the same `get_attachment()` method. ArXiv includes supplementary files when authors upload them. Journal and conference proceedings rarely provide supplementary materials through APIs.

## Authentication requirements: minimal barriers for most services

**Most academic APIs require no authentication for basic access**. ArXiv, OpenAlex, Crossref, and DBLP operate completely openly with no registration required. OpenReview allows unauthenticated access to public papers. This open model supports the research community and reduces friction for legitimate scholarship.

**API keys improve service quality when available**. Semantic Scholar offers free API keys that increase rate limits from shared 100 requests/5 minutes to dedicated 1 request/second, with higher limits available upon request. Registering an OpenReview account (free) provides better access to certain features. These keys require simple web form submission with project description, typically approved within 24-48 hours.

**Email parameters unlock "polite pool" access** at several services. OpenAlex, Crossref, and Unpaywall all prioritize requests that include an email parameter or User-Agent with contact information, routing these to dedicated server pools with faster, more consistent response times. This costs nothing and significantly improves performance.

**Institutional subscriptions remain necessary for paywalled content**. While APIs provide metadata for all papers, accessing PDFs behind paywalls requires institutional licenses or individual subscriptions. ACM Digital Library, IEEE Xplore, and other publisher platforms provide access through IP-based authentication or Shibboleth for university affiliates. However, even with institutional access, Terms of Service prohibit bulk automated downloading—subscriptions license reading and individual downloads, not mass collection.

**OAuth and token authentication** appears in some specialized services. Papers with Code's write API requires tokens for competition mirroring. Some institutional repository APIs use OAuth for authorization. Commercial services like SerpAPI for unofficial Google Scholar access require paid API keys.

## Recommended implementation strategy for your project deadline

**For immediate implementation before a project deadline**, use this multi-source approach:

**Phase 1: Metadata Collection (2-3 hours)**

1. **Register for API keys immediately**: 
   - Semantic Scholar: Submit form at api.semanticscholar.org
   - Consider OpenAlex as backup (no key needed)

2. **Query each conference systematically**:
   ```python
   # NeurIPS via OpenReview (2021+)
   client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
   papers = client.get_all_notes(content={'venueid': 'NeurIPS.cc/2023/Conference'})
   
   # AAAI/IJCAI/AAMAS via Semantic Scholar
   # Use venue filter or search query
   ```

3. **Store metadata in local database**: SQLite or CSV with fields for ID, title, authors, abstract, DOI, PDF URL, ArXiv ID

**Phase 2: PDF Acquisition (3-4 hours)**

1. **For NeurIPS**: Download directly from OpenReview API
2. **For papers with ArXiv IDs**: Download from ArXiv with 3-second delays
3. **For papers with DOIs**: Query Unpaywall API for open access PDFs
4. **For remaining papers**: Check Semantic Scholar's `openAccessPdf` field

**Phase 3: Gap Filling (1-2 hours)**

1. **Manual downloads** for high-priority papers without open access
2. **Contact authors** via provided email addresses for unavailable papers
3. **Check institutional repository** links provided by CORE or Unpaywall

**Code template for implementation**:

```python
import openreview
import arxiv
import requests
import time
from typing import Dict, List

class ConferencePaperDownloader:
    def __init__(self, output_dir='papers'):
        self.output_dir = output_dir
        self.s2_api_key = 'YOUR_SEMANTIC_SCHOLAR_KEY'
        
    def download_neurips(self, year: int):
        """Download NeurIPS papers via OpenReview"""
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        venue_id = f'NeurIPS.cc/{year}/Conference'
        papers = client.get_all_notes(content={'venueid': venue_id})
        
        for paper in papers:
            if paper.content.get('pdf'):
                pdf = client.get_attachment(field_name='pdf', id=paper.id)
                filename = f"{self.output_dir}/neurips{year}_{paper.number}.pdf"
                with open(filename, 'wb') as f:
                    f.write(pdf)
            time.sleep(1)  # Be polite
    
    def download_via_semantic_scholar(self, venue: str, limit=1000):
        """Download papers via Semantic Scholar API"""
        headers = {'x-api-key': self.s2_api_key}
        base_url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        
        params = {
            'query': f'venue:{venue}',
            'fields': 'paperId,title,authors,abstract,openAccessPdf,externalIds',
            'limit': 100,
            'offset': 0
        }
        
        papers = []
        while len(papers) < limit:
            response = requests.get(base_url, params=params, headers=headers)
            data = response.json()
            papers.extend(data.get('data', []))
            
            if not data.get('next'):
                break
            params['offset'] += 100
            time.sleep(1)  # Rate limiting
        
        return papers
    
    def download_from_arxiv(self, arxiv_id: str):
        """Download paper from ArXiv"""
        client = arxiv.Client(delay_seconds=3.5)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        paper.download_pdf(filename=f"{self.output_dir}/{arxiv_id}.pdf")
        
    def check_unpaywall(self, doi: str) -> str:
        """Check Unpaywall for open access PDF"""
        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {'email': '[email protected]'}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('best_oa_location'):
                return data['best_oa_location'].get('url_for_pdf')
        return None
```

**Critical success factors**:

- **Start immediately with API key requests** (24-48 hour approval time)
- **Use OpenReview for NeurIPS** as primary source (best API)
- **Rely on Semantic Scholar** for AAAI/IJCAI/AAMAS metadata
- **Layer in ArXiv and Unpaywall** for additional PDF coverage
- **Implement robust error handling** with retries and logging
- **Monitor progress** with clear logging to identify bottlenecks
- **Accept 30-50% PDF coverage** as realistic—don't block on 100% completion

## Conclusion: a layered approach delivers best results

The optimal strategy combines **OpenReview API for NeurIPS** (offering the best official conference access), **Semantic Scholar API as the primary aggregator** across all four conferences, **ArXiv API for preprint supplements**, and **Unpaywall for open access PDF discovery**. This multi-source approach maximizes both metadata completeness (90-95% coverage expected) and PDF availability (30-50% open access), while respecting legal and ethical constraints through designated API channels.

Technical feasibility for downloading 4,000 papers in one day is excellent—4 to 7 hours with proper implementation. However, the real timeline constraint involves API key approval (request immediately) and the reality that not all papers have freely available PDFs. Success requires starting with metadata collection across all sources, then pursuing PDFs through multiple channels, accepting that some papers will require institutional access or author contact.

The research community increasingly emphasizes ethical infrastructure use. By leveraging official APIs with proper rate limiting, implementing polite access patterns with contact information, and respecting Terms of Service boundaries, your project can achieve its goals while supporting the open scholarly ecosystem that makes this research possible.