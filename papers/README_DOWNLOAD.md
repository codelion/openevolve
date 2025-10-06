# Paper Download Script

Automated download of conference papers from NeurIPS, AAAI, IJCAI, and AAMAS (2023-2024/2025).

## Quick Start

### 1. Get API Key

**Semantic Scholar API Key** (Required):
- Visit: https://api.semanticscholar.org
- Request an API key (free)
- Typically approved within 24-48 hours
- Increases rate limit to ~1 request/second

### 2. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Edit .env file and add your API key
# Replace: SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
# With:    SEMANTIC_SCHOLAR_API_KEY=abc123yourkeyhere
```

### 3. Run Download

```bash
python download_papers.py
```

## What It Does

The script will:

1. **Query multiple sources** for papers:
   - OpenReview (NeurIPS, AAMAS)
   - Semantic Scholar (all conferences)
   - ArXiv (fallback for PDFs)
   - Unpaywall (fallback for open access)

2. **Download PDFs** to organized directories:
   ```
   papers/data/
   ├── neurips/
   │   ├── 2023/
   │   │   ├── metadata.json
   │   │   ├── s2_paper1.pdf
   │   │   └── openreview_paper2.pdf
   │   └── 2024/
   ├── aaai/
   ├── ijcai/
   └── aamas/
   ```

3. **Save metadata** for each conference/year in `metadata.json`:
   - Paper title, authors, abstract
   - PDF URLs and identifiers (ArXiv ID, DOI, etc.)
   - Download status

4. **Generate logs**:
   - Console: Real-time progress with tqdm bars
   - File: `download.log` with detailed debug info

## Expected Results

- **Papers**: ~2,000-4,000 total across all conferences
- **PDFs**: 30-50% coverage (open access only)
- **Runtime**: 4-7 hours with proper rate limiting
- **Disk space**: 2-5 GB for PDFs

## Configuration

Edit `download_papers.py` to customize:

```python
# Change conferences/years
CONFERENCES = {
    "neurips": {"years": [2023, 2024]},
    # Add more...
}

# Adjust rate limits (seconds between requests)
RATE_LIMITS = {
    "semantic_scholar": 1.0,
    "arxiv": 3.5,
}
```

## Troubleshooting

### No papers found
- Check venue names in `CONFERENCES` dict
- Some conferences may have different naming conventions
- Check logs for API errors

### Low PDF coverage
- Normal! Many papers are not open access
- ArXiv papers have higher coverage (~80%)
- Conference papers vary widely (10-60%)

### Rate limit errors
- Increase delays in `RATE_LIMITS` dict
- Semantic Scholar: max 1 req/sec with free API key
- ArXiv: recommends 3 seconds between requests

### API key errors
```
ValueError: SEMANTIC_SCHOLAR_API_KEY not set in .env file
```
- Make sure `.env` file exists in `papers/` directory
- Check that API key is valid (not the placeholder text)
- Verify no extra spaces around the key

## Data Sources

### OpenReview
- **Conferences**: NeurIPS, AAMAS
- **Coverage**: ~90% of papers with PDFs
- **Rate limit**: 0.5 seconds between requests
- **No API key required**

### Semantic Scholar
- **Conferences**: All (primary source)
- **Coverage**: Good metadata, variable PDF access
- **Rate limit**: 1 req/sec with API key
- **API key**: Required (free)

### ArXiv
- **Use**: Fallback for papers with ArXiv IDs
- **Coverage**: ~80% for papers on ArXiv
- **Rate limit**: 3.5 seconds (conservative)
- **No API key required**

### Unpaywall
- **Use**: Fallback for papers with DOIs
- **Coverage**: Variable (~20-40%)
- **Rate limit**: 1 second
- **No API key required**

## Resuming Downloads

The script automatically skips already-downloaded PDFs. To resume:

```bash
# Just run again - it will skip existing files
python download_papers.py
```

## Output Format

### metadata.json
```json
{
  "conference": "neurips",
  "year": 2023,
  "total_papers": 1234,
  "downloaded": 567,
  "failed": 667,
  "papers": [
    {
      "paper_id": "s2_abc123",
      "title": "Paper Title",
      "authors": ["Author 1", "Author 2"],
      "year": 2023,
      "venue": "NeurIPS 2023",
      "abstract": "...",
      "pdf_url": "https://...",
      "arxiv_id": "2301.12345",
      "doi": "10.1234/xyz",
      "downloaded": true
    }
  ]
}
```

## Tips

- **Run overnight**: Download takes several hours
- **Check logs**: `download.log` has detailed error messages
- **Incremental**: Add conferences/years and re-run to expand dataset
- **Filter later**: Download everything, filter by topic/keywords afterwards

## Next Steps

After downloading, you can:
1. **Filter papers** by keywords in title/abstract
2. **Extract text** from PDFs for analysis
3. **Build embeddings** for similarity search
4. **Create training data** for OpenEvolve experiments

See [BACKGROUND.md](BACKGROUND.md) for more details on using this dataset.
