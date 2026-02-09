# Dorothy - Newspaper of Averages

A news aggregator that synthesizes balanced coverage from sources across the political spectrum.

## Project Overview

Dorothy fetches news from 40+ sources (left, center, right), clusters similar stories together using semantic embeddings, then uses an LLM to generate neutral summaries that incorporate all perspectives.

**Goal:** Combat filter bubbles by showing how different outlets cover the same story.

## Architecture

```
RSS Feeds → OpenSearch → Embeddings → k-NN Clustering → LLM Synthesis → Static HTML → S3/CloudFront
```

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Models** | `src/models/` | Article, Source, enums (BiasRating, Column) |
| **Fetcher** | `src/fetcher/rss.py` | RSS feed parsing, image extraction |
| **Storage** | `src/storage/opensearch.py` | OpenSearch client, article/synthesis indices |
| **Clustering** | `src/clustering/` | Embedding generation, k-NN story grouping |
| **Synthesis** | `src/synthesis/` | LLM client, balanced summary generation |
| **Web** | `src/web/` | FastAPI dev server, Jinja2 templates |
| **Static Gen** | `scripts/render_static.py` | Renders all pages to static HTML |
| **Deploy** | `scripts/deploy_s3.py` | Syncs static site to S3, CloudFront invalidation |
| **Infra** | `terraform/` | AWS infrastructure (S3, CloudFront) |
| **Config** | `src/config.py` + `config/sources.yaml` | Source registry, settings |

## Key Files

- `config/sources.yaml` - All 40 news sources with RSS URLs, bias ratings, columns
- `src/synthesis/summarizer.py` - LLM prompts for balanced synthesis
- `src/storage/opensearch.py` - Index mappings, CRUD operations
- `scripts/run_pipeline.py` - Main daemon (fetch → cluster → synthesize)
- `scripts/run_server.py` - FastAPI dev server entry point
- `scripts/render_static.py` - Static site generator (renders to `output/`)
- `scripts/deploy_s3.py` - S3 deployment with CloudFront invalidation

## Data Flow

1. **Fetch:** RSS feeds parsed, articles stored in `dorothy-articles-YYYY-MM` index
2. **Embed:** Headlines converted to 1024-dim vectors via local embedding model
3. **Cluster:** k-NN search groups articles about same story
4. **Synthesize:** LLM generates neutral headline + summary from all perspectives
5. **Store:** Syntheses saved to `dorothy-synthesis` index with:
   - `hero_image_url` - Best image (prefers center sources)
   - `articles` - Full source references with URLs, headlines, bias
   - `bias_coverage` - Count of articles per bias category
6. **Render:** Static site generator builds HTML pages from templates
7. **Deploy:** Static HTML synced to S3, served via CloudFront at dorothy.cmm.sh

## Running Dorothy

### Prerequisites
- OpenSearch running on localhost:9200 (see `docker-compose.yml`)
- LLM server (LMStudio) at http://192.168.0.149:1234
- Python 3.13 with venv at `.venv/`

### Commands

```bash
# Activate environment
source .venv/bin/activate

# Single pipeline run (fetch + synthesize all columns)
python -m scripts.run_pipeline --once

# Daemon mode (runs every 60 minutes)
python -m scripts.run_pipeline

# Web server (http://localhost:8000)
python -m scripts.run_server

# Fetch only (no synthesis)
python -m scripts.run_fetch

# Synthesize specific column
python -m scripts.run_synthesis --column politics --limit 10

# Render static site (outputs to ./output/)
python -m scripts.render_static --clean

# Deploy to S3 + invalidate CloudFront cache
# (requires .env with S3_BUCKET and CLOUDFRONT_ID)
set -a && source .env && set +a && python -m scripts.deploy_s3 --invalidate
```

## OpenSearch Indices

### `dorothy-articles-YYYY-MM`
Articles with fields: id, source_name, source_slug, source_bias, column, headline, summary, url, pub_date, fetched_at, embedding (1024-dim), image_url

### `dorothy-synthesis`
Synthesized stories with fields: story_id, column, generated_headline, summary, sources_used, bias_coverage, article_count, hero_image_url, articles (nested)

## Bias Categories

| Bias | Color | Examples |
|------|-------|----------|
| Left | `#3b82f6` | The Intercept, HuffPost |
| Lean-Left | `#60a5fa` | NPR, NYT, Atlantic |
| Center | `#a855f7` | AP, BBC, Reuters |
| Lean-Right | `#f97316` | WSJ, Washington Times |
| Right | `#ef4444` | Fox News, Breitbart, Newsmax |

## Columns

- **politics** - Government, elections, policy
- **tech** - Technology, AI, startups
- **money** - Business, markets, economy
- **sports** - All sports coverage
- **lifestyle** - Entertainment, culture, health

## Current Status

**Completed Phases:**
1. Ingest - 40 sources, RSS fetching with image extraction
2. Cluster - Embeddings + k-NN story grouping
3. Bias Metadata - source_bias field on all articles
4. Synthesize - LLM balanced summaries with bias attribution
5. Front Page - Broadsheet newspaper layout with above/below-the-fold design
6. Source Links & Hero Images - Article references + featured images
7. Static Site Generation - Renders to HTML, deploys to S3/CloudFront
8. Front Page Redesign - Classic newspaper masthead with dateline, 2-column lead/sidebar above fold, 3-column section grid below fold

**Live Site:** https://dorothy.cmm.sh

**Data Stats:**
- 193 synthesized story pages
- 5 column pages (politics, tech, money, sports, lifestyle)
- 20 stories per column, 2 for lifestyle

## Common Tasks

### Add a new source
Edit `config/sources.yaml`:
```yaml
- name: "Source Name"
  slug: source-slug
  rss_url: "https://..."
  fetch_method: rss
  column: politics
  bias: center
  active: true
```

### Re-synthesize after code changes
```bash
# Delete old syntheses
python -c "from src.storage import OpenSearchClient; OpenSearchClient().clear_syntheses()"

# Re-run synthesis
python -m scripts.run_pipeline --once
```

### Debug missing images
```bash
# Check article image coverage
python -c "
from src.storage import OpenSearchClient
client = OpenSearchClient()
articles = client.search_articles(size=1000)
with_img = sum(1 for a in articles if a.get('image_url'))
print(f'{with_img}/{len(articles)} articles have images')
"
```

## Front Page Layout

The front page uses a classic broadsheet newspaper design:

- **Masthead:** "Dorothy" in Old English font, tagline, dateline with date + "Vol. 1"
- **Above the fold:** Lead story (politics, ~60% width) with large image/headline/summary, flanked by 3 secondary stories (tech, money, sports) in a sidebar column (~40%) with vertical rule divider
- **Below the fold:** 3-column section grid with all 5 columns, each showing a thumbnail, 2-3 headline links, and "More X" link. Column rules between sections.
- **Responsive:** Collapses to single column on mobile, 2-column on tablet

Templates: `src/web/templates/front_page.html`, `base.html`
CSS: `src/web/static/style.css`

## Tech Stack

- **Python 3.13** - Runtime
- **FastAPI** - Web framework (dev server)
- **OpenSearch 2.11** - Search/storage with k-NN plugin
- **feedparser** - RSS parsing
- **httpx** - HTTP client
- **Jinja2** - HTML templates
- **boto3** - AWS S3 deployment
- **structlog** - Logging
- **schedule** - Daemon scheduling
- **Terraform** - AWS infrastructure (S3 bucket, CloudFront distribution)
