# Dorothy

**The Newspaper of Averages**

A news aggregator that synthesizes balanced coverage from 40+ sources across the political spectrum. Dorothy fetches news from left, center, and right-leaning outlets, clusters similar stories using semantic embeddings, then generates neutral summaries that incorporate all perspectives.

Live at [dorothy.cmm.sh](https://dorothy.cmm.sh)

## How it works

```
RSS Feeds -> OpenSearch -> Embeddings -> k-NN Clustering -> LLM Synthesis -> Static HTML -> S3/CloudFront
```

1. **Fetch** - RSS feeds from 40+ sources are parsed and stored in OpenSearch
2. **Embed** - Headlines are converted to 1024-dimensional vectors via a local embedding model
3. **Cluster** - k-NN search groups articles covering the same story
4. **Synthesize** - An LLM generates a neutral headline and summary from all perspectives
5. **Render** - Static site generator builds HTML from Jinja2 templates
6. **Deploy** - HTML is synced to S3 and served through CloudFront

## Front page

The front page uses a classic broadsheet newspaper layout:

- Masthead with dateline and volume number
- Above the fold: lead story with large image and summary, flanked by secondary stories in a sidebar column
- Below the fold: section blocks in a 3-column grid with thumbnails and headline links
- Column rules (vertical dividers) between sections, like a real newspaper
- Responsive -- collapses gracefully on tablet and mobile

## Bias coverage

Stories include bias attribution showing how many sources from each category covered the topic.

| Bias | Examples |
|------|----------|
| Left | The Intercept, HuffPost |
| Lean-Left | NPR, NYT, Atlantic |
| Center | AP, BBC, Reuters |
| Lean-Right | WSJ, Washington Times |
| Right | Fox News, Breitbart, Newsmax |

## Columns

- **Politics** - Government, elections, policy
- **Tech** - Technology, AI, startups
- **Money** - Business, markets, economy
- **Sports** - All sports coverage
- **Lifestyle** - Entertainment, culture, health

## Running it

### Prerequisites

- Python 3.13
- OpenSearch with k-NN plugin (see `docker-compose.yml`)
- LLM server (LM Studio or similar) for synthesis
- AWS credentials for deployment

### Commands

```bash
source .venv/bin/activate

# full pipeline (fetch + cluster + synthesize)
python -m scripts.run_pipeline --once

# daemon mode (runs every 60 min)
python -m scripts.run_pipeline

# dev server at http://localhost:8000
python -m scripts.run_server

# render static site to ./output/
python -m scripts.render_static --clean

# deploy to s3 + invalidate cloudfront
set -a && source .env && set +a && python -m scripts.deploy_s3 --invalidate
```

## Stack

- Python 3.13
- FastAPI + Jinja2 (templates and dev server)
- OpenSearch 2.11 (storage, k-NN search)
- feedparser + httpx (RSS fetching)
- boto3 (S3 deployment)
- Terraform (AWS infrastructure)

## Project structure

```
config/sources.yaml          # all 40+ news sources with bias ratings
src/models/                  # article, source, enums
src/fetcher/                 # RSS feed parsing, image extraction
src/storage/                 # OpenSearch client
src/clustering/              # embedding generation, k-NN grouping
src/synthesis/               # LLM client, summary generation
src/web/                     # FastAPI server, templates, static assets
scripts/run_pipeline.py      # main daemon
scripts/render_static.py     # static site generator
scripts/deploy_s3.py         # S3 deployment
terraform/                   # AWS infra (S3, CloudFront, ACM)
```
