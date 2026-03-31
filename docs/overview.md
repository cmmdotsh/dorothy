# Dorothy — Overview

Dorothy is a news aggregator that synthesizes balanced coverage from 40+ sources across the political spectrum. It fetches RSS feeds, clusters similar articles by semantic similarity, and generates neutral summaries that incorporate perspectives from across the bias spectrum. It also publishes a daily audio briefing in the style of NPR radio.

**Tagline:** "A newspaper of averages."

## What it does

1. **Fetches** 40+ RSS feeds hourly from outlets spanning left to right
2. **Embeds** article headlines and summaries into 1024-dimensional vectors
3. **Clusters** similar articles using HDBSCAN density clustering into coherent stories
4. **Synthesizes** each multi-source story via LLM into a neutral news article plus a coverage analysis
5. **Ranks** stories by a hotness score that rewards freshness and source diversity
6. **Renders** a static HTML site: front page, per-column pages, and per-story detail pages
7. **Generates** an NPR-style audio briefing via text-to-speech
8. **Publishes** the site and podcast feed to AWS S3 + CloudFront

## Columns

Dorothy organizes content into five editorial columns:

| Column | Description |
|--------|-------------|
| `politics` | U.S. political news, grouped by bias rating |
| `tech` | Technology news, grouped by editorial perspective (consumer/enterprise/academic/culture) |
| `money` | Business, finance, and economy |
| `sports` | Sports news, grouped by geographic region |
| `lifestyle` | Culture, health, food, and human interest |

## Bias model

Sources are rated on the AllSides five-point scale:

| Rating | Examples |
|--------|---------|
| `left` | The Intercept, HuffPost |
| `lean-left` | NYT, Washington Post, Guardian, NPR |
| `center` | AP, Reuters, PBS, USA Today |
| `lean-right` | WSJ, Washington Times |
| `right` | Fox News, Breitbart, Newsmax |

Every synthesized story shows which bias tiers contributed coverage, so readers can see which outlets covered (or ignored) a story.

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Storage | OpenSearch 2.11 (k-NN via HNSW) |
| Embeddings | LMStudio (OpenAI-compatible API) |
| LLM | LMStudio (OpenAI-compatible API) |
| Clustering | HDBSCAN + scikit-learn |
| Web framework | FastAPI + Jinja2 |
| TTS | mlx-audio Chatterbox Turbo |
| Infrastructure | Docker, AWS S3 + CloudFront |
| IaC | Terraform |

## Repository layout

```
dorothy/
├── src/                   # Application library
│   ├── config.py          # Configuration (pydantic-settings)
│   ├── models/            # Data models (Article, Source, enums)
│   ├── storage/           # OpenSearch client
│   ├── fetcher/           # RSS feed fetching
│   ├── embeddings/        # Embedding API client + batch generator
│   ├── clustering/        # HDBSCAN story grouper
│   ├── synthesis/         # LLM client + story summarizer
│   ├── podcast/           # Script writer, TTS, audio assembly, RSS feed
│   └── web/               # FastAPI app + Jinja2 templates + static assets
├── scripts/               # Runnable entry points
│   ├── run_pipeline.py    # Main orchestrator (daemon or one-shot)
│   ├── run_fetch.py       # Fetch only
│   ├── run_embeddings.py  # Embeddings only
│   ├── run_synthesis.py   # Synthesis only
│   ├── run_podcast.py     # Podcast generation (daemon or one-shot)
│   ├── run_server.py      # FastAPI dev server
│   ├── render_static.py   # Static HTML renderer
│   └── deploy_s3.py       # S3 + CloudFront deployment
├── config/
│   ├── sources.yaml       # Source registry (name, URL, bias, column)
│   └── voices/            # TTS voice reference WAV files
├── Dockerfile             # Main pipeline image
├── Dockerfile.podcast     # Podcast image (TTS dependencies)
├── docker-compose.yml     # Full development/production environment
├── pyproject.toml         # Python dependencies
└── terraform/             # AWS infrastructure as code
```

## Documentation index

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Data flow, indices, algorithms |
| [configuration.md](configuration.md) | All environment variables and config options |
| [pipeline.md](pipeline.md) | Fetch → Embed → Cluster → Synthesize pipeline |
| [web.md](web.md) | FastAPI app and static site generation |
| [podcast.md](podcast.md) | Audio briefing generation |
| [deployment.md](deployment.md) | Docker, S3, and production deployment |
| [development.md](development.md) | Local development setup |
