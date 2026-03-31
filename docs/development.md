# Dorothy — Development

## Prerequisites

- Python 3.11+
- [LMStudio](https://lmstudio.ai/) (or any OpenAI-compatible API) for LLM and embeddings
- Docker + Docker Compose (for OpenSearch)
- FFmpeg (for podcast audio assembly only)

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo>
cd dorothy
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Core dependencies
pip install -e .

# With podcast TTS support (requires torch + ffmpeg)
pip install -e ".[podcast]"

# With development tools (pytest, black, isort, mypy)
pip install -e ".[dev]"
```

### 3. Start OpenSearch

```bash
docker-compose up opensearch -d
```

OpenSearch will be available at `http://localhost:9200`. Verify it's running:

```bash
curl http://localhost:9200/_cluster/health
```

### 4. Configure environment

Create a `.env` file in the project root:

```dotenv
# LLM (point at your LMStudio instance)
LLM_BASE_URL=http://localhost:1234
LLM_MODEL=mlx-community/qwen3.5-35b-a3b
LLM_CONTEXT_LENGTH=32768

# Embeddings (same LMStudio, different model loaded)
EMBEDDING_BASE_URL=http://localhost:1234
EMBEDDING_MODEL=text-embedding-mxbai-embed-large-v1

# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
```

### 5. Load models in LMStudio

Dorothy requires two models loaded in LMStudio:

- **Embedding model:** `text-embedding-mxbai-embed-large-v1` (produces 1024-dim vectors)
- **LLM:** `mlx-community/qwen3.5-35b-a3b` (or any instruction-tuned model with 32K+ context)

LMStudio must have its local server enabled (default port: 1234).

## Running locally

### Full pipeline (one shot)

```bash
python -m scripts.run_pipeline --once
```

### Individual pipeline steps

```bash
# Step 1: Fetch RSS feeds
python -m scripts.run_fetch

# Step 2: Generate embeddings for articles
python -m scripts.run_embeddings

# Step 3: Synthesize stories for a column
python -m scripts.run_synthesis --column politics
python -m scripts.run_synthesis --column tech --stories 5
```

### Web server (live queries OpenSearch)

```bash
python -m scripts.run_server --reload
# → http://localhost:8000
```

### Render static site

```bash
python -m scripts.render_static --clean
# output/ directory is created with all HTML files
```

### Deploy to S3

```bash
# Dry run (preview what would be uploaded)
python -m scripts.deploy_s3 --dry-run

# Deploy
python -m scripts.deploy_s3 --invalidate
```

### Podcast generation

```bash
# Generate script only (no audio — good for testing)
python -m scripts.run_podcast --script-only

# Full generation
python -m scripts.run_podcast --stories 3

# With GPU
python -m scripts.run_podcast --device cuda
```

## Utility scripts

| Script | Description |
|--------|-------------|
| `scripts/visualize_clusters.py` | Interactive 2D cluster visualization via UMAP + Plotly |
| `scripts/backfill_similarity.py` | Compute similarity edges for existing syntheses |
| `scripts/dedup_syntheses.py` | Deduplicate existing syntheses in OpenSearch |
| `scripts/export_data.py` | Export OpenSearch indices to JSONL files |
| `scripts/import_data.py` | Import JSONL data into OpenSearch |

### Cluster visualization

```bash
# Visualize politics column clusters
python -m scripts.visualize_clusters

# Visualize tech column, limit to 500 articles
python -m scripts.visualize_clusters --column tech --size 500

# Custom output file
python -m scripts.visualize_clusters --output clusters.html
```

Opens an interactive Plotly scatter plot: each point is an article, colored by HDBSCAN cluster. Hover for headline and source.

### Similarity backfill

```bash
# Preview
python -m scripts.backfill_similarity --dry-run

# Backfill all columns
python -m scripts.backfill_similarity

# Specific column
python -m scripts.backfill_similarity --column politics
```

### Synthesis deduplication

```bash
# Preview duplicates without modifying anything
python -m scripts.dedup_syntheses

# Apply: mark duplicates as historical
python -m scripts.dedup_syntheses --apply
```

### Data export/import

```bash
# Export all indices to ./data/
python -m scripts.export_data
python -m scripts.export_data --output-dir ./backup

# Import from ./data/
python -m scripts.import_data
python -m scripts.import_data --input-dir ./backup --clear
```

## OpenSearch Dashboards

For ad-hoc index exploration during development:

```bash
docker-compose --profile debug up opensearch-dashboards
# → http://localhost:5601
```

## Code style

```bash
# Format
black src/ scripts/
isort src/ scripts/

# Type check
mypy src/

# All in one
black src/ scripts/ && isort src/ scripts/ && mypy src/
```

Configuration in `pyproject.toml`:
- Line length: 100
- Black profile for isort
- Python 3.11 target

## Testing

```bash
pytest
pytest -x          # Stop after first failure
pytest -v          # Verbose output
```

Test files live alongside source code or in a `tests/` directory. Use `pytest-asyncio` for async tests and `pytest-mock` for mocking.

## Project layout reference

```
src/
├── config.py                   # DorothyConfig, all settings classes
├── models/
│   ├── __init__.py             # Re-exports Article, Source, enums
│   └── article.py              # Article, Source, BiasRating, Column, Region, Perspective
├── storage/
│   ├── __init__.py
│   └── opensearch.py           # OpenSearchClient
├── fetcher/
│   ├── __init__.py
│   └── rss.py                  # RSSFetcher, fetch_all_sources()
├── embeddings/
│   ├── __init__.py
│   ├── client.py               # EmbeddingClient
│   └── generator.py            # generate_embeddings()
├── clustering/
│   ├── __init__.py
│   └── story_grouper.py        # StoryGrouper, Story
├── synthesis/
│   ├── __init__.py
│   ├── llm_client.py           # LLMClient, model lifecycle
│   └── summarizer.py           # StorySummarizer, SynthesizedStory
├── podcast/
│   ├── __init__.py
│   ├── generator.py            # PodcastGenerator
│   ├── script_writer.py        # ScriptWriter
│   ├── tts_client.py           # TTSClient (Chatterbox + HF fallback)
│   ├── audio_assembler.py      # AudioAssembler (FFmpeg/pydub)
│   └── feed.py                 # generate_feed() (RSS 2.0)
└── web/
    ├── __init__.py
    ├── app.py                  # FastAPI app factory
    ├── routes.py               # Route handlers
    ├── static/                 # CSS, JS, SVG assets
    └── templates/              # Jinja2 HTML templates
        ├── base.html
        ├── front_page.html
        ├── column.html
        ├── story.html
        ├── about.html
        └── podcast.html

scripts/
├── run_pipeline.py             # Main orchestrator
├── run_fetch.py                # Fetch only
├── run_embeddings.py           # Embeddings only
├── run_synthesis.py            # Synthesis only
├── run_podcast.py              # Podcast generation
├── run_server.py               # FastAPI dev server
├── render_static.py            # Static site generator
├── deploy_s3.py                # S3 deployment
├── visualize_clusters.py       # Cluster visualization
├── backfill_similarity.py      # Similarity edge backfill
├── dedup_syntheses.py          # Synthesis deduplication
├── export_data.py              # Export to JSONL
└── import_data.py              # Import from JSONL

config/
├── sources.yaml                # Source registry
└── voices/                     # TTS voice reference WAVs
    ├── anchor_a.wav
    └── anchor_b.wav
```

## Common development workflows

### Adding a new news source

1. Open `config/sources.yaml`
2. Add a new entry with `name`, `slug`, `rss_url`, `fetch_method: rss`, `column`, `bias`
3. For sports sources, add `region`
4. For tech sources, add `perspective`
5. Run `python -m scripts.run_fetch` to test
6. Check for articles: `curl http://localhost:9200/dorothy-articles-$(date +%Y-%m)/_count`

### Inspecting a synthesis

```bash
# Start the web server
python -m scripts.run_server --reload

# Browse to http://localhost:8000 and navigate to any story
# Or query the API:
curl http://localhost:8000/api/stories?column=politics&limit=5 | python -m json.tool
```

### Debugging clustering

```bash
# Visualize clusters
python -m scripts.visualize_clusters --column politics

# Check article counts in OpenSearch
curl "http://localhost:9200/dorothy-articles-$(date +%Y-%m)/_count?q=column:politics"
```

### Resetting OpenSearch

```bash
# Delete all Dorothy indices (keeps OpenSearch running)
curl -X DELETE "http://localhost:9200/dorothy-*"

# Or restart the container with fresh data
docker-compose down -v  # WARNING: deletes persistent volume
docker-compose up opensearch -d
```
