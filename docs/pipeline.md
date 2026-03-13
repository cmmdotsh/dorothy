# Dorothy — Pipeline

The pipeline is the core of Dorothy. It runs on a schedule (default: every 60 minutes) and transforms raw RSS feeds into synthesized news stories stored in OpenSearch.

## Stages

```
Step 1: Fetch        RSS feeds → Article objects → OpenSearch (dorothy-articles-YYYY-MM)
Step 2: Embed        Articles without vectors → EmbeddingClient → update embeddings in OpenSearch
Step 3: Cluster      Embeddings per column → HDBSCAN → Story clusters
Step 4: Synthesize   Story clusters → LLM → SynthesizedStory → OpenSearch (dorothy-synthesis)
Step 5: Render       (optional, --publish) OpenSearch → static HTML → output/
Step 6: Deploy       (optional, --publish) output/ → S3 + CloudFront invalidation
```

## Running the pipeline

### Daemon mode (default)

Runs immediately on startup, then on a schedule:

```bash
python -m scripts.run_pipeline
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--interval N` / `-i N` | `60` | Minutes between pipeline cycles |
| `--stories N` / `-s N` | *unlimited* | Max stories to synthesize per column |
| `--publish` | off | Also render static site and deploy to S3 after each cycle |

The scheduler aligns to clock hours: with `--interval 60` it runs at `:00` past each hour, with `--interval 120` it runs every 2 hours at `:00`.

```bash
# Run every 30 minutes, render and deploy after each cycle
python -m scripts.run_pipeline --interval 30 --publish

# Limit synthesis to 10 stories per column (useful during development)
python -m scripts.run_pipeline --stories 10
```

### One-shot mode

Run the full pipeline once and exit:

```bash
python -m scripts.run_pipeline --once
python -m scripts.run_pipeline --once --publish   # with render + deploy
```

### Individual steps

Each step can be run independently for debugging or manual backfills:

```bash
# Fetch only
python -m scripts.run_fetch

# Generate embeddings for articles that don't have them
python -m scripts.run_embeddings

# Synthesize a specific column
python -m scripts.run_synthesis --column politics
python -m scripts.run_synthesis --column tech --stories 5
```

## Step 1: Fetch

**Module:** `src/fetcher/rss.py`
**Entry point:** `scripts/run_fetch.py`

1. Load all active RSS sources from `config/sources.yaml`
2. For each source, fetch and parse the RSS/Atom feed using `feedparser` over `httpx`
3. For each entry, extract:
   - Headline, summary (HTML stripped), canonical URL, publication date
   - Hero image: checks `media:content` (full-res) → `media:thumbnail` → enclosures. Minimum 400px width enforced.
4. Deduplicate in-memory by URL (seen-URL set per cycle)
5. Deduplicate against OpenSearch: `article_exists()` query by URL term
6. Batch-index new articles in groups of `FETCHER_BATCH_SIZE` (default 50)

**Bozo handling:** Malformed feed XML is logged at WARNING level but does not halt processing of other sources.

## Step 2: Embeddings

**Module:** `src/embeddings/`
**Entry point:** `scripts/run_embeddings.py`

1. Health-check the embedding service; skip if unavailable
2. Query OpenSearch for articles missing an `embedding` field
3. For each batch of articles (default batch: 32):
   - Build input text: `"{headline}\n\n{summary}"` (summary may be empty)
   - POST to `/v1/embeddings` (OpenAI-compatible API)
   - Bulk-update the `embedding` field in OpenSearch
4. Log per-batch statistics: success count, error count

The embedding model (`text-embedding-mxbai-embed-large-v1`) produces 1024-dimensional vectors. The OpenSearch k-NN index is pre-configured with `hnsw` / `cosine` for this dimension.

## Step 3: Cluster (within Synthesis)

**Module:** `src/clustering/story_grouper.py`

The `StoryGrouper.get_stories_for_column(column, size=2000)` method:

1. Fetches up to 2000 articles with embeddings for the column via k-NN scroll
2. Builds a pairwise cosine distance matrix (`1 - cosine_similarity`)
3. Runs HDBSCAN:
   - `min_cluster_size=3`
   - `min_samples=2`
   - `cluster_selection_method="eom"` (excess of mass)
4. Merges nearby cluster centroids iteratively (threshold: 0.15 cosine distance):
   - Compute centroid of each cluster as mean of member embeddings
   - Find the closest pair of centroids below threshold
   - Merge them; repeat until no pairs remain within threshold
5. Returns a list of `Story` objects

**`Story` properties:**

| Property | Description |
|----------|-------------|
| `id` | Deterministic SHA256 of sorted article URLs |
| `articles` | List of article dicts |
| `source_count` | Count of unique source slugs |
| `bias_spread` | Dict: bias tier → article count |
| `region_spread` | Dict: region → article count (sports) |
| `perspective_spread` | Dict: perspective → article count (tech) |
| `coverage_spread` | The relevant spread dict for this column |

Only stories with `source_count >= 2` are passed to synthesis. Single-source clusters (including all HDBSCAN noise articles labelled `-1`) are silently discarded.

## Step 4: Synthesize

**Module:** `src/synthesis/`
**Entry point:** `scripts/run_synthesis.py`

For each column, for each multi-source story cluster:

### Deduplication check

1. **Intra-batch dedup:** Compute Jaccard index of article URLs against every story already synthesized in this pipeline cycle. Skip if Jaccard > 0.30.

2. **Cross-batch dedup:** Call `OpenSearchClient.find_overlapping_synthesis()` to find any existing synthesis overlapping the cluster's article URLs.
   - If Jaccard > 0.15 and `new_urls < 3` → skip (story unchanged)
   - If Jaccard > 0.15 and `new_urls >= 3` → mark existing as historical, re-synthesize

### LLM synthesis (two-pass)

**`StorySummarizer.synthesize(story)`:**

**Pass 1 — Neutral article:**
- System prompt: "You are a wire service journalist writing for a neutral, factual news wire."
- Input: Article headlines and summaries, grouped by bias tier / region / perspective
- Output: JSON `{"headline": "...", "article": "..."}`
- Token budget awareness: if full text exceeds budget, samples one representative article per coverage bucket (article closest to cluster centroid)

**Pass 2 — Coverage analysis:**
- System prompt: "You are a media analyst examining how different news outlets cover the same story."
- Input: The neutral article from Pass 1 + the original articles with source attribution
- Output: Markdown analysis of what different outlets emphasized, omitted, or framed differently

Both passes use `skip_thinking=true` (for Qwen thinking models) to avoid wasted tokens.

### Storage

Synthesized stories are stored to the `dorothy-synthesis` index with:
- `is_current: true` for new stories
- `is_current: false` for superseded versions (with `superseded_by` pointing to the new story_id)

### `run_synthesis.py` options

```bash
python -m scripts.run_synthesis --column politics
python -m scripts.run_synthesis --column tech --stories 5
python -m scripts.run_synthesis --column sports
```

| Flag | Default | Description |
|------|---------|-------------|
| `--column` | all columns | Column to synthesize |
| `--stories N` / `-s N` | *unlimited* | Max stories per column |

## Step 5: Render (optional)

See [web.md](web.md) for the static site generator.

```bash
python -m scripts.render_static --clean
```

## Step 6: Deploy (optional)

See [deployment.md](deployment.md) for S3 deployment.

```bash
python -m scripts.deploy_s3 --invalidate
```

## Docker

In production, the full pipeline including render and deploy runs as a Docker daemon:

```bash
docker-compose up pipeline
```

This runs:
```
python -m scripts.run_pipeline --interval 60 --publish --stories 20
```

For a manual one-shot run in Docker:
```bash
docker-compose --profile manual run pipeline-once
```

## Health checks

The pipeline performs health checks before each cycle:

- **OpenSearch:** HTTP GET `/_cluster/health`, checks status `green` or `yellow`
- **LLM:** Chat completions request with minimal input, checks for valid JSON response
- **Embedding service:** Embed a test string, verify response has expected dimensions

If OpenSearch or the LLM is unavailable at startup (daemon mode), the process exits with code 1. If they go down mid-cycle, errors are logged and the affected steps are skipped (the cycle doesn't crash).

## Logging

The pipeline prints a Rich console summary at the start and end of each cycle:

```
╭─────────────────────────────────────────╮
│ Pipeline Cycle Starting                  │
│ 2026-03-13 14:00:00 UTC                  │
╰─────────────────────────────────────────╯

  Step 1: Fetching articles...      → 127 new articles
  Step 2: Generating embeddings...  → Generated embeddings for 127 articles
  Step 3: Synthesizing stories...
    politics...  12 stories
    tech...       8 stories
    money...      5 stories
    sports...     6 stories
    lifestyle...  4 stories
  Step 4: Rendering static site...  → Static site rendered successfully
  Step 5: Deploying to S3...        → Uploaded 89 files

╭─────────────────────────────────────────────────────────────────────────╮
│ Cycle Complete                                                           │
│ Articles: 127 | Embedded: 127 | Stories: 35 | Duration: 312.4s          │
╰─────────────────────────────────────────────────────────────────────────╯
```
