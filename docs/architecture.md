# Dorothy — Architecture

## Data flow

```
RSS Feeds (40+ sources)
        │
        ▼
  ┌─────────────┐
  │  RSSFetcher  │  feedparser + httpx
  └──────┬──────┘
         │ Article objects (headline, summary, URL, pub_date, bias, column)
         ▼
  ┌──────────────────┐
  │ OpenSearch        │  Monthly index: dorothy-articles-YYYY-MM
  │ (article store)   │  Deduplication by URL
  └──────┬───────────┘
         │ Articles without embeddings
         ▼
  ┌──────────────────┐
  │ EmbeddingClient  │  LMStudio (text-embedding-mxbai-embed-large-v1, 1024-dim)
  └──────┬───────────┘
         │ 1024-dim float vectors written back to OpenSearch
         ▼
  ┌──────────────────┐
  │ StoryGrouper     │  HDBSCAN density clustering on cosine distance matrix
  │ (per column)     │  Centroid merging (threshold 0.15 cosine distance)
  └──────┬───────────┘
         │ Story clusters (list of articles)
         ▼
  ┌──────────────────┐
  │ StorySummarizer  │  Two-pass LLM generation (LMStudio, Qwen)
  │                  │  Pass 1: neutral wire-service article
  │                  │  Pass 2: coverage analysis
  └──────┬───────────┘
         │ SynthesizedStory objects
         ▼
  ┌──────────────────────┐
  │ OpenSearch            │  Index: dorothy-synthesis
  │ (synthesis store)     │  Hotness ranking, deduplication, version tracking
  └──────┬───────────────┘
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
┌──────────┐      ┌──────────────────┐
│ FastAPI  │      │ StaticSiteGen    │  Jinja2 → HTML files → output/
│ (dev)    │      │                  │
└──────────┘      └────────┬─────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ S3Deployer   │  boto3 → S3 + CloudFront invalidation
                   └──────────────┘

                   (Separately)
                   ┌──────────────────┐
                   │ PodcastGenerator │  ScriptWriter (LLM) → TTSClient → AudioAssembler
                   └──────┬───────────┘
                          │ MP3 + feed.xml
                          ▼
                   ┌──────────────┐
                   │ S3Deployer   │  podcast/ subdirectory
                   └──────────────┘
```

## OpenSearch indices

Dorothy uses three indices:

### `dorothy-articles-YYYY-MM`

Monthly rolling article index. A new index is created automatically at the start of each calendar month.

| Field | Type | Description |
|-------|------|-------------|
| `id` | keyword | UUID |
| `source_name` | text | Human-readable outlet name |
| `source_slug` | keyword | URL-safe source identifier |
| `source_bias` | keyword | `left` / `lean-left` / `center` / `lean-right` / `right` |
| `column` | keyword | `politics` / `tech` / `money` / `sports` / `lifestyle` |
| `headline` | text | Article headline |
| `summary` | text | Article lede/summary (HTML stripped) |
| `url` | keyword | Canonical article URL (used for deduplication) |
| `pub_date` | date | When the article was published |
| `fetched_at` | date | When Dorothy fetched it |
| `embedding` | knn_vector (dim=1024) | Semantic embedding, HNSW, cosine space metric |
| `image_url` | keyword | Thumbnail/hero image URL |
| `source_region` | keyword | Geographic region (sports sources) |
| `source_perspective` | keyword | Editorial perspective (tech sources) |

The k-NN mapping uses `hnsw` engine with `cosine` space metric. This enables efficient approximate nearest-neighbor search across the full article corpus.

### `dorothy-synthesis`

Synthesized stories with LLM-generated content and metadata.

| Field | Type | Description |
|-------|------|-------------|
| `story_id` | keyword | SHA256 of sorted article URLs |
| `column` | keyword | Column this story belongs to |
| `headline` | text | Original most-prominent headline |
| `generated_headline` | text | LLM-generated neutral headline |
| `article_text` | text | LLM-generated neutral article body |
| `analysis` | text | LLM-generated coverage analysis |
| `sources_used` | keyword[] | List of source slugs |
| `bias_coverage` | object | Count per bias tier |
| `article_count` | integer | Total source articles |
| `articles` | nested | Article refs (url, source_name, headline) |
| `hero_image_url` | keyword | Selected hero image URL |
| `hero_image_source` | keyword | Source of hero image |
| `similarity_edges` | nested | Cosine similarity pairs between articles |
| `hotness_score` | float | Ranking score |
| `median_pub_date` | date | Median publication date of source articles |
| `first_pub_date` | date | Earliest publication date |
| `last_pub_date` | date | Most recent publication date |
| `edition` | integer | Pipeline run number |
| `is_current` | boolean | Whether this is the latest version |
| `superseded_by` | keyword | story_id of newer version (if any) |
| `synthesized_at` | date | When this synthesis was created |

### `dorothy-metadata`

Singleton metadata document used for edition tracking.

| Field | Description |
|-------|-------------|
| `edition` | Current pipeline run counter |

## Key algorithms

### Story clustering (HDBSCAN)

Each column is clustered independently:

1. Fetch all articles with embeddings for the column (up to 2000)
2. Build a pairwise cosine distance matrix
3. Run HDBSCAN with `min_cluster_size=3`, `min_samples=2`, `cluster_selection_method="eom"`
4. Merge cluster centroids that are within 0.15 cosine distance (iterative greedy merge)
5. Articles labeled `-1` (noise) become single-article "stories" — they will not be synthesized because they don't meet the `source_count >= 2` threshold

This approach is preferable to k-NN chaining because HDBSCAN won't link loosely related articles through intermediate articles; clusters are density-based.

### Story deduplication (Jaccard index)

Dorothy deduplicates stories at two levels:

**Intra-batch dedup** (within one pipeline cycle):
- Before synthesizing each story, compute Jaccard index against every story already synthesized in this batch
- Skip if Jaccard > 0.30

**Cross-batch dedup** (against stored syntheses):
- Query OpenSearch for any existing synthesis whose article URLs overlap with the current cluster
- If Jaccard > 0.15 and there are fewer than 3 genuinely new articles → skip (story hasn't changed enough)
- If Jaccard > 0.15 and there are 3+ new articles → mark existing synthesis as historical, re-synthesize

**Pre-render dedup** (before static site generation):
- Final per-column pass removes stories with > 30% article URL overlap
- Ensures clean column pages without near-duplicate stories

### Hotness scoring

```
hotness = (article_count / hours_since_median_pub_date) * diversity_bonus
diversity_bonus = 1.0 + 0.1 * (unique_coverage_dimensions - 1)
```

- `unique_coverage_dimensions` counts distinct bias tiers (politics), geographic regions (sports), or editorial perspectives (tech) represented
- Score naturally decays as `hours_since_median_pub_date` grows
- Higher diversity of sources (more tiers/regions/perspectives) boosts the score

### Token-aware article sampling

The LLM has a fixed context window (default: 32768 tokens). When a story's full article text would exceed the budget:

1. Estimate token count: `len(text) / 3.5`
2. Group articles by coverage bucket (bias tier / region / perspective)
3. For each bucket, select the article closest to the cluster centroid (computed as mean of all embeddings in the cluster)
4. Include one representative article per bucket until the token budget is consumed

This ensures the LLM sees balanced coverage even when the full corpus is too large.

### Hero image selection

When choosing a hero image for a story:

1. Collect all article images across the cluster
2. Prefer images from center-bias sources (most neutral framing)
3. Fallback order: `center` → `lean-left` → `lean-right` → `left` → `right`
4. Filter out thumbnails detected by URL pattern (e.g., small Google News thumbnails)

## Data models

### `Article`

```python
class Article(BaseModel):
    id: UUID
    source_name: str        # "Associated Press"
    source_slug: str        # "ap"
    source_bias: BiasRating # center
    column: Column          # politics
    headline: str
    summary: Optional[str]
    url: HttpUrl
    pub_date: datetime
    fetched_at: datetime
    embedding: Optional[list[float]]   # 1024-dim vector
    source_region: Optional[str]       # sports: "us", "uk", etc.
    source_perspective: Optional[str]  # tech: "consumer", "enterprise", etc.
    image_url: Optional[str]
```

### `Source`

```python
class Source(BaseModel):
    name: str
    slug: str
    rss_url: Optional[HttpUrl]
    fetch_method: FetchMethod   # rss | scrape (scrape deferred)
    column: Column
    bias: BiasRating
    region: Optional[str]
    perspective: Optional[str]
    active: bool
```

### `SynthesizedStory`

```python
@dataclass
class SynthesizedStory:
    story_id: str
    column: str
    headline: str                  # original headline
    generated_headline: str        # LLM-written neutral headline
    article_text: str              # LLM-written neutral article
    analysis: str                  # LLM-written coverage analysis
    sources_used: list[str]        # source slugs
    bias_coverage: dict            # {"center": 3, "lean-left": 2, ...}
    article_count: int
    articles: list[dict]           # refs with url, source_name, headline
    hero_image_url: Optional[str]
    hero_image_source: Optional[str]
    similarity_edges: list[dict]   # [{source, target, weight}, ...]
    hotness_score: float
    median_pub_date: datetime
    first_pub_date: datetime
    last_pub_date: datetime
    edition: int
    is_current: bool
    superseded_by: Optional[str]
    synthesized_at: datetime
```

## Logging

Dorothy uses [structlog](https://www.structlog.org/) with ISO timestamps and color console output via Rich. Key log events:

| Event | Context |
|-------|---------|
| `fetch_job_started` / `fetch_job_completed` | Fetch cycle |
| `embeddings_generated` | Per-batch embedding stats |
| `articles_grouped` | Clustering output per column |
| `story_unchanged` | Dedup: existing synthesis is current |
| `story_evolved` | Dedup: existing marked historical, re-synthesizing |
| `story_duplicate_in_batch` | Intra-batch Jaccard dedup |
| `article_synthesized` | LLM synthesis complete |
| `synthesis_stored` | Stored to OpenSearch |
| `llm_generation_complete` | LLM call stats |
| `opensearch_connected` / `opensearch_error` | Database health |
