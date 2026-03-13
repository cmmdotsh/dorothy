# Dorothy — Configuration

Dorothy is configured entirely through environment variables. All settings have sane defaults for local development. The `src/config.py` module loads them at startup via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

## OpenSearch

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENSEARCH_HOST` | `localhost` | OpenSearch hostname |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `OPENSEARCH_USERNAME` | *(empty)* | Username for HTTP auth (leave empty to disable) |
| `OPENSEARCH_PASSWORD` | *(empty)* | Password for HTTP auth |
| `OPENSEARCH_USE_SSL` | `false` | Enable TLS/SSL |
| `OPENSEARCH_VERIFY_CERTS` | `false` | Verify SSL certificates |

## LLM (story synthesis)

Dorothy uses any OpenAI-compatible chat completions API — LMStudio is the default.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://192.168.0.149:1234` | Base URL of the LLM API |
| `LLM_MODEL` | `mlx-community/qwen3.5-35b-a3b` | Model identifier |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (lower = more factual) |
| `LLM_MAX_TOKENS` | `1500` | Maximum tokens per generation pass |
| `LLM_CONTEXT_LENGTH` | `32768` | Context window length in tokens |

The LLM client manages model lifecycle: it ensures the model is loaded with the correct context length before each generation, and sets `ttl: -1` to prevent LMStudio from auto-unloading it between requests.

## Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_BASE_URL` | `http://192.168.0.149:1234` | Base URL of the embedding API |
| `EMBEDDING_MODEL` | `text-embedding-mxbai-embed-large-v1` | Embedding model identifier |
| `EMBEDDING_BATCH_SIZE` | `32` | Articles per embedding request |
| `EMBEDDING_ENABLED` | `true` | Set to `false` to skip embedding generation |

The embedding model must produce 1024-dimensional vectors. The OpenSearch k-NN index is configured for this dimension at creation time and cannot be changed.

## Fetcher

| Variable | Default | Description |
|----------|---------|-------------|
| `FETCHER_TIMEOUT` | `30.0` | Per-request timeout in seconds |
| `FETCHER_USER_AGENT` | `Dorothy/0.1 (news aggregator)` | HTTP User-Agent header |
| `FETCHER_BATCH_SIZE` | `50` | Articles batched per OpenSearch bulk index call |

## Scheduler

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_FETCH_INTERVAL_MINUTES` | `60` | Interval between pipeline runs (daemon mode) |

This is the default; the `--interval` CLI flag on `run_pipeline.py` overrides it at runtime.

## Podcast

| Variable | Default | Description |
|----------|---------|-------------|
| `PODCAST_ENABLED` | `false` | Must be `true` to generate episodes |
| `PODCAST_VOICE_REF_A` | `config/voices/anchor_a.wav` | Path to voice reference WAV for anchor A |
| `PODCAST_VOICE_REF_B` | `config/voices/anchor_b.wav` | Path to voice reference WAV for anchor B |
| `PODCAST_TTS_DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda`, or `mps` |
| `PODCAST_TTS_WORKERS` | `1` | Parallel TTS threads |
| `PODCAST_STORY_COUNT` | `5` | Stories per episode |
| `PODCAST_TARGET_WPM` | `150` | Target words per minute for script writing |
| `PODCAST_ATEMPO` | `1.1` | FFmpeg atempo filter (1.1 = 10% speed-up) |
| `PODCAST_OUTPUT_FORMAT` | `mp3` | Audio output format |
| `PODCAST_BITRATE` | `128k` | MP3 bitrate |
| `PODCAST_HF_FALLBACK` | `false` | Use HuggingFace Inference API as TTS fallback |
| `PODCAST_HF_TOKEN` | *(empty)* | HuggingFace API token (required if fallback enabled) |

Voice reference files are WAV files used for voice cloning by the Chatterbox TTS model. Place them in `config/voices/`. If `PODCAST_VOICE_REF_B` does not exist, the pipeline falls back to using anchor A's voice for both roles.

## AWS (S3 deployment)

| Variable | Default | Description |
|----------|---------|-------------|
| `S3_BUCKET` | *(required)* | S3 bucket name for static site hosting |
| `AWS_REGION` | `us-east-1` | AWS region |
| `CLOUDFRONT_ID` | *(empty)* | CloudFront distribution ID; if set, cache is invalidated after each deploy |
| `AWS_ACCESS_KEY_ID` | *(from environment or `~/.aws`)* | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | *(from environment or `~/.aws`)* | AWS secret key |

If `S3_BUCKET` is not set, the deploy step is silently skipped. AWS credentials can be provided via environment variables, `~/.aws/credentials`, or an IAM instance role.

## Source registry (`config/sources.yaml`)

Sources are defined in `config/sources.yaml`, not via environment variables. Each entry:

```yaml
sources:
  - name: "Associated Press"
    slug: "ap"
    rss_url: "https://news.google.com/rss/search?q=source:AP+when:1d&hl=en-US"
    fetch_method: rss
    column: politics
    bias: center
    active: true

  - name: "ESPN"
    slug: "espn"
    rss_url: "https://www.espn.com/espn/rss/news"
    fetch_method: rss
    column: sports
    bias: center
    region: us
    active: true

  - name: "The Verge"
    slug: "theverge"
    rss_url: "https://www.theverge.com/rss/index.xml"
    fetch_method: rss
    column: tech
    bias: lean-left
    perspective: consumer
    active: true
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Display name |
| `slug` | yes | URL-safe identifier (used in UI and dedup) |
| `rss_url` | yes (for rss) | RSS/Atom feed URL |
| `fetch_method` | yes | `rss` (scrape is deferred) |
| `column` | yes | `politics` / `tech` / `money` / `sports` / `lifestyle` |
| `bias` | yes | `left` / `lean-left` / `center` / `lean-right` / `right` |
| `region` | no | Required for sports sources; see `Region` enum |
| `perspective` | no | Required for tech sources; see `Perspective` enum |
| `active` | no | Defaults to `true`. Set to `false` to disable without deleting |

**Region values:** `us`, `canada`, `mexico`, `uk`, `australia`, `india`, `japan`, `korea`, `international`

**Perspective values:** `consumer`, `enterprise`, `academic`, `culture`

## `.env` file

For local development, create a `.env` file in the project root. `python-dotenv` loads it automatically:

```dotenv
# LLM (point to your local LMStudio instance)
LLM_BASE_URL=http://localhost:1234
LLM_MODEL=mlx-community/qwen3.5-35b-a3b
LLM_CONTEXT_LENGTH=32768

# Embeddings (same LMStudio instance, different model loaded)
EMBEDDING_BASE_URL=http://localhost:1234
EMBEDDING_MODEL=text-embedding-mxbai-embed-large-v1

# OpenSearch (local docker-compose default)
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200

# S3 (optional; omit to skip deploy)
S3_BUCKET=my-dorothy-bucket
CLOUDFRONT_ID=E1234ABCD5EFGH
AWS_REGION=us-east-1
```
