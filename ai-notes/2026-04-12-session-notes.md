# Session Notes — 2026-04-12/13

## 1. mlx-lm Setup for Gemma 4 Reviewer

### Problem
Gemma 4 31B running through Ollama on macstudio.local (192.168.0.149, M1 Ultra 128GB) was extremely slow for the review pass.

### What We Did
- Installed `mlx-lm` in a dedicated venv at `~/.mlx-gemma` (Python 3.13 via Homebrew)
- Downloaded `mlx-community/gemma-4-31b-it-8bit` (~31.5 GB)
- Created a **launchd service** (`com.dorothy.mlx-gemma`) that auto-starts on boot, serves on port 8081
- Key flags: `--pipeline` (for M1 Ultra dual-die), `--chat-template-args '{"enable_thinking": false}'`, `--prompt-cache-bytes 8589934592`
- Port 8081 chosen to avoid conflicts (Ollama stays on 11434)

### Files Changed
- `scripts/setup_mlx_gemma.sh` — **new** — setup script to install and configure mlx-lm on macstudio.local
- `src/config.py` — `ReviewerSettings` defaults changed to `http://192.168.0.149:8081` and `mlx-community/gemma-4-31b-it-8bit`
- `src/synthesis/ollama_client.py` — `health_check()` now tries `/v1/models` first (works with both mlx-lm and Ollama), falls back to `/api/tags`

### Performance
- With swap (45GB used): 5.1 tok/s, 85.9s per review
- Without swap (apps closed): **14.1 tok/s, 27.4s per review**
- Theoretical max on M1 Ultra: ~25 tok/s (800 GB/s bandwidth / 31GB model)
- Both Qwen (~46GB) and Gemma (~26GB) loaded simultaneously = ~72GB, fits in 128GB

### Key Lesson
The Mac Studio was swapping 45GB to disk because Firefox, Discord, Steam, and both models were loaded simultaneously. Closing desktop apps fixed it.

---

## 2. Proxmox LXC Container for Pipeline

### Problem
Pipeline was running ad-hoc from the dev machine, needed a permanent home.

### What We Did
- Created LXC container (CT 110 "dorothy") on Proxmox at dev.local (root@dev.local)
- Ubuntu 24.04, 2 cores, 1GB RAM, 12GB disk, on vmbr0 (same 192.168.0.x subnet as Mac Studio)
- Installed Docker inside the LXC (nesting=1 enabled)
- Created `docker-compose.override.yml` that:
  - Disables local OpenSearch (uses Mac Studio's at 192.168.0.149:9200)
  - Points all services at Mac Studio (Ollama :11434, mlx-lm :8081)
  - Uses `network_mode: host` and `!reset` to clear base compose deps
  - Mounts AWS creds from `/root/.aws`
- AWS deploy creds (`S3_BUCKET=dorothy-cmm-sh`, `CLOUDFRONT_ID=E28XL76SLBWXL`) in `/opt/dorothy/.env`

### Container Layout
```
dev.local (Proxmox)
  └── CT 110 "dorothy" (Ubuntu 24.04 LXC, IP 192.168.0.152)
       └── Docker
            └── dorothy-pipeline container
                 ├── → 192.168.0.149:9200  (OpenSearch on Mac Studio)
                 ├── → 192.168.0.149:11434 (Ollama/Qwen synthesis)
                 └── → 192.168.0.149:8081  (mlx-lm/Gemma reviewer)
```

### Management Commands
```bash
# One-shot run
ssh root@dev.local 'pct exec 110 -- bash -c "cd /opt/dorothy && docker compose run --rm pipeline-once"'

# Persistent daemon (every 60 min)
ssh root@dev.local 'pct exec 110 -- bash -c "cd /opt/dorothy && docker compose up -d pipeline"'

# Check logs
ssh root@dev.local 'pct exec 110 -- docker logs <container-name>'
```

### Files Changed
- `docker-compose.override.yml` — **exists only on CT 110**, not in repo. Disables OpenSearch, points at Mac Studio.

---

## 3. Embedding Context Length Bug

### Problem
Ollama's `mxbai-embed-large` has a 512 token context window. Embedding batches of 8-32 articles (headline + summary) exceeded this limit, causing infinite 400 error loops.

### Root Cause
- Ollama's `/v1/embeddings` counts ALL inputs in the batch against the 512-token context, not each individually
- The generator's `while True` loop retried the same failing batch forever because failed articles still didn't have embeddings
- LMStudio (previous backend) silently truncated; Ollama strictly rejects

### What We Did
- Reduced `_prepare_text()` max_chars from 1500 → **500** (conservative fit for 512-token limit)
- Reduced default `batch_size` from 32 → **8** in `EmbeddingSettings`
- Added **individual fallback**: when a batch fails, retry each article one-by-one; store a zero-vector for articles that still fail (prevents infinite retry)
- Added response body logging to embedding API errors

### Files Changed
- `src/embeddings/generator.py` — `_prepare_text()` truncation, individual fallback on batch failure
- `src/embeddings/client.py` — response body in error logs
- `src/config.py` — `batch_size` default 32 → 8

### Status
**Still needs commit and deploy** — the individual fallback and 500-char truncation are staged but not yet pushed.

---

## 4. Full Article Text Extraction (trafilatura)

### Problem
Dorothy only used RSS headline + summary blurbs (1-2 sentences) for everything — embedding, clustering, and synthesis. The LLM was writing articles from headlines and vibes, not actual reporting.

### Design Decision
**Hybrid approach**: keep headline+summary embeddings for clustering (works well, 512-token model limit), but use full body text for synthesis. No chunking, no new indices, no embedding model changes.

### What We Did
- Added `body` (Optional[str]) and `body_extracted_at` (Optional[datetime]) to Article model
- Added both fields to OpenSearch ARTICLE_MAPPING
- Created `src/fetcher/extractor.py` — new `ArticleExtractor` class using **trafilatura**
  - `extract(url)` → fetches URL, returns Markdown via `trafilatura.extract(output_format='markdown', favor_precision=True)`
  - `extract_batch(articles, os_client)` → iterates articles with politeness delay, updates OpenSearch
  - Failed extractions get `body_extracted_at` set (prevents retry) but `body` stays null
- Added `ExtractorSettings` to config (enabled, timeout, delay=1.0s, batch_size=50)
- Added `run_extraction()` as **Step 2** in the pipeline (between fetch and embed)
- Modified `_format_article()` in synthesizer to prefer `body[:2000]` over `summary[:500]`
- Added `trafilatura>=2.0.0` to pyproject.toml (also needs `lxml_html_clean`)

### Smoke Test Results
| Source | RSS Summary | Extracted Body |
|--------|------------|----------------|
| IEEE Spectrum | 2000 chars | **7533 chars** |
| Daily Wire | 328 chars | **26627 chars** |
| Wired | 111 chars | **2001 chars** |
| Politico | 135 chars | fetch failed (Google News redirect) |

### Pipeline Step Order (new)
1. Fetch RSS articles
2. **Extract article bodies** (new)
3. Generate embeddings
4. Synthesize stories
5. Render static site + deploy

### Files Changed
- `src/models/article.py` — added `body`, `body_extracted_at` fields
- `src/storage/opensearch.py` — mapping + `get_articles_without_body()`, `update_article_body()`, `mark_body_extraction_failed()`
- `src/fetcher/extractor.py` — **new** — ArticleExtractor class
- `src/config.py` — added `ExtractorSettings`
- `scripts/run_pipeline.py` — added `run_extraction()` as Step 2, renumbered steps
- `src/synthesis/summarizer.py` — `_format_article()` prefers body over summary
- `pyproject.toml` — added `trafilatura>=2.0.0`

---

## Commits Made This Session

| Hash | Message |
|------|---------|
| `413cf59` | mlx-lm support for gemma4 reviewer, universal health check |
| `736ff34` | log response body on embedding API errors |
| `ee6cf86` | reduce embedding batch size to 8 to avoid context length overflow |
| `4d1fc52` | full article text extraction via trafilatura, feed body text to synthesis |

## Uncommitted Changes
- `src/embeddings/generator.py` — 500-char truncation + individual fallback on batch failure (needs commit)

## Known Issues
- Embedding batch still failing on the LXC pipeline — the uncommitted fix should resolve this
- Google News redirect URLs (many sources) will fail extraction — falls back to RSS summary gracefully
- Paywalled sites return sparse or no text — same fallback
- AWS STS session tokens on Mac Studio expire; LXC uses IAM keys from `/opt/dorothy/.env`
