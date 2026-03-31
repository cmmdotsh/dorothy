# Dorothy — Deployment

## Docker

Dorothy ships with two Dockerfiles and a `docker-compose.yml` that defines all services.

### Images

#### `Dockerfile` — Main pipeline image

- Base: `python:3.13-slim`
- Multi-stage build: `builder` (installs deps) → `runtime` (copies deps + source)
- Includes: `boto3` for S3 deployment
- Default CMD: `python -m scripts.run_pipeline --once`
- Health check: curls `/_cluster/health` on the configured OpenSearch host

#### `Dockerfile.podcast` — Podcast image

- Base: `python:3.12-slim` (required for Chatterbox TTS compatibility)
- System packages: `ffmpeg`, `build-essential`
- Installs TTS dependencies in a specific order to work around numpy version constraints
- Default CMD: `python -m scripts.run_podcast --daemon`

### docker-compose services

| Service | Profile | Description |
|---------|---------|-------------|
| `opensearch` | *(always)* | OpenSearch 2.11 database |
| `opensearch-dashboards` | `debug` | OpenSearch Dashboards UI on port 5601 |
| `pipeline` | *(always)* | Hourly pipeline daemon (fetch → synthesize → render → deploy) |
| `pipeline-once` | `manual` | One-shot pipeline run for testing |
| `render` | `manual` | Static site render only |
| `web` | `dev` | FastAPI dev server on port 8000 |
| `podcast` | *(always)* | Podcast generation daemon |

### Starting the full stack

```bash
# Start core services (opensearch + pipeline + podcast)
docker-compose up -d

# View logs
docker-compose logs -f pipeline

# Include dev web server
docker-compose --profile dev up -d

# Include OpenSearch Dashboards for debugging
docker-compose --profile debug up -d
```

### One-shot runs (testing)

```bash
# Run the full pipeline once
docker-compose --profile manual run pipeline-once

# Render static site only
docker-compose --profile manual run render
```

### Environment variables for Docker

Create a `.env` file in the project root — `docker-compose` picks it up automatically:

```dotenv
# LLM (accessible from inside the container via host.docker.internal)
LLM_BASE_URL=http://host.docker.internal:1234
LLM_MODEL=mlx-community/qwen3.5-35b-a3b
LLM_CONTEXT_LENGTH=32768

# Embeddings
EMBEDDING_BASE_URL=http://host.docker.internal:1234
EMBEDDING_MODEL=text-embedding-mxbai-embed-large-v1

# AWS
S3_BUCKET=my-dorothy-bucket
AWS_REGION=us-east-1
CLOUDFRONT_ID=E1234ABCD5EFGH
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

`host.docker.internal` resolves to the Docker host machine. If your LMStudio runs on a separate machine, use its IP address instead.

### Volumes

| Volume | Purpose |
|--------|---------|
| `opensearch-data` | Persists OpenSearch indices across restarts |
| `./output:/app/output` | Local mount for rendered static site (useful for debugging) |
| `~/.aws:/root/.aws:ro` | AWS credentials file (alternative to env vars) |
| `./config/voices:/app/config/voices:ro` | Voice reference WAVs for podcast |

---

## S3 deployment

### Setup

1. Create an S3 bucket configured for static website hosting
2. Create a CloudFront distribution pointing to the S3 bucket (optional but recommended)
3. Set the required environment variables (`S3_BUCKET`, `AWS_REGION`, etc.)

### Running the deployer

```bash
# Deploy the rendered static site
python -m scripts.deploy_s3

# With explicit arguments
python -m scripts.deploy_s3 --bucket my-dorothy-bucket --region us-east-1

# Preview what would be uploaded (dry run)
python -m scripts.deploy_s3 --dry-run

# Deploy and invalidate CloudFront cache
python -m scripts.deploy_s3 --invalidate
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bucket` / `-b` | `$S3_BUCKET` | S3 bucket name |
| `--source` / `-s` | `./output` | Local directory to sync |
| `--region` / `-r` | `$AWS_REGION` or `us-east-1` | AWS region |
| `--cloudfront-id` | `$CLOUDFRONT_ID` | CloudFront distribution ID |
| `--invalidate` / `-i` | off | Invalidate `/*` after upload |
| `--dry-run` / `-n` | off | Preview without uploading |

### Cache-Control headers

The deployer sets appropriate `Cache-Control` headers per file type:

| File type | Cache-Control |
|-----------|--------------|
| `.html` | `public, max-age=300` (5 minutes) |
| `.css`, `.js` | `public, max-age=31536000` (1 year — files are version-stamped) |
| `.png`, `.jpg` | `public, max-age=86400` (1 day) |
| `.mp3` | `public, max-age=3600` (1 hour) |
| `.xml` (RSS) | `public, max-age=300` (5 minutes) |
| `sw.js` | `public, max-age=0, must-revalidate` |
| `manifest.json` | `public, max-age=3600` |

### Podcast deployment

The podcast generator deploys its own files separately to avoid re-uploading the entire static site on every episode:

```
output/podcast/
    feed.xml          → always uploaded
    latest.mp3        → always uploaded
    {episode}.mp3     → uploaded once when created
    {episode}.manifest.json  → uploaded once when created
```

Previous episode MP3s already on S3 are not re-uploaded. The rest of the static site (HTML/CSS/JS) is excluded from podcast deploys.

---

## Terraform (AWS infrastructure)

The `terraform/` directory contains AWS infrastructure definitions:

```bash
cd terraform/

# Initialize
terraform init

# Preview changes
terraform plan

# Apply
terraform apply
```

This manages the S3 bucket and CloudFront distribution. Refer to the Terraform state and variable definitions in `terraform/` for the full resource configuration.

---

## Production architecture

```
LMStudio (local)          OpenSearch (Docker)
    │                           │
    │  LLM + Embedding APIs     │ dorothy-articles-*, dorothy-synthesis
    │                           │
    └──────── pipeline ─────────┘
                │
                │ render static HTML
                ▼
            output/
                │
                │ boto3 upload
                ▼
         S3 Bucket ──────────── CloudFront CDN ──── Users
         (origin)                (dorothy.cmm.sh)

         output/podcast/
                │
                │ boto3 upload (podcast-specific files only)
                ▼
         S3 Bucket / podcast/
```

The pipeline and podcast containers need outbound network access to:
- LMStudio API (typically on the host or a local network)
- OpenSearch (on the `dorothy-net` Docker network)
- AWS S3 endpoints

No inbound ports need to be exposed (the pipeline is a daemon, not a server).
