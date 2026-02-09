# Dorothy - Newspaper of Averages
# Multi-stage build for smaller image

FROM python:3.13-slim AS base

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --target=/app/deps .

# Install boto3 for S3 deployment
RUN pip install --target=/app/deps boto3


FROM base AS runtime

# Copy installed dependencies
COPY --from=builder /app/deps /app/deps
ENV PYTHONPATH=/app/deps

# Copy application code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Create output directory for static site
RUN mkdir -p /app/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/columns || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "scripts.run_pipeline", "--once"]
