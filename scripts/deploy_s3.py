#!/usr/bin/env python3
"""
S3 Deployment for Dorothy Static Site

Syncs the rendered static site to an S3 bucket.

Usage:
    python -m scripts.deploy_s3                           # Sync to configured bucket
    python -m scripts.deploy_s3 --bucket my-bucket        # Custom bucket
    python -m scripts.deploy_s3 --dry-run                 # Preview changes
    python -m scripts.deploy_s3 --invalidate              # Invalidate CloudFront cache

Requires:
    - AWS credentials configured (via env vars, ~/.aws/credentials, or IAM role)
    - boto3 installed (add to dependencies)

Environment Variables:
    S3_BUCKET       - Target S3 bucket name
    CLOUDFRONT_ID   - CloudFront distribution ID (optional, for cache invalidation)
    AWS_REGION      - AWS region (default: us-east-1)
"""

import argparse
import mimetypes
import os
from pathlib import Path
from datetime import datetime, timezone

import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()


# Content types for common extensions
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}

# Cache control settings
CACHE_CONTROL = {
    ".html": "public, max-age=300",  # 5 minutes for HTML
    ".css": "public, max-age=31536000",  # 1 year for CSS (versioned)
    ".js": "public, max-age=31536000",  # 1 year for JS (versioned)
    ".png": "public, max-age=86400",  # 1 day for images
    ".jpg": "public, max-age=86400",
    ".jpeg": "public, max-age=86400",
}


class S3Deployer:
    """Deploys static site to S3."""

    def __init__(
        self,
        bucket: str,
        source_dir: Path,
        region: str = "us-east-1",
        cloudfront_id: str | None = None,
        dry_run: bool = False,
    ):
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 deployment. Install with: pip install boto3")

        self.bucket = bucket
        self.source_dir = source_dir
        self.region = region
        self.cloudfront_id = cloudfront_id
        self.dry_run = dry_run

        self.s3 = boto3.client("s3", region_name=region)
        if cloudfront_id:
            self.cloudfront = boto3.client("cloudfront", region_name=region)

    def get_content_type(self, path: Path) -> str:
        """Get content type for a file."""
        ext = path.suffix.lower()
        if ext in CONTENT_TYPES:
            return CONTENT_TYPES[ext]
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "application/octet-stream"

    def get_cache_control(self, path: Path) -> str:
        """Get cache control header for a file."""
        name = path.name
        # sw.js must always be revalidated so browsers pick up new versions
        if name == "sw.js":
            return "public, max-age=0, must-revalidate"
        # manifest.json — short cache
        if name == "manifest.json":
            return "public, max-age=3600"
        ext = path.suffix.lower()
        return CACHE_CONTROL.get(ext, "public, max-age=3600")

    def get_s3_key(self, local_path: Path) -> str:
        """Convert local path to S3 key."""
        relative = local_path.relative_to(self.source_dir)
        return str(relative).replace("\\", "/")

    def upload_file(self, local_path: Path) -> bool:
        """Upload a single file to S3."""
        s3_key = self.get_s3_key(local_path)
        content_type = self.get_content_type(local_path)
        cache_control = self.get_cache_control(local_path)

        if self.dry_run:
            console.print(f"[dim]  Would upload: {s3_key} ({content_type})[/dim]")
            return True

        try:
            self.s3.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": content_type,
                    "CacheControl": cache_control,
                },
            )
            return True
        except ClientError as e:
            logger.error("upload_failed", path=s3_key, error=str(e))
            return False

    def sync(self) -> dict:
        """Sync all files to S3."""
        start = datetime.now(timezone.utc)

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        # Collect all files
        files = list(self.source_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        if not files:
            console.print("[yellow]No files to upload[/yellow]")
            return {"uploaded": 0, "failed": 0}

        prefix = "[DRY RUN] " if self.dry_run else ""
        console.print(f"[bold blue]{prefix}Uploading {len(files)} files to s3://{self.bucket}/[/bold blue]")

        uploaded = 0
        failed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading...", total=len(files))

            for file_path in files:
                s3_key = self.get_s3_key(file_path)
                progress.update(task, description=f"Uploading {s3_key}")

                if self.upload_file(file_path):
                    uploaded += 1
                else:
                    failed += 1

                progress.advance(task)

        duration = (datetime.now(timezone.utc) - start).total_seconds()

        console.print(f"\n[bold green]{prefix}Uploaded {uploaded} files in {duration:.1f}s[/bold green]")
        if failed:
            console.print(f"[red]Failed: {failed} files[/red]")

        return {
            "uploaded": uploaded,
            "failed": failed,
            "duration_seconds": duration,
            "bucket": self.bucket,
        }

    def invalidate_cloudfront(self) -> dict | None:
        """Invalidate CloudFront cache."""
        if not self.cloudfront_id:
            return None

        if self.dry_run:
            console.print(f"[dim]Would invalidate CloudFront distribution: {self.cloudfront_id}[/dim]")
            return {"dry_run": True}

        try:
            response = self.cloudfront.create_invalidation(
                DistributionId=self.cloudfront_id,
                InvalidationBatch={
                    "Paths": {
                        "Quantity": 1,
                        "Items": ["/*"],
                    },
                    "CallerReference": datetime.now(timezone.utc).isoformat(),
                },
            )
            invalidation_id = response["Invalidation"]["Id"]
            console.print(f"[green]Created CloudFront invalidation: {invalidation_id}[/green]")
            return {"invalidation_id": invalidation_id}
        except ClientError as e:
            logger.error("cloudfront_invalidation_failed", error=str(e))
            return {"error": str(e)}


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy Dorothy static site to S3")
    parser.add_argument(
        "--bucket",
        "-b",
        type=str,
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket name (or set S3_BUCKET env var)",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="output",
        help="Source directory (default: ./output)",
    )
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--cloudfront-id",
        type=str,
        default=os.environ.get("CLOUDFRONT_ID"),
        help="CloudFront distribution ID for cache invalidation",
    )
    parser.add_argument(
        "--invalidate",
        "-i",
        action="store_true",
        help="Invalidate CloudFront cache after upload",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview changes without uploading",
    )

    args = parser.parse_args()

    if not args.bucket:
        console.print("[red]Error: S3 bucket required. Use --bucket or set S3_BUCKET env var[/red]")
        return

    source_dir = Path(args.source)
    cloudfront_id = args.cloudfront_id if args.invalidate else None

    deployer = S3Deployer(
        bucket=args.bucket,
        source_dir=source_dir,
        region=args.region,
        cloudfront_id=cloudfront_id,
        dry_run=args.dry_run,
    )

    deployer.sync()

    if args.invalidate and args.cloudfront_id:
        deployer.invalidate_cloudfront()


if __name__ == "__main__":
    main()
