#!/usr/bin/env python3
"""
Dorothy Embedding Generator

Usage:
    python -m scripts.run_embeddings           # Generate for all missing
    python -m scripts.run_embeddings --limit 100  # Process only 100 articles
    python -m scripts.run_embeddings --batch-size 50  # Custom batch size
"""

import argparse
import sys

import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.config import config
from src.embeddings import EmbeddingClient
from src.embeddings.generator import generate_embeddings
from src.storage import OpenSearchClient

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


def run_embedding_job(batch_size: int, limit: int | None) -> dict:
    """Execute embedding generation job. Returns stats dict."""
    logger.info("embedding_job_started", batch_size=batch_size, limit=limit)

    # Initialize OpenSearch client
    auth_kwargs = {}
    if config.opensearch.username and config.opensearch.password:
        auth_kwargs["username"] = config.opensearch.username
        auth_kwargs["password"] = config.opensearch.password

    os_client = OpenSearchClient(
        host=config.opensearch.host,
        port=config.opensearch.port,
        use_ssl=config.opensearch.use_ssl,
        **auth_kwargs,
    )

    if not os_client.health_check():
        return {"success": False, "error": "OpenSearch unavailable"}

    # Initialize embedding client
    embed_client = EmbeddingClient(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
    )

    if not embed_client.health_check():
        return {"success": False, "error": f"Embedding service unavailable at {config.embedding.base_url}"}

    try:
        index_name = os_client.get_current_index_name()

        # Get count of articles without embeddings
        articles_without = os_client.get_articles_without_embeddings(size=1, index_name=index_name)
        if not articles_without:
            console.print("[green]All articles already have embeddings![/green]")
            return {"success": True, "total_processed": 0, "already_complete": True}

        # Run generation with progress display
        final_stats = {"success": True}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=limit or 10000)

            for batch_stats in generate_embeddings(
                os_client=os_client,
                embed_client=embed_client,
                batch_size=batch_size,
                limit=limit,
                index_name=index_name,
            ):
                progress.update(task, advance=batch_stats.get("batch_size", 0))
                final_stats = batch_stats

            progress.update(task, completed=final_stats.get("total_processed", 0))

        final_stats["success"] = True
        return final_stats

    finally:
        embed_client.close()


def print_stats(stats: dict) -> None:
    """Pretty print embedding stats."""
    if not stats.get("success"):
        console.print(f"[red]Embedding generation failed: {stats.get('error', 'Unknown error')}[/red]")
        return

    if stats.get("already_complete"):
        return

    table = Table(title="Dorothy Embedding Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Processed", str(stats.get("total_processed", 0)))
    table.add_row("Successful", str(stats.get("total_success", 0)))
    table.add_row("Errors", str(stats.get("total_errors", 0)))

    console.print(table)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy Embedding Generator")
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help=f"Batch size for embedding generation (default: {config.embedding.batch_size})",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all)",
    )

    args = parser.parse_args()

    batch_size = args.batch_size or config.embedding.batch_size

    console.print("[bold blue]Dorothy Embedding Generator[/bold blue]")
    console.print(f"Embedding service: {config.embedding.base_url}")
    console.print(f"Model: {config.embedding.model}")
    console.print(f"Batch size: {batch_size}")
    if args.limit:
        console.print(f"Limit: {args.limit}")
    console.print()

    stats = run_embedding_job(batch_size=batch_size, limit=args.limit)
    print_stats(stats)

    sys.exit(0 if stats.get("success") else 1)


if __name__ == "__main__":
    main()
