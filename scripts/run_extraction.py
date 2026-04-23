#!/usr/bin/env python3
"""
Dorothy Article Extractor (standalone)

Continuously extracts full article body text from URLs in OpenSearch.
Runs independently of the main pipeline — just processes whatever
articles are missing body text.

Usage:
    python -m scripts.run_extraction              # Run until caught up, then exit
    python -m scripts.run_extraction --daemon     # Run continuously, polling every 5 min
    python -m scripts.run_extraction --workers 20 # More parallel domain workers
"""

import argparse
import signal
import sys
import time

import structlog
from rich.console import Console

from src.config import config
from src.fetcher.extractor import ArticleExtractor
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


def run_extraction_pass(
    os_client: OpenSearchClient,
    extractor: ArticleExtractor,
    batch_size: int = 500,
) -> dict:
    """Extract all articles missing body text. Returns stats."""
    index_name = os_client.get_current_index_name()
    total = {"processed": 0, "success": 0, "failed": 0}

    while True:
        articles = os_client.get_articles_without_body(
            size=batch_size,
            index_name=index_name,
        )

        if not articles:
            break

        stats = extractor.extract_batch(articles, os_client, index_name)
        total["processed"] += stats["processed"]
        total["success"] += stats["success"]
        total["failed"] += stats["failed"]

        console.print(
            f"[dim]  Batch: {stats['success']}/{stats['processed']} "
            f"(total: {total['processed']})[/dim]"
        )

        if stats["processed"] < batch_size:
            break

    return total


def main():
    parser = argparse.ArgumentParser(description="Dorothy Article Extractor")
    parser.add_argument(
        "--workers", "-w", type=int, default=10,
        help="Parallel domain workers (default: 10)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=500,
        help="Articles per extraction batch (default: 500)",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run continuously, polling for new articles",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=300,
        help="Seconds between polls in daemon mode (default: 300)",
    )
    args = parser.parse_args()

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
        console.print("[red]OpenSearch unavailable[/red]")
        sys.exit(1)

    extractor = ArticleExtractor(
        timeout=config.extractor.timeout,
        delay=config.extractor.delay,
        max_workers=args.workers,
    )

    # Graceful shutdown
    running = True
    def shutdown(signum, frame):
        nonlocal running
        console.print("\n[yellow]Shutting down...[/yellow]")
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.daemon:
        console.print(f"[bold]Dorothy Extractor[/bold] — daemon mode, polling every {args.interval}s")
        while running:
            stats = run_extraction_pass(os_client, extractor, args.batch_size)
            if stats["processed"] > 0:
                console.print(
                    f"[green]Extracted {stats['success']}/{stats['processed']} articles[/green]"
                )
            else:
                console.print("[dim]No articles need extraction[/dim]")
            for _ in range(args.interval):
                if not running:
                    break
                time.sleep(1)
    else:
        console.print("[bold]Dorothy Extractor[/bold] — single pass")
        stats = run_extraction_pass(os_client, extractor, args.batch_size)
        console.print(
            f"[green]Done: {stats['success']}/{stats['processed']} extracted, "
            f"{stats['failed']} failed[/green]"
        )


if __name__ == "__main__":
    main()
