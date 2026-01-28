#!/usr/bin/env python3
"""
Dorothy RSS Fetch Runner

Usage:
    python -m scripts.run_fetch           # Run once
    python -m scripts.run_fetch --daemon  # Run on schedule
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timezone

import schedule
import structlog
from rich.console import Console
from rich.table import Table

from src.config import config
from src.fetcher import fetch_all_sources
from src.models import Article
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


def run_fetch_job() -> dict:
    """Execute a single fetch cycle. Returns stats dict."""
    start_time = datetime.now(timezone.utc)
    logger.info("fetch_job_started")

    # Only pass auth if configured
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
        logger.error("opensearch_unavailable")
        return {"success": False, "error": "OpenSearch unavailable"}

    index_name = os_client.ensure_index()

    sources = config.get_active_rss_sources()
    logger.info("sources_loaded", count=len(sources))

    articles: list[Article] = []
    seen_urls: set[str] = set()

    for article in fetch_all_sources(sources):
        url_str = str(article.url)
        if url_str in seen_urls:
            continue
        seen_urls.add(url_str)

        if os_client.article_exists(url_str, index_name):
            continue

        articles.append(article)

        if len(articles) >= config.fetcher.batch_size:
            os_client.bulk_index_articles(articles, index_name)
            articles = []

    if articles:
        os_client.bulk_index_articles(articles, index_name)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    total_count = os_client.get_article_count(index_name)

    stats = {
        "success": True,
        "sources_processed": len(sources),
        "new_articles": len(seen_urls),
        "total_in_index": total_count,
        "duration_seconds": duration,
        "timestamp": end_time.isoformat(),
    }

    logger.info("fetch_job_completed", **stats)
    return stats


def print_stats(stats: dict) -> None:
    """Pretty print fetch stats."""
    if not stats.get("success"):
        console.print(f"[red]Fetch failed: {stats.get('error', 'Unknown error')}[/red]")
        return

    table = Table(title="Dorothy Fetch Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Sources Processed", str(stats.get("sources_processed", 0)))
    table.add_row("New Articles", str(stats.get("new_articles", 0)))
    table.add_row("Total in Index", str(stats.get("total_in_index", 0)))
    table.add_row("Duration", f"{stats.get('duration_seconds', 0):.2f}s")

    console.print(table)


def daemon_mode() -> None:
    """Run fetch job on schedule."""
    interval = config.scheduler.fetch_interval_minutes

    logger.info("daemon_mode_started", interval_minutes=interval)
    console.print("[bold green]Dorothy daemon started[/bold green]")
    console.print(f"Fetching every {interval} minutes. Press Ctrl+C to stop.")

    stats = run_fetch_job()
    print_stats(stats)

    schedule.every(interval).minutes.do(lambda: print_stats(run_fetch_job()))

    def shutdown_handler(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        schedule.run_pending()
        time.sleep(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy RSS Fetcher")
    parser.add_argument(
        "--daemon",
        "-d",
        action="store_true",
        help="Run continuously on schedule",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Override fetch interval (minutes)",
    )

    args = parser.parse_args()

    if args.interval:
        config.scheduler.fetch_interval_minutes = args.interval

    if args.daemon:
        daemon_mode()
    else:
        stats = run_fetch_job()
        print_stats(stats)
        sys.exit(0 if stats.get("success") else 1)


if __name__ == "__main__":
    main()
