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
from typing import Optional

import schedule
import structlog
from rich.console import Console
from rich.table import Table

from src.config import config
from src.embeddings import EmbeddingClient
from src.embeddings.generator import generate_embeddings_for_articles
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


def run_fetch_job(embed: bool = True) -> dict:
    """Execute a single fetch cycle, optionally embedding inline. Returns stats dict.

    Articles are stored as soon as they're fetched. If embed=True and the
    embedding service is reachable, vectors are generated in the same batch.
    Failed embedding doesn't fail the fetch — the publisher's embedding
    catch-up step will pick up unembedded articles later.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("fetch_job_started", embed=embed)

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

    embed_client: Optional[EmbeddingClient] = None
    if embed and config.embedding.enabled:
        embed_client = EmbeddingClient(
            base_url=config.embedding.base_url,
            model=config.embedding.model,
        )
        if not embed_client.health_check():
            logger.warning("embedding_unavailable_skipping_inline", base_url=config.embedding.base_url)
            embed_client = None

    index_name = os_client.ensure_index()

    sources = config.get_active_rss_sources()
    logger.info("sources_loaded", count=len(sources))

    articles: list[Article] = []
    seen_urls: set[str] = set()
    embedded_count = 0

    def _store_and_embed(batch: list[Article]) -> int:
        os_client.bulk_index_articles(batch, index_name)
        if not embed_client:
            return 0
        article_dicts = [
            {"id": a.id, "headline": a.headline, "summary": a.summary}
            for a in batch
        ]
        try:
            updates = generate_embeddings_for_articles(article_dicts, embed_client)
            if updates:
                os_client.bulk_update_embeddings(updates, index_name)
            return len(updates) if updates else 0
        except Exception as e:
            logger.warning("inline_embed_failed", error=str(e), batch_size=len(batch))
            return 0

    try:
        for article in fetch_all_sources(sources):
            url_str = str(article.url)
            if url_str in seen_urls:
                continue
            seen_urls.add(url_str)

            if os_client.article_exists(url_str, index_name):
                continue

            articles.append(article)

            if len(articles) >= config.fetcher.batch_size:
                embedded_count += _store_and_embed(articles)
                articles = []

        if articles:
            embedded_count += _store_and_embed(articles)
    finally:
        if embed_client:
            embed_client.close()

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    total_count = os_client.get_article_count(index_name)

    stats = {
        "success": True,
        "sources_processed": len(sources),
        "new_articles": len(seen_urls),
        "embedded_inline": embedded_count,
        "total_in_index": total_count,
        "duration_seconds": duration,
        "timestamp": end_time.isoformat(),
    }

    logger.info("fetch_job_completed", **stats)
    return stats


def run_fetch_job_guarded(embed: bool = True) -> dict:
    """Run one fetch cycle, never letting an exception escape.

    Wraps run_fetch_job for daemon use: a failing run (e.g. an OpenSearch
    rollover error) is logged and reported via the stats dict so the daemon
    loop and scheduler survive to try again next interval. KeyboardInterrupt
    and SystemExit derive from BaseException, not Exception, so they still
    propagate to the caller. One-shot (--once) runs call run_fetch_job
    directly and keep their unhandled-exception behavior.
    """
    try:
        return run_fetch_job(embed=embed)
    except Exception as e:
        logger.error("fetch_job_failed", error=str(e), exc_info=True)
        return {"success": False, "error": str(e)}


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
    table.add_row("Embedded Inline", str(stats.get("embedded_inline", 0)))
    table.add_row("Total in Index", str(stats.get("total_in_index", 0)))
    table.add_row("Duration", f"{stats.get('duration_seconds', 0):.2f}s")

    console.print(table)


def daemon_mode(embed: bool = True) -> None:
    """Run fetch job on schedule."""
    interval = config.scheduler.fetch_interval_minutes

    logger.info("daemon_mode_started", interval_minutes=interval, embed=embed)
    console.print("[bold green]Dorothy fetcher daemon started[/bold green]")
    console.print(
        f"Fetching every {interval} minutes "
        f"({'with' if embed else 'without'} inline embedding). Press Ctrl+C to stop."
    )

    stats = run_fetch_job_guarded(embed=embed)
    print_stats(stats)

    schedule.every(interval).minutes.do(lambda: print_stats(run_fetch_job_guarded(embed=embed)))

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
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip inline embedding (use when LMStudio is intentionally offline)",
    )

    args = parser.parse_args()

    if args.interval:
        config.scheduler.fetch_interval_minutes = args.interval

    embed = not args.no_embed

    if args.daemon:
        daemon_mode(embed=embed)
    else:
        stats = run_fetch_job(embed=embed)
        print_stats(stats)
        sys.exit(0 if stats.get("success") else 1)


if __name__ == "__main__":
    main()
