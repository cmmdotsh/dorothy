#!/usr/bin/env python3
"""
Dorothy News Pipeline

Runs the full fetch → embed → cluster → synthesize pipeline on a schedule.

Usage:
    python -m scripts.run_pipeline                    # Run continuously
    python -m scripts.run_pipeline --interval 60     # Custom interval (minutes)
    python -m scripts.run_pipeline --once            # Run once and exit
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
from rich.panel import Panel

from src.config import config
from src.fetcher import fetch_all_sources
from src.models import Article
from src.storage import OpenSearchClient
from src.clustering import StoryGrouper
from src.synthesis import LLMClient, StorySummarizer
from src.embeddings import EmbeddingClient
from src.embeddings.generator import generate_embeddings

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

COLUMNS = ["politics", "tech", "money", "sports", "lifestyle"]


def run_fetch(os_client: OpenSearchClient) -> int:
    """Fetch new articles from all sources. Returns count of new articles."""
    index_name = os_client.ensure_index()
    sources = config.get_active_rss_sources()

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

    return len(seen_urls)


def run_embeddings(os_client: OpenSearchClient) -> int:
    """Generate embeddings for articles missing them. Returns count processed."""
    embed_client = EmbeddingClient(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
    )

    if not embed_client.health_check():
        logger.warning("embedding_service_unavailable", base_url=config.embedding.base_url)
        return 0

    try:
        index_name = os_client.get_current_index_name()

        # Check if any articles need embeddings
        articles_without = os_client.get_articles_without_embeddings(size=1, index_name=index_name)
        if not articles_without:
            return 0

        total_processed = 0
        for batch_stats in generate_embeddings(
            os_client=os_client,
            embed_client=embed_client,
            batch_size=config.embedding.batch_size,
            limit=None,
            index_name=index_name,
        ):
            total_processed = batch_stats.get("total_processed", 0)

        return total_processed
    finally:
        embed_client.close()


def run_synthesis(
    os_client: OpenSearchClient,
    llm_client: LLMClient,
    column: str,
    limit: Optional[int] = None,
) -> int:
    """Synthesize stories for a column. Returns count of stories synthesized."""
    grouper = StoryGrouper(
        os_client,
        min_cluster_size=3,
        min_samples=2,
        max_cluster_size=30,
    )
    stories = grouper.get_stories_for_column(column, size=500)

    # Filter to multi-source stories primarily about this column
    multi_source = []
    for s in stories:
        if s.source_count < 2:
            continue
        col_count = sum(1 for a in s.articles if a.get("column") == column)
        if col_count >= len(s.articles) * 0.5:
            multi_source.append(s)

    if not multi_source:
        return 0

    summarizer = StorySummarizer(llm_client)
    results = []

    stories_to_process = multi_source[:limit] if limit else multi_source
    for story in stories_to_process:
        synthesized = summarizer.synthesize(story)
        if synthesized:
            results.append(synthesized)

    # Store in OpenSearch
    if results:
        result_dicts = [r.to_dict() for r in results]
        os_client.bulk_store_syntheses(result_dicts, column)

    return len(results)


def run_pipeline_cycle(
    os_client: OpenSearchClient,
    llm_client: LLMClient,
    stories_per_column: Optional[int] = None,
) -> dict:
    """Run a full pipeline cycle: fetch → synthesize all columns."""
    start_time = datetime.now(timezone.utc)

    console.print(Panel.fit(
        f"[bold blue]Pipeline Cycle Starting[/bold blue]\n{start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    ))

    # Step 1: Fetch
    console.print("\n[dim]Step 1: Fetching articles...[/dim]")
    new_articles = run_fetch(os_client)
    console.print(f"[green]  Fetched {new_articles} new articles[/green]")

    # Step 2: Generate embeddings
    console.print("\n[dim]Step 2: Generating embeddings...[/dim]")
    embedded_count = run_embeddings(os_client)
    if embedded_count > 0:
        console.print(f"[green]  Generated embeddings for {embedded_count} articles[/green]")
    else:
        console.print(f"[dim]  All articles already have embeddings[/dim]")

    # Step 3: Synthesize each column
    console.print("\n[dim]Step 3: Synthesizing stories...[/dim]")
    synthesis_counts = {}

    for column in COLUMNS:
        console.print(f"  [dim]{column}...[/dim]", end=" ")
        count = run_synthesis(os_client, llm_client, column, limit=stories_per_column)
        synthesis_counts[column] = count
        console.print(f"[green]{count} stories[/green]")

    # Summary
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    total_stories = sum(synthesis_counts.values())

    console.print(Panel.fit(
        f"[bold green]Cycle Complete[/bold green]\n"
        f"Articles: {new_articles} | Embedded: {embedded_count} | Stories: {total_stories} | Duration: {duration:.1f}s"
    ))

    return {
        "new_articles": new_articles,
        "embedded_count": embedded_count,
        "stories_synthesized": total_stories,
        "by_column": synthesis_counts,
        "duration_seconds": duration,
        "timestamp": end_time.isoformat(),
    }


def daemon_mode(interval_minutes: int, stories_per_column: int) -> None:
    """Run pipeline on schedule."""
    console.print(Panel.fit(
        f"[bold green]Dorothy Pipeline Daemon[/bold green]\n"
        f"Running every {interval_minutes} minutes\n"
        f"Press Ctrl+C to stop"
    ))

    # Initialize clients
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

    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    if not llm_client.health_check():
        console.print(f"[red]LLM unavailable at {config.llm.base_url}[/red]")
        sys.exit(1)

    console.print(f"[dim]OpenSearch: {config.opensearch.host}:{config.opensearch.port}[/dim]")
    console.print(f"[dim]LLM: {config.llm.model} @ {config.llm.base_url}[/dim]")
    console.print()

    # Run immediately
    run_pipeline_cycle(os_client, llm_client, stories_per_column)

    # Schedule future runs
    def scheduled_run():
        run_pipeline_cycle(os_client, llm_client, stories_per_column)

    schedule.every(interval_minutes).minutes.do(scheduled_run)

    # Graceful shutdown
    def shutdown_handler(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        llm_client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Main loop
    while True:
        schedule.run_pending()
        time.sleep(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy News Pipeline")
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Pipeline interval in minutes (default: 60)",
    )
    parser.add_argument(
        "--stories",
        "-s",
        type=int,
        default=None,
        help="Max stories per column (default: all multi-source stories)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't loop)",
    )

    args = parser.parse_args()

    if args.once:
        # Single run mode
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

        llm_client = LLMClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

        try:
            run_pipeline_cycle(os_client, llm_client, args.stories)
        finally:
            llm_client.close()
    else:
        daemon_mode(args.interval, args.stories)


if __name__ == "__main__":
    main()
