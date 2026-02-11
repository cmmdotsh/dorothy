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
from scripts.render_static import StaticSiteGenerator

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
    try:
        grouper = StoryGrouper(
            os_client,
            min_cluster_size=3,
            min_samples=2,
        )
        stories = grouper.get_stories_for_column(column, size=2000)

        # Filter to multi-source stories
        multi_source = [s for s in stories if s.source_count >= 2]

        if not multi_source:
            return 0

        summarizer = StorySummarizer(llm_client)
        results = []

        stories_to_process = multi_source[:limit] if limit else multi_source
        for story in stories_to_process:
            try:
                synthesized = summarizer.synthesize(story)
                if synthesized:
                    results.append(synthesized)
            except Exception as e:
                logger.error(
                    "story_synthesis_error",
                    story_id=story.id,
                    column=column,
                    error=str(e),
                )
                # Continue processing other stories
                continue

        # Store in OpenSearch
        if results:
            try:
                result_dicts = [r.to_dict() for r in results]
                os_client.bulk_store_syntheses(result_dicts, column)
            except Exception as e:
                logger.error(
                    "synthesis_storage_error",
                    column=column,
                    count=len(results),
                    error=str(e),
                )
                # Return 0 since we couldn't store the results
                return 0

        return len(results)

    except Exception as e:
        logger.error(
            "synthesis_column_error",
            column=column,
            error=str(e),
        )
        return 0


def run_render_deploy() -> None:
    """Render static site and deploy to S3 if configured."""
    from pathlib import Path
    import os

    output_dir = Path("output")

    # Render
    console.print("\n[dim]Step 4: Rendering static site...[/dim]")
    try:
        generator = StaticSiteGenerator(output_dir)
        generator.clean()
        generator.generate()
        console.print("[green]  Static site rendered successfully[/green]")
    except Exception as e:
        logger.error("render_failed", error=str(e))
        console.print(f"[red]  Render failed: {e}[/red]")
        return

    # Deploy
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        console.print("[dim]  S3_BUCKET not set, skipping deploy[/dim]")
        return

    console.print("\n[dim]Step 5: Deploying to S3...[/dim]")
    try:
        from scripts.deploy_s3 import S3Deployer
        cloudfront_id = os.environ.get("CLOUDFRONT_ID") or None
        deployer = S3Deployer(
            bucket=bucket,
            source_dir=output_dir,
            region=os.environ.get("AWS_REGION", "us-east-1"),
            cloudfront_id=cloudfront_id,
        )
        result = deployer.sync()
        console.print(f"[green]  Uploaded {result['uploaded']} files[/green]")
        if cloudfront_id:
            deployer.invalidate_cloudfront()
            console.print("[green]  CloudFront cache invalidated[/green]")
    except Exception as e:
        logger.error("deploy_failed", error=str(e))
        console.print(f"[red]  Deploy failed: {e}[/red]")


def run_pipeline_cycle(
    os_client: OpenSearchClient,
    llm_client: LLMClient,
    stories_per_column: Optional[int] = None,
    render_and_deploy: bool = False,
) -> dict:
    """Run a full pipeline cycle: fetch → synthesize all columns."""
    start_time = datetime.now(timezone.utc)

    console.print(Panel.fit(
        f"[bold blue]Pipeline Cycle Starting[/bold blue]\n{start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    ))

    # Step 1: Fetch
    console.print("\n[dim]Step 1: Fetching articles...[/dim]")
    try:
        new_articles = run_fetch(os_client)
        console.print(f"[green]  Fetched {new_articles} new articles[/green]")
    except Exception as e:
        logger.error("fetch_failed", error=str(e))
        console.print(f"[red]  Fetch failed: {e}[/red]")
        new_articles = 0

    # Step 2: Generate embeddings
    console.print("\n[dim]Step 2: Generating embeddings...[/dim]")
    try:
        embedded_count = run_embeddings(os_client)
        if embedded_count > 0:
            console.print(f"[green]  Generated embeddings for {embedded_count} articles[/green]")
        else:
            console.print(f"[dim]  All articles already have embeddings[/dim]")
    except Exception as e:
        logger.error("embedding_failed", error=str(e))
        console.print(f"[red]  Embedding failed: {e}[/red]")
        embedded_count = 0

    # Step 3: Synthesize each column
    console.print("\n[dim]Step 3: Synthesizing stories...[/dim]")
    synthesis_counts = {}

    for column in COLUMNS:
        console.print(f"  [dim]{column}...[/dim]", end=" ")
        try:
            count = run_synthesis(os_client, llm_client, column, limit=stories_per_column)
            synthesis_counts[column] = count
            console.print(f"[green]{count} stories[/green]")
        except Exception as e:
            logger.error("column_synthesis_failed", column=column, error=str(e))
            console.print(f"[red]0 stories (error)[/red]")
            synthesis_counts[column] = 0

    # Step 4+5: Render and deploy
    if render_and_deploy:
        try:
            run_render_deploy()
        except Exception as e:
            logger.error("render_deploy_failed", error=str(e))
            console.print(f"[red]  Render/deploy phase failed: {e}[/red]")

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


def daemon_mode(interval_minutes: int, stories_per_column: int, render_and_deploy: bool = False) -> None:
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
    run_pipeline_cycle(os_client, llm_client, stories_per_column, render_and_deploy)

    # Schedule future runs
    def scheduled_run():
        run_pipeline_cycle(os_client, llm_client, stories_per_column, render_and_deploy)

    # Schedule on the hour if interval is a whole number of hours
    if interval_minutes == 60:
        schedule.every().hour.at(":00").do(scheduled_run)
    elif interval_minutes % 60 == 0:
        hours = interval_minutes // 60
        schedule.every(hours).hours.at(":00").do(scheduled_run)
    else:
        # For non-hour intervals, use minute-based scheduling
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
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Render static site and deploy to S3 after each cycle",
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
            run_pipeline_cycle(os_client, llm_client, args.stories, args.publish)
        finally:
            llm_client.close()
    else:
        daemon_mode(args.interval, args.stories, args.publish)


if __name__ == "__main__":
    main()
