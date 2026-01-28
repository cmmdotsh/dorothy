#!/usr/bin/env python3
"""
Dorothy Story Synthesizer

Usage:
    python -m scripts.run_synthesis                    # Synthesize top politics stories
    python -m scripts.run_synthesis --column tech      # Different column
    python -m scripts.run_synthesis --limit 5          # Limit number of stories
    python -m scripts.run_synthesis --output json      # JSON output
"""

import argparse
import json
import sys

import structlog
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.config import config
from src.storage import OpenSearchClient
from src.clustering import StoryGrouper
from src.synthesis import LLMClient, StorySummarizer

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


def run_synthesis(
    column: str,
    limit: int,
    output_format: str,
    min_cluster_size: int = 3,
    max_cluster_size: int = 30,
    store: bool = True,
) -> list[dict]:
    """Run story synthesis. Returns list of synthesized stories."""
    logger.info("synthesis_job_started", column=column, limit=limit)

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
        logger.error("opensearch_unavailable")
        return []

    # Initialize LLM client
    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    console.print(f"[dim]Checking LLM at {config.llm.base_url}...[/dim]")
    if not llm_client.health_check():
        logger.error("llm_unavailable", url=config.llm.base_url)
        console.print(f"[red]LLM service unavailable at {config.llm.base_url}[/red]")
        return []

    try:
        # Get clustered stories
        console.print(f"[dim]Finding stories in {column} column...[/dim]")
        grouper = StoryGrouper(
            os_client,
            min_cluster_size=min_cluster_size,
            min_samples=2,
            max_cluster_size=max_cluster_size,
        )
        stories = grouper.get_stories_for_column(column, size=500)

        # Filter to multi-source stories that are primarily about this column
        multi_source = []
        for s in stories:
            if s.source_count < 2:
                continue
            # Check that most articles are from this column
            col_count = sum(1 for a in s.articles if a.get("column") == column)
            if col_count >= len(s.articles) * 0.5:  # At least 50% from this column
                multi_source.append(s)

        console.print(f"[green]Found {len(multi_source)} multi-source stories for {column}[/green]")

        if not multi_source:
            console.print("[yellow]No multi-source stories found to synthesize[/yellow]")
            return []

        # Synthesize stories
        summarizer = StorySummarizer(llm_client)
        results = []

        stories_to_process = multi_source[:limit]
        console.print(f"[dim]Synthesizing {len(stories_to_process)} stories...[/dim]")
        console.print()

        for i, story in enumerate(stories_to_process, 1):
            console.print(
                f"[dim]Processing story {i}/{len(stories_to_process)}: "
                f"{story.headline[:50]}...[/dim]"
            )

            synthesized = summarizer.synthesize(story)
            if synthesized:
                results.append(synthesized)

                if output_format == "markdown":
                    console.print()
                    console.print(Panel(Markdown(synthesized.to_markdown())))
                    console.print()

        logger.info(
            "synthesis_job_completed",
            stories_processed=len(stories_to_process),
            successful=len(results),
        )

        # Store results in OpenSearch
        result_dicts = [r.to_dict() for r in results]

        if store and result_dicts:
            console.print(f"[dim]Storing {len(result_dicts)} syntheses in OpenSearch...[/dim]")
            success, errors = os_client.bulk_store_syntheses(result_dicts, column)
            console.print(f"[green]Stored {success} syntheses ({errors} errors)[/green]")

        return result_dicts

    finally:
        llm_client.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy Story Synthesizer")
    parser.add_argument(
        "--column",
        "-c",
        type=str,
        default="politics",
        help="News column to synthesize (default: politics)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=5,
        help="Maximum number of stories to synthesize (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum articles to form a cluster (default: 3)",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=30,
        help="Split clusters larger than this (default: 30)",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't store syntheses in OpenSearch (default: store)",
    )

    args = parser.parse_args()

    console.print("[bold blue]Dorothy Story Synthesizer[/bold blue]")
    console.print(f"LLM: {config.llm.model} @ {config.llm.base_url}")
    console.print(f"Column: {args.column}")
    console.print(f"Limit: {args.limit}")
    console.print()

    results = run_synthesis(
        column=args.column,
        limit=args.limit,
        output_format=args.output,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        store=not args.no_store,
    )

    if args.output == "json":
        print(json.dumps(results, indent=2))

    if results:
        console.print(f"\n[green]Successfully synthesized {len(results)} stories[/green]")
        sys.exit(0)
    else:
        console.print("\n[yellow]No stories were synthesized[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
