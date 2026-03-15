#!/usr/bin/env python3
"""
Find and re-synthesize degenerate stories (e.g. "..." headlines/articles).

Usage:
    python -m scripts.fix_degenerate              # Dry run — show bad syntheses
    python -m scripts.fix_degenerate --fix         # Delete bad ones and re-synthesize
"""

import argparse
import re

import structlog
from rich.console import Console

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

# Same thresholds as summarizer
_MIN_HEADLINE_WORDS = 3
_MIN_ARTICLE_WORDS = 30


def _is_degenerate(text: str, min_words: int) -> bool:
    """Check if text is degenerate (empty, ellipsis, punctuation-only, etc.)."""
    stripped = re.sub(r'[^\w\s]', '', text).strip()
    if not stripped:
        return True
    return len(stripped.split()) < min_words


def find_degenerate_syntheses(os_client: OpenSearchClient) -> list[dict]:
    """Find all syntheses with degenerate headlines or articles."""
    # Fetch all current syntheses across all columns
    all_syntheses = []
    for column in ["politics", "tech", "money", "sports", "lifestyle"]:
        syntheses = os_client.get_syntheses(column=column, limit=500)
        all_syntheses.extend(syntheses)

    bad = []
    for s in all_syntheses:
        headline = s.get("generated_headline", "")
        article = s.get("article", "")
        analysis = s.get("analysis", "")

        reasons = []
        if _is_degenerate(headline, _MIN_HEADLINE_WORDS):
            reasons.append(f"headline: {headline!r}")
        if _is_degenerate(article, _MIN_ARTICLE_WORDS):
            reasons.append(f"article: {article[:60]!r}")
        if _is_degenerate(analysis, 20):
            reasons.append(f"analysis: {analysis[:60]!r}")

        if reasons:
            s["_degenerate_reasons"] = reasons
            bad.append(s)

    return bad


def fix_degenerate(os_client: OpenSearchClient, llm_client: LLMClient) -> int:
    """Delete degenerate syntheses and re-synthesize their stories."""
    bad = find_degenerate_syntheses(os_client)

    if not bad:
        console.print("[green]No degenerate syntheses found[/green]")
        return 0

    console.print(f"[yellow]Found {len(bad)} degenerate syntheses to fix[/yellow]")

    # Group by column for re-synthesis
    by_column: dict[str, list[dict]] = {}
    for s in bad:
        col = s.get("column", "politics")
        by_column.setdefault(col, []).append(s)

    summarizer = StorySummarizer(llm_client)
    fixed = 0

    for column, syntheses in by_column.items():
        console.print(f"\n[bold]{column}[/bold]: {len(syntheses)} to fix")

        # Delete the bad syntheses
        for s in syntheses:
            story_id = s["story_id"]
            try:
                os_client.client.delete(index="dorothy-synthesis", id=story_id)
                console.print(f"  [dim]Deleted {story_id}[/dim]")
            except Exception as e:
                console.print(f"  [red]Failed to delete {story_id}: {e}[/red]")

        # Re-cluster and find the stories that match the deleted ones
        grouper = StoryGrouper(os_client, min_cluster_size=3, min_samples=2)
        stories = grouper.get_stories_for_column(column, size=2000)
        multi_source = [s for s in stories if s.source_count >= 2]

        # Match stories by article URL overlap
        deleted_url_sets = []
        for s in syntheses:
            urls = set(s.get("article_urls", []))
            if urls:
                deleted_url_sets.append(urls)

        stories_to_resynthesize = []
        for story in multi_source:
            story_urls = set(
                str(a.get("url", "")) for a in story.articles if a.get("url")
            )
            if not story_urls:
                continue

            for deleted_urls in deleted_url_sets:
                intersection = len(story_urls & deleted_urls)
                union = len(story_urls | deleted_urls)
                if union > 0 and intersection / union > 0.15:
                    stories_to_resynthesize.append(story)
                    break

        if not stories_to_resynthesize:
            console.print(f"  [yellow]No matching clusters found to re-synthesize[/yellow]")
            continue

        console.print(f"  [dim]Re-synthesizing {len(stories_to_resynthesize)} stories...[/dim]")

        results = []
        for story in stories_to_resynthesize:
            console.print(f"  [dim]{story.headline[:60]}[/dim]")
            synthesized = summarizer.synthesize(story)
            if synthesized:
                results.append(synthesized)

        if results:
            result_dicts = [r.to_dict() for r in results]
            success, errors = os_client.bulk_store_syntheses(result_dicts, column)
            console.print(f"  [green]Stored {success} re-synthesized stories ({errors} errors)[/green]")
            fixed += success

    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix degenerate syntheses")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Actually delete and re-synthesize (default: dry run)",
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
        return

    if not args.fix:
        # Dry run
        bad = find_degenerate_syntheses(os_client)
        if not bad:
            console.print("[green]No degenerate syntheses found[/green]")
            return

        console.print(f"[yellow]Found {len(bad)} degenerate syntheses:[/yellow]\n")
        for s in bad:
            console.print(f"  [bold]{s['story_id']}[/bold] ({s.get('column', '?')})")
            for reason in s["_degenerate_reasons"]:
                console.print(f"    {reason}")
        console.print(f"\nRun with --fix to delete and re-synthesize")
        return

    # Fix mode
    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    console.print(f"[dim]Checking LLM at {config.llm.base_url}...[/dim]")
    if not llm_client.health_check():
        console.print(f"[red]LLM unavailable at {config.llm.base_url}[/red]")
        return

    try:
        fixed = fix_degenerate(os_client, llm_client)
        console.print(f"\n[green]Fixed {fixed} stories[/green]")
    finally:
        llm_client.close()


if __name__ == "__main__":
    main()
