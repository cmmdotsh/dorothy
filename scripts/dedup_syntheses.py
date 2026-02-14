#!/usr/bin/env python3
"""
Deduplicate syntheses in OpenSearch.

Groups syntheses by headline similarity (Jaccard on words), keeps the best
version per group (highest article_count), marks the rest as historical.

Usage:
    python -m scripts.dedup_syntheses              # Dry run (preview)
    python -m scripts.dedup_syntheses --apply       # Actually delete duplicates
"""

import argparse
import re
from collections import defaultdict

import structlog
from rich.console import Console
from rich.table import Table

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

console = Console()

COLUMNS = ["politics", "tech", "money", "sports", "lifestyle"]

# Words to ignore when computing headline similarity
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "if", "then", "than", "that", "this", "it", "its",
    "over", "after", "amid", "among", "about", "into", "up", "out",
}


def headline_words(headline: str) -> set[str]:
    """Extract meaningful words from a headline."""
    words = set(re.findall(r"[a-z]+", headline.lower()))
    return words - STOPWORDS


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_duplicates(syntheses: list[dict], threshold: float = 0.5) -> list[list[dict]]:
    """Group syntheses by headline similarity using union-find."""
    n = len(syntheses)
    word_sets = [headline_words(s.get("generated_headline", "")) for s in syntheses]

    # Union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(word_sets[i], word_sets[j]) >= threshold:
                union(i, j)

    # Group by root
    groups: dict[int, list[dict]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(syntheses[i])

    return list(groups.values())


def dedup_column(client: OpenSearchClient, column: str, apply: bool = False) -> tuple[int, int]:
    """Deduplicate syntheses for a column. Returns (kept, removed)."""
    syntheses = client.get_syntheses(column=column, limit=500)

    if not syntheses:
        return 0, 0

    groups = group_duplicates(syntheses, threshold=0.5)

    kept = 0
    removed = 0
    to_delete = []

    for group in groups:
        if len(group) == 1:
            kept += 1
            continue

        # Keep the one with highest article_count
        group.sort(key=lambda s: s.get("article_count", 0), reverse=True)
        best = group[0]
        kept += 1

        for dup in group[1:]:
            to_delete.append(dup)
            removed += 1

        console.print(
            f"  [green]KEEP[/green] {best['story_id']}: "
            f"[{best.get('article_count', 0)} articles] "
            f"{best.get('generated_headline', '?')[:70]}"
        )
        for dup in group[1:]:
            console.print(
                f"  [red] DEL[/red] {dup['story_id']}: "
                f"[{dup.get('article_count', 0)} articles] "
                f"{dup.get('generated_headline', '?')[:70]}"
            )
        console.print()

    if apply and to_delete:
        for dup in to_delete:
            story_id = dup.get("story_id")
            if story_id:
                try:
                    client.client.delete(index="dorothy-synthesis", id=story_id)
                except Exception as e:
                    console.print(f"  [red]Failed to delete {story_id}: {e}[/red]")

    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="Deduplicate Dorothy syntheses")
    parser.add_argument("--apply", action="store_true", help="Actually delete duplicates")
    parser.add_argument("--column", type=str, help="Only dedup a specific column")
    args = parser.parse_args()

    client = OpenSearchClient()

    if not client.health_check():
        console.print("[red]OpenSearch unavailable[/red]")
        return

    columns = [args.column] if args.column else COLUMNS

    table = Table(title="Dedup Results")
    table.add_column("Column")
    table.add_column("Kept", justify="right")
    table.add_column("Removed", justify="right")

    total_kept = 0
    total_removed = 0

    for column in columns:
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]{column.upper()}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

        kept, removed = dedup_column(client, column, apply=args.apply)
        table.add_row(column, str(kept), str(removed))
        total_kept += kept
        total_removed += removed

    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_kept}[/bold]", f"[bold]{total_removed}[/bold]")
    console.print()
    console.print(table)

    if not args.apply:
        console.print("\n[yellow]Dry run — no changes made. Use --apply to delete duplicates.[/yellow]")


if __name__ == "__main__":
    main()
