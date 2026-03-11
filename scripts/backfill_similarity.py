#!/usr/bin/env python3
"""
Backfill similarity edges for existing syntheses.

Looks up original articles by URL, computes pairwise cosine similarity
from their embeddings, and patches the synthesis docs in OpenSearch.

Usage:
    python -m scripts.backfill_similarity              # All current syntheses
    python -m scripts.backfill_similarity --column politics
    python -m scripts.backfill_similarity --dry-run    # Preview without writing
"""

import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import structlog
from rich.console import Console

from src.config import config
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

COLUMNS = ["politics", "tech", "money", "sports", "lifestyle"]


def compute_edges(embeddings_by_index: dict[int, list[float]], threshold: float = 0.3) -> list[dict]:
    """Compute pairwise cosine similarity edges from a sparse index->embedding map."""
    indices = sorted(embeddings_by_index.keys())
    if len(indices) < 2:
        return []

    matrix = np.array([embeddings_by_index[i] for i in indices])
    dist = cosine_distances(matrix)

    edges = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sim = 1.0 - dist[i][j]
            if sim >= threshold:
                edges.append({
                    "source": indices[i],
                    "target": indices[j],
                    "similarity": round(float(sim), 4),
                })
    return edges


def backfill(os_client: OpenSearchClient, columns: list[str], dry_run: bool = False) -> None:
    """Backfill similarity_edges for current syntheses."""
    index_name = os_client.get_current_index_name()

    for column in columns:
        syntheses = os_client.get_syntheses(column=column, limit=200)
        console.print(f"\n[bold]{column}[/bold]: {len(syntheses)} syntheses")

        updated = 0
        skipped = 0

        for syn in syntheses:
            story_id = syn.get("story_id")
            articles = syn.get("articles", [])
            article_urls = [a.get("url") for a in articles if a.get("url")]

            if not article_urls or len(articles) < 2:
                skipped += 1
                continue

            # Already has edges? Skip.
            if syn.get("similarity_edges"):
                skipped += 1
                continue

            # Look up original articles by URL to get embeddings
            embeddings_by_index: dict[int, list[float]] = {}
            for i, url in enumerate(article_urls):
                try:
                    result = os_client.client.search(
                        index=f"{index_name}*",
                        body={
                            "query": {"term": {"url": url}},
                            "size": 1,
                            "_source": ["embedding"],
                        },
                    )
                    hits = result["hits"]["hits"]
                    if hits and hits[0]["_source"].get("embedding"):
                        embeddings_by_index[i] = hits[0]["_source"]["embedding"]
                except Exception:
                    continue

            if len(embeddings_by_index) < 2:
                skipped += 1
                continue

            edges = compute_edges(embeddings_by_index)

            if not edges:
                skipped += 1
                continue

            if dry_run:
                console.print(f"  [dim]{story_id}: {len(edges)} edges (dry run)[/dim]")
                updated += 1
                continue

            try:
                os_client.client.update(
                    index="dorothy-synthesis",
                    id=story_id,
                    body={"doc": {"similarity_edges": edges}},
                )
                updated += 1
                console.print(f"  [green]{story_id}: {len(edges)} edges[/green]")
            except Exception as e:
                console.print(f"  [red]{story_id}: failed - {e}[/red]")

        console.print(f"  Updated: {updated}, Skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill similarity edges")
    parser.add_argument("--column", "-c", type=str, help="Single column to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
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

    columns = [args.column] if args.column else COLUMNS
    backfill(os_client, columns, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
