#!/usr/bin/env python3
"""
Backfill event-thread candidates for existing syntheses.

For every dorothy-synthesis doc lacking `summary_embedding`, embeds
generated_headline + "\\n" + article[:500] (the same text the pipeline
embeds for new syntheses) and updates the doc with `summary_embedding`
and `thread_candidate: true` so the event matcher can pick them up as
recurrence-birth candidates.

Idempotent: docs that already have a summary_embedding are skipped.

Usage:
    python -m scripts.backfill_event_candidates            # Embed + tag all syntheses
    python -m scripts.backfill_event_candidates --dry-run  # Preview without writing
"""

import argparse

import structlog
from rich.console import Console

from src.config import config
from src.embeddings import EmbeddingClient
from src.storage import OpenSearchClient

# structlog is configured in main() (not at import) so the module is
# import-safe for tests; see scripts/backfill_similarity.py for the
# CLI-time configuration this mirrors.
logger = structlog.get_logger(__name__)
console = Console()

COLUMNS = ["politics", "tech", "money", "sports", "lifestyle"]


def embedding_text(syn: dict) -> str:
    """Build the text embedded for a synthesis doc (headline + article prefix)."""
    # Synthesis docs store their prose in the `article` field
    # (SynthesizedStory.to_dict); 500-char cap matches the pipeline.
    return (syn.get("generated_headline") or "") + "\n" + (syn.get("article") or "")[:500]


def backfill(
    os_client: OpenSearchClient,
    embed_client: EmbeddingClient,
    columns: list[str],
    dry_run: bool = False,
) -> None:
    """Backfill summary_embedding + thread_candidate for current syntheses."""
    for column in columns:
        syntheses = os_client.get_syntheses(column=column, limit=200)
        console.print(f"\n[bold]{column}[/bold]: {len(syntheses)} syntheses")

        updated = 0
        skipped = 0
        failed = 0

        for syn in syntheses:
            story_id = syn.get("story_id")

            # Already embedded? Skip (idempotent).
            if syn.get("summary_embedding"):
                skipped += 1
                continue

            try:
                embedding = embed_client.embed_single(embedding_text(syn))
            except Exception as e:
                logger.warning("synthesis_embedding_failed", story_id=story_id, error=str(e))
                console.print(f"  [red]{story_id}: embed failed - {e}[/red]")
                failed += 1
                continue

            if not embedding:
                logger.warning("synthesis_embedding_empty", story_id=story_id)
                console.print(f"  [red]{story_id}: empty embedding[/red]")
                failed += 1
                continue

            if dry_run:
                console.print(
                    f"  [dim]{story_id}: would set summary_embedding "
                    f"({len(embedding)}-d) + thread_candidate=true (dry run)[/dim]"
                )
                updated += 1
                continue

            try:
                os_client.client.update(
                    index="dorothy-synthesis",
                    id=story_id,
                    body={"doc": {"summary_embedding": embedding, "thread_candidate": True}},
                )
                updated += 1
                console.print(
                    f"  [green]{story_id}: summary_embedding ({len(embedding)}-d) "
                    f"+ thread_candidate=true[/green]"
                )
            except Exception as e:
                logger.warning("synthesis_update_failed", story_id=story_id, error=str(e))
                console.print(f"  [red]{story_id}: update failed - {e}[/red]")
                failed += 1

        logger.info(
            "backfill_column_complete",
            column=column,
            updated=updated,
            skipped=skipped,
            failed=failed,
        )
        console.print(f"  Updated: {updated}, Skipped: {skipped}, Failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill summary embeddings + thread_candidate for existing syntheses"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

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

    # Built the same way as scripts/run_pipeline.py's run_embeddings.
    embed_client = EmbeddingClient(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
    )

    if not embed_client.health_check():
        logger.warning("embedding_service_unavailable", base_url=config.embedding.base_url)
        return

    try:
        backfill(os_client, embed_client, COLUMNS, dry_run=args.dry_run)
    finally:
        embed_client.close()


if __name__ == "__main__":
    main()
