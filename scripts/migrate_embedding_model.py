"""Re-embed every LIVE vector store with the configured embedding model.

Cosine similarity between vectors from different embedding models is
meaningless, so a model change must atomically re-embed everything a k-NN
or cosine comparison reads:

  1. Articles in the clustering window (current + previous monthly index,
     pub_date >= now - 2 * CLUSTERING_WINDOW_HOURS for slack) — clustering
     never reads older docs, so older months are left stale on purpose.
  2. Every dorothy-synthesis doc with a summary_embedding (event matching).
  3. Every dorothy-events summary_embedding (event matching).

Each rewritten doc is stamped with `embedding_model` so uniformity can be
asserted afterward. Idempotent: docs already stamped with the target model
are skipped. Run with the fetcher/publisher STOPPED.

Usage:
    python -m scripts.migrate_embedding_model [--dry-run]
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

import structlog
from rich.console import Console

from src.config import config
from src.embeddings.client import EmbeddingClient
from src.embeddings.generator import _prepare_text
from src.storage.opensearch import OpenSearchClient

logger = structlog.get_logger()
console = Console()

BATCH = 16


def _batched(items, size=BATCH):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _scan(os_client, index, query, source_fields):
    """Fetch all matching docs (id + selected fields) via search_after."""
    docs, after = [], None
    while True:
        body = {
            "size": 500,
            "query": query,
            "_source": source_fields,
            "sort": [{"_id": "asc"}],
        }
        if after:
            body["search_after"] = after
        r = os_client.client.search(index=index, body=body, ignore_unavailable=True)
        hits = r["hits"]["hits"]
        if not hits:
            return docs
        docs.extend({"_id": h["_id"], "_index": h["_index"], **h["_source"]} for h in hits)
        after = hits[-1]["sort"]


def _reembed(os_client, client, model, rows, text_of, label, dry_run):
    """rows: list of doc dicts. text_of: doc -> embedding text."""
    todo = [r for r in rows if r.get("embedding_model") != model]
    skipped = len(rows) - len(todo)
    updated = failed = 0
    for chunk in _batched(todo):
        texts = [text_of(r) for r in chunk]
        try:
            vectors = client.embed(texts)
        except Exception as e:
            logger.error("migration_embed_failed", label=label, error=str(e))
            failed += len(chunk)
            continue
        for row, vec in zip(chunk, vectors):
            if dry_run:
                updated += 1
                continue
            try:
                os_client.client.update(
                    index=row["_index"],
                    id=row["_id"],
                    body={"doc": {row["_field"]: vec, "embedding_model": model}},
                )
                updated += 1
            except Exception as e:
                logger.error("migration_update_failed", label=label, id=row["_id"], error=str(e))
                failed += 1
    console.print(
        f"  {label}: [green]{updated} re-embedded[/green], {skipped} already {model}, "
        f"{'[red]' + str(failed) + ' failed[/red]' if failed else '0 failed'}"
        + (" [dim](dry run — no writes)[/dim]" if dry_run else "")
    )
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = config.embedding.model
    console.print(f"[bold]Migrating all live vectors to:[/bold] {model}")

    os_client = OpenSearchClient(
        host=config.opensearch.host,
        port=config.opensearch.port,
        use_ssl=config.opensearch.use_ssl,
    )
    client = EmbeddingClient(base_url=config.embedding.base_url, model=model)
    if not client.health_check():
        console.print("[red]Embedding service unreachable — aborting.[/red]")
        return 1

    now = datetime.now(timezone.utc)
    window = timedelta(hours=config.clustering.window_hours * 2)
    since = (now - window).isoformat()
    prev_month = (now - window).strftime("%Y-%m")
    indices = f"dorothy-articles-{prev_month},{os_client.get_current_index_name()}"

    failures = 0

    # 1. Articles in (twice) the clustering window
    articles = _scan(
        os_client, indices,
        {"bool": {"must": [
            {"range": {"pub_date": {"gte": since}}},
            {"exists": {"field": "embedding"}},
        ]}},
        ["headline", "summary", "embedding_model"],
    )
    for r in articles:
        r["_field"] = "embedding"
    failures += _reembed(
        os_client, client, model, articles,
        lambda r: _prepare_text({"headline": r.get("headline"), "summary": r.get("summary")}),
        f"articles ({indices}, pub_date>={since[:16]})", args.dry_run,
    )

    # 2. Synthesis summary embeddings (same text formula as the pipeline)
    synths = _scan(
        os_client, "dorothy-synthesis",
        {"exists": {"field": "summary_embedding"}},
        ["generated_headline", "article", "embedding_model"],
    )
    for r in synths:
        r["_field"] = "summary_embedding"
    failures += _reembed(
        os_client, client, model, synths,
        lambda r: (r.get("generated_headline") or "") + "\n" + (r.get("article") or "")[:500],
        "syntheses", args.dry_run,
    )

    # 3. Event thread summaries
    events = _scan(
        os_client, "dorothy-events",
        {"exists": {"field": "summary_embedding"}},
        ["title", "summary", "embedding_model"],
    )
    for r in events:
        r["_field"] = "summary_embedding"
    failures += _reembed(
        os_client, client, model, events,
        lambda r: r.get("summary") or r.get("title") or "",
        "events", args.dry_run,
    )

    client.close()
    if not args.dry_run and not failures:
        console.print("[bold green]Migration complete — all live vector stores uniform.[/bold green]")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
