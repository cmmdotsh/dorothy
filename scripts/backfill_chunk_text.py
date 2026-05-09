"""Strip leftover markdown from chunk text in already-stored claim graphs.

Trafilatura emits article bodies as markdown. Older syntheses chunked that
body verbatim, so passages with `**bold**` or `## headers` ended up rendered
literally on the live site. The chunker now strips markdown at chunk creation
time; this script applies the same cleanup to docs that were synthesized
before the fix landed.

Usage:
    python -m scripts.backfill_chunk_text                    # current edition
    python -m scripts.backfill_chunk_text --all              # all editions
    python -m scripts.backfill_chunk_text --dry-run          # report only
    python -m scripts.backfill_chunk_text --host 100.64.0.8  # remote OS
"""

import argparse
import os
import re
import sys

import structlog

from src.claim_graph.chunker import _drop_boilerplate, strip_markdown
from src.storage import OpenSearchClient

logger = structlog.get_logger(__name__)


def _normalize_title(t: str) -> str:
    """Strip trailing source-suffixes (`'... - AP News'`, `'... | Reuters'`)
    so headlines compare cleanly against trafilatura's H1 (which omits them).
    """
    t = (t or "").strip()
    return re.sub(r'\s+[-|]\s+[\w&\s.]+$', '', t).strip()


def _looks_like_title(first_line: str, candidate_titles: list[str]) -> bool:
    fl = first_line.strip()
    if len(fl) < 20:
        return False
    fl_norm = _normalize_title(fl)
    for title in candidate_titles:
        tn = _normalize_title(title)
        if not tn:
            continue
        if fl_norm == tn:
            return True
        # Prefix relationship in either direction (handles RSS-suffix mismatch)
        if len(fl_norm) >= 20 and (tn.startswith(fl_norm) or fl_norm.startswith(tn)):
            return True
    return False


def _strip_leading_title(text: str, candidate_titles: list[str]) -> str:
    """If the chunk text starts with one of the article headlines from the same
    synthesis (the H1 trafilatura emitted at the top of the body), drop that
    leading line plus any trailing blank line.
    """
    stripped = text.lstrip()
    if not stripped:
        return text
    first_line = stripped.split('\n', 1)[0]
    if _looks_like_title(first_line, candidate_titles):
        rest = stripped.split('\n', 1)[1] if '\n' in stripped else ''
        return rest.lstrip('\n')
    return text


def _clean_text(text: str, candidate_titles: list[str] | None = None) -> str:
    out = strip_markdown(text or "")
    if candidate_titles:
        out = _strip_leading_title(out, candidate_titles)
    return _drop_boilerplate(out)


def _clean_chunk(chunk: dict, candidate_titles: list[str] | None = None) -> bool:
    """Clean a chunk's text in place. Returns True if changed."""
    original = chunk.get("text") or ""
    cleaned = _clean_text(original, candidate_titles)
    if cleaned != original:
        chunk["text"] = cleaned
        return True
    return False


def _clean_claim_graph(graph: dict, articles: list[dict] | None = None) -> int:
    """Apply strip_markdown / boilerplate stripping / leading-title stripping
    to every text field in the claim graph in place.

    Returns the count of fields that changed.
    """
    candidate_titles = [a.get("headline") for a in (articles or []) if a.get("headline")]
    changes = 0
    new_clusters = []
    for cluster in graph.get("corroborated") or []:
        rep = cluster.get("representative_text") or ""
        cleaned_rep = _clean_text(rep, candidate_titles)
        if cleaned_rep != rep:
            cluster["representative_text"] = cleaned_rep
            changes += 1
        new_sources = []
        for src in cluster.get("sources") or []:
            if _clean_chunk(src, candidate_titles):
                changes += 1
            if (src.get("text") or "").strip():
                new_sources.append(src)
        cluster["sources"] = new_sources
        if new_sources:
            # If the rep was the title and got stripped to empty, fall back to
            # the first remaining source's text so the cluster still renders.
            if not (cluster.get("representative_text") or "").strip():
                cluster["representative_text"] = new_sources[0].get("text") or ""
                changes += 1
            new_clusters.append(cluster)
        else:
            changes += 1
    graph["corroborated"] = new_clusters
    new_unique = []
    for detail in graph.get("unique_details") or []:
        if _clean_chunk(detail, candidate_titles):
            changes += 1
        if (detail.get("text") or "").strip():
            new_unique.append(detail)
    graph["unique_details"] = new_unique

    # article_blocks is a derived view (passage + other_versions) that the
    # synthesis step assembled from the pre-backfill chunk text. Clean it too
    # so the embedded JSON on each story page doesn't leak markdown into any
    # downstream consumer (today: nothing renders it visibly, but it's part of
    # the public payload).
    for block in graph.get("article_blocks") or []:
        passage = block.get("passage") or ""
        cleaned = _clean_text(passage, candidate_titles)
        if cleaned != passage:
            block["passage"] = cleaned
            changes += 1
        for ov in block.get("other_versions") or []:
            ov_text = ov.get("text") or ""
            cleaned_ov = _clean_text(ov_text, candidate_titles)
            if cleaned_ov != ov_text:
                ov["text"] = cleaned_ov
                changes += 1
    return changes


def _clean_article_field(article_md: str, candidate_titles: list[str]) -> str:
    """Clean the assembled article markdown without mangling its intentional
    `*Source Name*` italic attributions. Strip leading `## Heading` lines
    inside passages and `**bold**` markers, but preserve the trailing
    `— *Source Name*` markup the assembler emits.
    """
    if not article_md:
        return article_md
    paragraphs = re.split(r'\n{2,}', article_md)
    out_paragraphs: list[str] = []
    for para in paragraphs:
        # Each paragraph from assemble_article looks like:
        #   "<chunk text> — *Source Name*"
        # Split off the trailing attribution before cleaning, then reattach.
        attribution = ""
        m = re.search(r'(\s+[—-]\s+\*[^*\n]+\*\s*)$', para)
        if m:
            attribution = m.group(1)
            body = para[: m.start()]
        else:
            body = para
        cleaned_body = _clean_text(body, candidate_titles)
        if cleaned_body or attribution:
            out_paragraphs.append((cleaned_body + attribution).strip())
    return "\n\n".join(p for p in out_paragraphs if p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OPENSEARCH_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENSEARCH_PORT", "9200")))
    parser.add_argument("--all", action="store_true", help="Process all editions, not just is_current")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    client = OpenSearchClient(host=args.host, port=args.port)
    index = "dorothy-synthesis"

    query: dict
    if args.all:
        query = {"match_all": {}}
    else:
        # Match how render_static selects "current" syntheses: is_current=true
        # OR the field is missing (legacy docs predate the flag).
        query = {
            "bool": {
                "should": [
                    {"term": {"is_current": True}},
                    {"bool": {"must_not": [{"exists": {"field": "is_current"}}]}},
                ],
                "minimum_should_match": 1,
            }
        }

    docs_seen = 0
    docs_changed = 0
    fields_changed = 0
    after = None
    sort = [{"_id": "asc"}]

    while True:
        body: dict = {
            "size": args.batch_size,
            "query": query,
            "sort": sort,
            "_source": ["claim_graph", "articles", "article"],
        }
        if after is not None:
            body["search_after"] = after

        resp = client.client.search(index=index, body=body)
        hits = resp["hits"]["hits"]
        if not hits:
            break

        for hit in hits:
            docs_seen += 1
            doc_id = hit["_id"]
            source = hit["_source"]
            graph = source.get("claim_graph") or {}
            articles = source.get("articles") or []
            candidate_titles = [a.get("headline") for a in articles if a.get("headline")]
            changes = _clean_claim_graph(graph, articles)

            article_md = source.get("article") or ""
            cleaned_article = _clean_article_field(article_md, candidate_titles)
            article_changed = cleaned_article != article_md
            if article_changed:
                changes += 1

            if changes:
                docs_changed += 1
                fields_changed += changes
                if not args.dry_run:
                    doc_patch: dict = {"claim_graph": graph}
                    if article_changed:
                        doc_patch["article"] = cleaned_article
                    client.client.update(
                        index=index,
                        id=doc_id,
                        body={"doc": doc_patch},
                    )

        after = hits[-1]["sort"]
        logger.info(
            "backfill_progress",
            seen=docs_seen,
            docs_changed=docs_changed,
            fields_changed=fields_changed,
        )

    logger.info(
        "backfill_complete",
        seen=docs_seen,
        docs_changed=docs_changed,
        fields_changed=fields_changed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
