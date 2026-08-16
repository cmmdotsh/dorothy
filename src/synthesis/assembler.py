"""Assemble articles from claim graph clusters and LLM ordering."""

import re

import structlog

logger = structlog.get_logger(__name__)


def _entry_index(entry) -> int:
    """Coerce one LLM ordering entry into a cluster index.

    Small models return dicts with varying key names, bare ints, or strings
    like "Fact 3" — accept them all; -1 means unusable.
    """
    if isinstance(entry, dict):
        raw = entry.get("cluster", entry.get("fact_id", entry.get("fact", entry.get("index", -1))))
    else:
        raw = entry
    if isinstance(raw, bool):
        return -1
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        match = re.search(r"\d+", raw)
        if match:
            return int(match.group())
    return -1


def _entry_transition(entry) -> str:
    if isinstance(entry, dict):
        transition = entry.get("transition", "")
        return transition if isinstance(transition, str) else ""
    return ""


def assemble_article(claim_graph: dict, ordering: dict) -> str:
    """Assemble an extractive article from claim graph data and LLM ordering.

    Returns assembled article as markdown text for storage. The template
    also has access to claim_graph directly for richer rendering.
    """
    clusters = claim_graph.get("corroborated", [])
    unique_details = claim_graph.get("unique_details", [])
    order = ordering.get("ordering", [])

    paragraphs = []

    for entry in order:
        idx = _entry_index(entry)
        transition = _entry_transition(entry)

        if idx < 0 or idx >= len(clusters):
            logger.warning("invalid_cluster_index", index=idx, max=len(clusters))
            continue

        cluster = clusters[idx]
        sources = cluster.get("sources", [])
        if not sources:
            continue

        passage = sources[0]["text"]
        attribution = sources[0]["source_name"]

        if transition:
            paragraphs.append(transition)
        paragraphs.append(passage + " — *" + attribution + "*")

    if unique_details:
        by_source = {}
        for detail in unique_details:
            name = detail.get("source_name", "Unknown")
            if name not in by_source:
                by_source[name] = []
            by_source[name].append(detail["text"])

        for source_name, texts in by_source.items():
            paragraphs.append("*" + source_name + "* — " + texts[0])

    return "\n\n".join(paragraphs)


def build_article_blocks(claim_graph: dict, ordering: dict) -> list[dict]:
    """Build structured article blocks for template rendering.

    Returns a list of block dicts, each with:
      - type: "fact" or "unique"
      - transition: optional transition sentence (facts only)
      - passage: the source text
      - attribution: source name
      - source_bias: bias rating of the attributed source
      - corroborated_by: list of source names that confirm this fact
      - source_count: number of confirming sources
      - similarity: avg similarity score
      - other_versions: list of {source_name, source_bias, text} from other sources
    """
    clusters = claim_graph.get("corroborated", [])
    unique_details = claim_graph.get("unique_details", [])
    order = ordering.get("ordering", [])

    blocks = []

    for entry in order:
        idx = _entry_index(entry)
        transition = _entry_transition(entry)

        if idx < 0 or idx >= len(clusters):
            continue

        cluster = clusters[idx]
        sources = cluster.get("sources", [])
        if not sources:
            continue

        other_versions = []
        for src in sources[1:]:
            other_versions.append({
                "source_name": src["source_name"],
                "source_bias": src.get("source_bias", ""),
                "text": src["text"],
            })

        blocks.append({
            "type": "fact",
            "transition": transition,
            "passage": sources[0]["text"],
            "attribution": sources[0]["source_name"],
            "source_bias": sources[0].get("source_bias", ""),
            "corroborated_by": cluster.get("source_names", []),
            "source_count": cluster.get("source_count", 1),
            "similarity": cluster.get("avg_similarity", 0),
            "other_versions": other_versions,
        })

    # Group unique details by source
    if unique_details:
        by_source = {}
        for detail in unique_details:
            name = detail.get("source_name", "Unknown")
            if name not in by_source:
                by_source[name] = {
                    "source_bias": detail.get("source_bias", ""),
                    "texts": [],
                }
            by_source[name]["texts"].append(detail["text"])

        for source_name, info in by_source.items():
            blocks.append({
                "type": "unique",
                "passage": info["texts"][0],
                "attribution": source_name,
                "source_bias": info["source_bias"],
                "extra_count": len(info["texts"]) - 1,
            })

    return blocks
