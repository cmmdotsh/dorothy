"""Assemble articles from claim graph clusters and LLM ordering."""

import structlog

logger = structlog.get_logger(__name__)


def assemble_article(claim_graph: dict, ordering: dict) -> str:
    """Assemble an extractive article from claim graph data and LLM ordering.

    Args:
        claim_graph: The claim graph viz dict with 'corroborated' and 'unique_details'.
        ordering: LLM output with 'ordering' list of {cluster, transition} dicts.

    Returns:
        Assembled article text with attributed passages.
    """
    clusters = claim_graph.get("corroborated", [])
    unique_details = claim_graph.get("unique_details", [])
    order = ordering.get("ordering", [])

    paragraphs = []

    for entry in order:
        idx = entry.get("cluster", -1)
        transition = entry.get("transition", "")

        if idx < 0 or idx >= len(clusters):
            logger.warning("invalid_cluster_index", index=idx, max=len(clusters))
            continue

        cluster = clusters[idx]
        sources = cluster.get("sources", [])
        if not sources:
            continue

        passage = sources[0]["text"]
        attribution = sources[0]["source_name"]
        corroboration = ", ".join(cluster.get("source_names", []))

        block = ""
        if transition:
            block += transition + "\n\n"
        block += passage + " *(" + attribution + ")*"
        if cluster.get("source_count", 0) > 1:
            block += "\n*Corroborated by: " + corroboration + "*"

        paragraphs.append(block)

    if unique_details:
        paragraphs.append("---\n\n**Reported by single sources:**")
        by_source = {}
        for detail in unique_details:
            name = detail.get("source_name", "Unknown")
            if name not in by_source:
                by_source[name] = []
            by_source[name].append(detail["text"])

        for source_name, texts in by_source.items():
            for text in texts:
                paragraphs.append("*" + source_name + "* — " + text)

    return "\n\n".join(paragraphs)
