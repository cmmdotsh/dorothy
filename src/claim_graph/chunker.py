"""Chunk articles into paragraph-sized passages for claim graph analysis."""

import re

from src.claim_graph.models import Chunk


# Sentence boundary: period/question/exclamation followed by space and uppercase letter
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def chunk_article(
    article: dict,
    min_chars: int = 80,
    max_chars: int = 800,
) -> list[Chunk]:
    """Split an article into paragraph-sized chunks.

    Uses body text if available, otherwise falls back to headline + summary
    as a single chunk. Merges short paragraphs and splits long ones.
    """
    article_id = article.get("id", "")
    source_name = article.get("source_name", "Unknown")
    source_slug = article.get("source_slug", "")
    source_bias = article.get("source_bias", "unknown")
    source_region = article.get("source_region")
    source_perspective = article.get("source_perspective")
    column = article.get("column", "")

    body = article.get("body")
    if not body:
        # Fallback: headline + summary as a single chunk
        headline = article.get("headline", "")
        summary = article.get("summary", "")
        text = f"{headline}\n\n{summary}".strip() if summary else headline
        if not text:
            return []
        return [
            Chunk(
                article_id=article_id,
                source_name=source_name,
                source_slug=source_slug,
                source_bias=source_bias,
                source_region=source_region,
                source_perspective=source_perspective,
                column=column,
                chunk_index=0,
                text=text[:max_chars],
            )
        ]

    # Split on double newlines (markdown paragraph boundaries)
    raw_paragraphs = re.split(r'\n{2,}', body.strip())

    # Clean up: strip whitespace, drop empty
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Merge short paragraphs with the next one
    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        if buffer:
            buffer = f"{buffer}\n\n{para}"
            if len(buffer) >= min_chars:
                merged.append(buffer)
                buffer = ""
        elif len(para) < min_chars:
            buffer = para
        else:
            merged.append(para)
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{buffer}"
        else:
            merged.append(buffer)

    # Split long paragraphs at sentence boundaries
    final: list[str] = []
    for para in merged:
        if len(para) <= max_chars:
            final.append(para)
        else:
            sentences = _SENTENCE_SPLIT.split(para)
            current = ""
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) > max_chars and current:
                    final.append(current)
                    current = sentence
                else:
                    current = candidate
            if current:
                final.append(current)

    # Build Chunk objects
    return [
        Chunk(
            article_id=article_id,
            source_name=source_name,
            source_slug=source_slug,
            source_bias=source_bias,
            source_region=source_region,
            source_perspective=source_perspective,
            column=column,
            chunk_index=i,
            text=text,
        )
        for i, text in enumerate(final)
        if text.strip()
    ]


def chunk_story(articles: list[dict], **kwargs) -> list[Chunk]:
    """Chunk all articles in a story cluster.

    Filters out chunks that have no topical relevance to the story
    (e.g. sidebar content, related stories, navigation text).
    """
    # Build keyword set from all headlines in the cluster
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'not', 'no', 'it', 'its', 'this',
        'that', 'these', 'those', 'he', 'she', 'they', 'we', 'his', 'her',
        'their', 'our', 'my', 'your', 'who', 'what', 'when', 'where', 'how',
        'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'than', 'too', 'very', 'just', 'about', 'after',
        'before', 'between', 'during', 'into', 'through', 'over', 'under',
        'up', 'out', 'off', 'down', 'then', 'once', 'here', 'there', 'also',
        'new', 'says', 'said', 'say', 'news', 'report', 'reports', 'reported',
    }
    headline_words = set()
    for a in articles:
        words = re.findall(r'[a-z]+', a.get('headline', '').lower())
        headline_words.update(w for w in words if w not in stop_words and len(w) > 2)

    chunks: list[Chunk] = []
    for article in articles:
        for chunk in chunk_article(article, **kwargs):
            # Check topical relevance: chunk must share at least 2 keywords with headlines
            chunk_words = set(re.findall(r'[a-z]+', chunk.text.lower()))
            overlap = chunk_words & headline_words
            if len(overlap) >= 2:
                chunks.append(chunk)
    return chunks
