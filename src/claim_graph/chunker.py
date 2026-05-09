"""Chunk articles into paragraph-sized passages for claim graph analysis."""

import re

from src.claim_graph.models import Chunk


# Sentence boundary: period/question/exclamation followed by space and uppercase letter
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

_RE_IMAGE = re.compile(r'!\[([^\]]*)\]\([^\)]*\)')
_RE_LINK_INLINE = re.compile(r'\[([^\]]+)\]\([^\)]*\)')
_RE_LINK_REF = re.compile(r'\[([^\]]+)\]\[[^\]]*\]')
_RE_HEADER = re.compile(r'^[ \t]*#{1,6}[ \t]+', re.MULTILINE)
# trafilatura sometimes drops ## directly after non-newline content (e.g. video
# index pages where a timestamp is followed by an h2): "01:30## Headline".
# Replace any 2+ consecutive '#' anywhere with a space so words don't fuse.
_RE_HEADER_INLINE = re.compile(r'#{2,}')
_RE_LIST_BULLET = re.compile(r'^[ \t]*[*+\-][ \t]+', re.MULTILINE)
_RE_LIST_ORDERED = re.compile(r'^[ \t]*\d+\.[ \t]+', re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r'^[ \t]*>[ \t]?', re.MULTILINE)
_RE_HRULE = re.compile(r'^[ \t]*[-*_]{3,}[ \t]*$', re.MULTILINE)
_RE_EMPHASIS_STAR = re.compile(r'\*{1,3}([^*\n]+?)\*{1,3}')
_RE_EMPHASIS_UNDER = re.compile(r'(?<!\w)_{1,3}([^_\n]+?)_{1,3}(?!\w)')
_RE_INLINE_CODE = re.compile(r'`+([^`\n]+?)`+')
_RE_BACKSLASH_ESCAPE = re.compile(r'\\([\\`*_{}\[\]()#+\-.!>])')
_RE_BLANK_LINES = re.compile(r'\n{3,}')
# Trafilatura emits the article title as a leading H1. That title is already
# stored separately on the article, and including it as a chunk causes the
# article's own headline to leak into the rendered passages.
_RE_LEADING_TITLE = re.compile(r'\A[ \t]*#[ \t]+[^\n]+(?:\n+|\Z)')


def _drop_leading_title(body: str) -> str:
    """Strip the leading H1 title from a markdown article body, if present."""
    return _RE_LEADING_TITLE.sub('', body, count=1)


# Lines that are pure CTA / footer / promo / byline boilerplate. Trafilatura
# pulls these in as standalone paragraphs from article footers, sidebars, and
# newsletter widgets. They survive merging because the next paragraph is real
# content, polluting both the embeddings and the rendered "From the margins"
# section. Match conservatively — only the obvious cases.
_BOILERPLATE_PATTERNS = [
    re.compile(r'^sign up( here)?\.?$', re.I),
    re.compile(r'^subscribe( now| here| today)?\.?$', re.I),
    re.compile(r'^read more:', re.I),
    re.compile(r'^also read:', re.I),
    re.compile(r'^see also:', re.I),
    re.compile(r'^related( stories| articles| coverage| topics)?:?\s*$', re.I),
    re.compile(r'^more from\s', re.I),
    re.compile(r'^for more\b(?!\s+than\b|\s+information than\b)', re.I),
    # Tweet / Bluesky attribution lines: "— Mario Nawfal (@handle) April 26, 2026"
    re.compile(r'^[—–-]\s*.+\s\(@[\w.]+\)\s+\w+\s+\d+,\s*\d{4}', re.I),
    # Affiliate / commerce disclosures
    re.compile(r'\bmay earn (a )?commission\b', re.I),
    re.compile(r'\bif you buy something through (a )?link', re.I),
    re.compile(r'\bsubscribe for free\b', re.I),
    re.compile(r'^this is an extract from\b', re.I),
    # USA Today-style "Reach her at email and follow her on X @handle"
    re.compile(r'^reach (her|him|them) at \S+@\S+', re.I),
    # "Contributing: Name" / "Contributing from: Name" reporter credits
    re.compile(r'^contributing( from)?:?\s+\w', re.I),
    re.compile(r'^continue reading', re.I),
    re.compile(r'^click here', re.I),
    re.compile(r'^follow us\b', re.I),
    re.compile(r'^reporting by\b', re.I),
    re.compile(r'^editing by\b', re.I),
    re.compile(r'^writing by\b', re.I),
    re.compile(r'^additional reporting by\b', re.I),
    re.compile(r'^our standards:', re.I),
    re.compile(r'^the thomson reuters trust principles', re.I),
    re.compile(r'^available to .+ users only\.?$', re.I),
    re.compile(r'^advertisement$', re.I),
    re.compile(r'^trending( now| stories)?\.?$', re.I),
    re.compile(r'^you may (also )?like', re.I),
    re.compile(r'^recommended( for you| stories)?', re.I),
    re.compile(r'^most popular', re.I),
    re.compile(r'^tags?:', re.I),
    re.compile(r'^topics:', re.I),
    re.compile(r'^categor(y|ies):', re.I),
    re.compile(r'^©\s*\d{4}', re.I),
    re.compile(r'^copyright\b', re.I),
    re.compile(r'all rights reserved', re.I),
]


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _BOILERPLATE_PATTERNS)


def _drop_boilerplate(paragraph: str) -> str:
    """Remove leading/trailing boilerplate lines from a paragraph.

    Internal boilerplate lines (between two prose lines) are left alone — those
    are usually rare and removing them can fuse unrelated thoughts. Blank lines
    are transparent to this scan so we can chain past them.
    """
    lines = paragraph.split('\n')
    while lines and (not lines[0].strip() or _is_boilerplate_line(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _is_boilerplate_line(lines[-1])):
        lines.pop()
    return '\n'.join(lines).strip()


def strip_markdown(text: str) -> str:
    """Convert markdown-formatted text to plain prose, preserving paragraph breaks.

    trafilatura emits the article body as markdown. The chunker then splits
    it into passages that get rendered verbatim in story templates, so any
    leftover markers (## headers, **bold**, [links](url)) leak into the page.
    Strip them here so the rendered text reads as clean prose.
    """
    if not text:
        return ""
    text = _RE_IMAGE.sub(r'\1', text)
    text = _RE_LINK_INLINE.sub(r'\1', text)
    text = _RE_LINK_REF.sub(r'\1', text)
    text = _RE_HEADER.sub('', text)
    text = _RE_HEADER_INLINE.sub(' ', text)
    text = _RE_LIST_BULLET.sub('', text)
    text = _RE_LIST_ORDERED.sub('', text)
    text = _RE_BLOCKQUOTE.sub('', text)
    text = _RE_HRULE.sub('', text)
    text = _RE_EMPHASIS_STAR.sub(r'\1', text)
    text = text.replace('*', '')
    text = _RE_EMPHASIS_UNDER.sub(r'\1', text)
    text = _RE_INLINE_CODE.sub(r'\1', text)
    text = _RE_BACKSLASH_ESCAPE.sub(r'\1', text)
    text = _RE_BLANK_LINES.sub('\n\n', text)
    return text


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
    if body:
        body = _drop_leading_title(body)
        body = strip_markdown(body)
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

    # Clean up: strip whitespace, drop empty, drop pure boilerplate paragraphs.
    cleaned_paragraphs: list[str] = []
    for p in raw_paragraphs:
        p = _drop_boilerplate(p.strip())
        if p:
            cleaned_paragraphs.append(p)
    raw_paragraphs = cleaned_paragraphs

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
