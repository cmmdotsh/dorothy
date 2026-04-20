"""Story summarizer for generating balanced news synthesis."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from dateutil import parser as dateutil_parser
from sklearn.metrics.pairwise import cosine_distances
import structlog

from src.claim_graph.models import ClaimGraph
from src.clustering import Story
from src.synthesis.assembler import assemble_article
from src.synthesis.json_utils import parse_llm_json
from src.synthesis.llm_client import LLMClient, LLMError

logger = structlog.get_logger(__name__)

ORDERING_SYSTEM_PROMPT = """You arrange verified news facts into a coherent article structure.

You receive a list of corroborated facts, each confirmed by multiple news sources.
Return a JSON object with:
- "headline": a neutral, factual headline for the story
- "ordering": the facts arranged in logical narrative order, each with a short transition sentence

Rules:
- The first fact gets an empty transition (it is the lead)
- Transitions are structural only ("Meanwhile...", "The situation was further complicated by...")
- NEVER introduce new facts, names, or claims in transitions
- Order facts from most newsworthy to least newsworthy
- Respond ONLY with the JSON object"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SynthesizedStory:
    """A story assembled from extracted source passages."""

    story_id: str
    original_headline: str
    generated_headline: str
    article: str
    sources_used: list[str] = field(default_factory=list)
    bias_coverage: dict[str, int] = field(default_factory=dict)
    article_count: int = 0
    generated_at: datetime = field(default_factory=_utcnow)
    articles: list[dict] = field(default_factory=list)
    hero_image_url: Optional[str] = None
    hero_image_source: Optional[str] = None
    article_urls: list[str] = field(default_factory=list)
    similarity_edges: list[dict] = field(default_factory=list)
    edition: int = 1
    is_current: bool = True
    hotness_score: float = 0.0
    median_pub_date: Optional[str] = None
    first_pub_date: Optional[str] = None
    last_pub_date: Optional[str] = None
    claim_graph: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "story_id": self.story_id,
            "original_headline": self.original_headline,
            "generated_headline": self.generated_headline,
            "article": self.article,
            "sources_used": self.sources_used,
            "bias_coverage": self.bias_coverage,
            "article_count": self.article_count,
            "generated_at": self.generated_at.isoformat(),
            "articles": self.articles,
            "hero_image_url": self.hero_image_url,
            "hero_image_source": self.hero_image_source,
            "article_urls": self.article_urls,
            "similarity_edges": self.similarity_edges,
            "edition": self.edition,
            "is_current": self.is_current,
            "hotness_score": self.hotness_score,
            "median_pub_date": self.median_pub_date,
            "first_pub_date": self.first_pub_date,
            "last_pub_date": self.last_pub_date,
            "claim_graph": self.claim_graph,
        }


@dataclass
class StoryTiming:
    """Timing metadata derived from article pub_dates."""
    hotness_score: float = 0.0
    median_pub_date: Optional[str] = None
    first_pub_date: Optional[str] = None
    last_pub_date: Optional[str] = None


def compute_story_timing(articles: list[dict], now: Optional[datetime] = None) -> StoryTiming:
    """Compute hotness score and story timing from article pub_dates.

    hotness = article_count / max(1, hours_since_median_pub_date) * source_diversity_bonus
    """
    if now is None:
        now = _utcnow()

    pub_dates = []
    for a in articles:
        pd = a.get("pub_date")
        if not pd:
            continue
        if isinstance(pd, str):
            try:
                pd = dateutil_parser.isoparse(pd)
            except (ValueError, TypeError):
                continue
        if pd.tzinfo is None:
            pd = pd.replace(tzinfo=timezone.utc)
        pub_dates.append(pd)

    if not pub_dates:
        return StoryTiming()

    pub_dates.sort()
    median_idx = len(pub_dates) // 2
    median_date = pub_dates[median_idx]

    hours_since_median = max(1.0, (now - median_date).total_seconds() / 3600)
    article_count = len(articles)

    column = next((a.get("column") for a in articles if a.get("column")), None)
    if column == "sports":
        unique_dims = len(set(a.get("source_region", "unknown") for a in articles))
    elif column == "tech":
        unique_dims = len(set(a.get("source_perspective", "unknown") for a in articles))
    else:
        unique_dims = len(set(a.get("source_bias", "unknown") for a in articles))
    source_diversity_bonus = 1.0 + 0.1 * max(0, unique_dims - 1)

    hotness = (article_count / hours_since_median) * source_diversity_bonus

    return StoryTiming(
        hotness_score=round(hotness, 4),
        median_pub_date=median_date.isoformat(),
        first_pub_date=pub_dates[0].isoformat(),
        last_pub_date=pub_dates[-1].isoformat(),
    )


class StorySummarizer:
    """Generates balanced summaries for multi-source stories."""

    def __init__(
        self,
        llm_client: LLMClient,
    ):
        self.llm = llm_client

    def _story_column(self, story: Story) -> str:
        """Get the column of a story."""
        return next(
            (a.get("column") for a in story.articles if a.get("column")),
            "politics",
        )

    # URL patterns that indicate a tiny thumbnail
    _THUMB_PATTERNS = re.compile(
        r'/thumb[s]?/'
        r'|[-_.]thumb\.'
        r'|[-_.]small\.'
        r'|[-_.]tiny\.'
        r'|[-_./]\d{2,3}x\d{2,3}[-_./]'  # e.g. 120x90, 150x150
        r'|/s\d{2,3}/'            # e.g. /s100/
        r'|[?&]w=\d{1,2}\b'      # e.g. ?w=75
        r'|[?&]width=\d{1,2}\b',
        re.IGNORECASE,
    )

    def _is_hero_worthy(self, url: str) -> bool:
        """Check if an image URL looks like a full-size image (not a thumbnail)."""
        return not self._THUMB_PATTERNS.search(url)

    def _pick_hero_image(self, articles: list[dict], is_sports: bool = False) -> tuple[Optional[str], Optional[str]]:
        """Pick the best hero image from articles.

        For non-sports: prefers center sources for neutral framing.
        For sports: just picks the first available image.
        Filters out obvious thumbnails by URL pattern.

        Returns:
            (image_url, source_name) tuple
        """
        if not is_sports:
            preference_order = ["center", "lean-left", "lean-right", "left", "right"]
            for bias in preference_order:
                for article in articles:
                    url = article.get("image_url", "")
                    if article.get("source_bias") == bias and url and self._is_hero_worthy(url):
                        return url, article.get("source_name", "")

        # Fallback (or sports default): first article with a decent image
        for article in articles:
            url = article.get("image_url", "")
            if url and self._is_hero_worthy(url):
                return url, article.get("source_name", "")

        # Last resort: any image at all
        for article in articles:
            if article.get("image_url"):
                return article["image_url"], article.get("source_name", "")

        return None, None

    def _compute_similarity_edges(self, articles: list[dict], threshold: float = 0.3) -> list[dict]:
        """Compute pairwise cosine similarity between articles with embeddings.

        Returns edge list with indices into the articles list:
        [{"source": 0, "target": 2, "similarity": 0.87}, ...]
        """
        embeddings = []
        indices = []
        for i, a in enumerate(articles):
            if a.get("embedding"):
                embeddings.append(a["embedding"])
                indices.append(i)

        if len(embeddings) < 2:
            return []

        emb_matrix = np.array(embeddings)
        dist_matrix = cosine_distances(emb_matrix)

        edges = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                similarity = 1.0 - dist_matrix[i][j]
                if similarity >= threshold:
                    edges.append({
                        "source": indices[i],
                        "target": indices[j],
                        "similarity": round(float(similarity), 4),
                    })

        return edges

    def _build_article_refs(self, articles: list[dict]) -> list[dict]:
        """Build article reference list for storage."""
        return [
            {
                "url": str(a.get("url", "")),
                "headline": a.get("headline", ""),
                "source_name": a.get("source_name", ""),
                "source_slug": a.get("source_slug", ""),
                "source_bias": a.get("source_bias", ""),
                "source_region": a.get("source_region"),
                "source_perspective": a.get("source_perspective"),
                "image_url": a.get("image_url"),
            }
            for a in articles
        ]

    def synthesize(
        self, story: Story, claim_graph: ClaimGraph,
    ) -> Optional[SynthesizedStory]:
        """Synthesize a story using extractive assembly."""
        if not claim_graph or len(claim_graph.corroborated) < 3:
            logger.info(
                "skipping_insufficient_corroborated_facts",
                story_id=story.id,
                corroborated=len(claim_graph.corroborated) if claim_graph else 0,
            )
            return None

        articles_with_body = [a for a in story.articles if a.get("body")]
        body_sources = set(a.get("source_slug", "") for a in articles_with_body)
        if len(articles_with_body) < 3 or len(body_sources) < 2:
            logger.info(
                "skipping_insufficient_body_text",
                story_id=story.id,
                articles_with_body=len(articles_with_body),
                body_sources=len(body_sources),
            )
            return None

        facts = []
        for i, cluster in enumerate(claim_graph.corroborated):
            src_names = ", ".join(cluster.source_names)
            facts.append(
                "Fact %d (%d sources: %s): %s" % (
                    i, cluster.source_count, src_names,
                    cluster.representative_text[:200],
                )
            )

        prompt = (
            "Arrange these corroborated facts into a news article.\n\n"
            + "\n".join(facts)
            + '\n\nReturn JSON with "headline" and "ordering" keys.'
        )

        try:
            response = self.llm.generate(
                prompt,
                system_prompt=ORDERING_SYSTEM_PROMPT,
                skip_thinking=True,
                max_tokens=1024,
            )
            ordering = parse_llm_json(response)

            if "headline" not in ordering or "ordering" not in ordering:
                logger.error("invalid_ordering_response", story_id=story.id)
                return None

            viz_dict = claim_graph.to_viz_dict()
            article = assemble_article(viz_dict, ordering)

            if not article or len(article.split()) < 20:
                logger.warning("degenerate_article", story_id=story.id)
                return None

            is_sports = self._story_column(story) == "sports"
            sources_used = list(set(a.get("source_slug", "") for a in story.articles))
            similarity_edges = self._compute_similarity_edges(story.articles)
            article_refs = self._build_article_refs(story.articles)
            hero_url, hero_src = self._pick_hero_image(story.articles, is_sports)
            article_urls = sorted(str(a.get("url", "")) for a in story.articles if a.get("url"))
            timing = compute_story_timing(story.articles)
            coverage = story.coverage_spread

            return SynthesizedStory(
                story_id=story.id,
                original_headline=story.headline,
                generated_headline=ordering["headline"],
                article=article,
                sources_used=sources_used,
                bias_coverage=coverage,
                article_count=len(story.articles),
                articles=article_refs,
                hero_image_url=hero_url,
                hero_image_source=hero_src,
                article_urls=article_urls,
                similarity_edges=similarity_edges,
                hotness_score=timing.hotness_score,
                median_pub_date=timing.median_pub_date,
                first_pub_date=timing.first_pub_date,
                last_pub_date=timing.last_pub_date,
                claim_graph=viz_dict,
            )

        except (LLMError, json.JSONDecodeError, KeyError) as e:
            logger.error("synthesis_failed", story_id=story.id, error=str(e))
            return None
