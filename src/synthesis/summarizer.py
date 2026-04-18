"""Story summarizer for generating balanced news synthesis."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from dateutil import parser as dateutil_parser
from sklearn.metrics.pairwise import cosine_distances
import structlog

from src.claim_graph.models import ClaimGraph
from src.clustering import Story
from src.synthesis.json_utils import extract_json, parse_llm_json, ensure_str, is_degenerate
from src.synthesis.llm_client import LLMClient, LLMError

logger = structlog.get_logger(__name__)

BIAS_ORDER = ["left", "lean-left", "center", "lean-right", "right"]

REGION_ORDER = ["us", "canada", "mexico", "uk", "australia", "india", "japan", "korea", "international"]
REGION_LABELS = {
    "us": "United States",
    "canada": "Canada",
    "mexico": "Mexico",
    "uk": "United Kingdom",
    "australia": "Australia",
    "india": "India",
    "japan": "Japan",
    "korea": "South Korea",
    "international": "International",
}

PERSPECTIVE_ORDER = ["consumer", "enterprise", "academic", "culture"]
PERSPECTIVE_LABELS = {
    "consumer": "Consumer",
    "enterprise": "Enterprise",
    "academic": "Academic",
    "culture": "Culture",
}

# ── Pass 1: Neutral Article ──

ARTICLE_SYSTEM_PROMPT = """You are a senior wire service journalist. Your job is to write
clear, factual news articles using ONLY information from the provided source reports.

Guidelines:
- Write in standard news article style: lead paragraph with the key facts,
  then expanding detail in subsequent paragraphs
- Use neutral, precise language — no editorializing or opinion
- Attribute specific claims to their sources when appropriate
- ONLY include names, quotes, facts, and figures that appear in the source material
- NEVER invent expert commentary, analyst quotes, or reactions not in the sources
- If the sources don't provide enough detail on a point, omit it — do not fill gaps
- Respond in JSON with "headline" and "article" fields"""

# ── Pass 2: Coverage Analysis ──

ANALYSIS_SYSTEM_PROMPT = """You are a media analyst who studies how news outlets across
the political spectrum cover the same events differently. You identify meaningful
patterns in framing, emphasis, omission, and language — not surface-level differences.

Guidelines:
- Focus on substantive differences that reveal editorial perspective
- Note what specific outlets emphasize, downplay, or omit entirely
- Identify differences in language, sourcing, and narrative framing
- Be specific — cite outlet names and concrete examples
- Don't just list differences; explain why they matter
- Write in an analytical but accessible tone
- Respond in JSON with an "analysis" field"""

# ── Sports-specific prompts ──

SPORTS_ARTICLE_SYSTEM_PROMPT = """You are a senior sports journalist. Your job is to write
clear, factual sports articles using ONLY information from the provided source reports.

Guidelines:
- Write in standard sports journalism style: lead with the key result or development,
  then expanding detail in subsequent paragraphs
- Use neutral, precise language — no editorializing or homerism
- Note how different regions cover the same story when relevant
- ONLY include names, scores, stats, and quotes that appear in the source material
- NEVER invent player quotes, analyst commentary, or statistics not in the sources
- If the sources don't provide enough detail on a point, omit it — do not fill gaps
- Respond in JSON with "headline" and "article" fields"""

SPORTS_ANALYSIS_SYSTEM_PROMPT = """You are a sports media analyst who studies how outlets
from different countries and regions cover the same sporting events differently. You
identify meaningful patterns in emphasis, framing, and what stories get covered at all.

Guidelines:
- Focus on regional differences: what a US outlet emphasizes vs UK, vs Australia, etc.
- Note which regions covered the story and which didn't
- Identify differences in which athletes, teams, or angles get prominence
- Be specific — cite outlet names, countries, and concrete examples
- Explain why regional perspectives differ (national interest, local heroes, league relevance)
- Write in an analytical but accessible tone
- Respond in JSON with an "analysis" field"""

# ── Tech-specific prompts ──

TECH_ARTICLE_SYSTEM_PROMPT = """You are a senior technology journalist. Your job is to write
clear, factual tech articles using ONLY information from the provided source reports across
different editorial perspectives — consumer, enterprise, academic, and cultural.

Guidelines:
- Write in standard tech journalism style: lead with the key development or announcement,
  then expanding detail in subsequent paragraphs
- Use neutral, precise language — no hype or editorializing
- Note how different perspectives frame the same story when relevant
- ONLY include names, quotes, figures, and technical details that appear in the source material
- NEVER invent analyst commentary, market predictions, or expert quotes not in the sources
- If the sources don't provide enough detail on a point, omit it — do not fill gaps
- Respond in JSON with "headline" and "article" fields"""

TECH_ANALYSIS_SYSTEM_PROMPT = """You are a tech media analyst who studies how outlets with
different editorial perspectives cover the same technology stories differently. Consumer
outlets focus on products and users, enterprise outlets on business impact, academic outlets
on research and engineering, and culture outlets on societal implications.

Guidelines:
- Focus on perspective differences: what a consumer outlet emphasizes vs enterprise, vs academic
- Note which perspectives covered the story and which didn't
- Identify differences in framing, technical depth, and what implications get highlighted
- Be specific — cite outlet names and concrete examples
- Explain why perspectives differ (audience, editorial mission, expertise)
- Write in an analytical but accessible tone
- Respond in JSON with an "analysis" field"""


_extract_json = extract_json
_parse_llm_json = parse_llm_json


# Minimum word counts to consider LLM output substantive
_MIN_HEADLINE_WORDS = 3
_MIN_ARTICLE_WORDS = 30
_MIN_ANALYSIS_WORDS = 20


_ensure_str = ensure_str
_is_degenerate = is_degenerate


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SynthesizedStory:
    """A story with LLM-generated article and coverage analysis."""

    story_id: str
    original_headline: str
    generated_headline: str
    article: str
    analysis: str
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
    quality_scores: Optional[dict[str, float]] = None
    review_improvements: Optional[list[str]] = None
    claim_graph: Optional[dict] = None

    @property
    def summary(self) -> str:
        """Backward-compat alias: returns the article text."""
        return self.article

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "story_id": self.story_id,
            "original_headline": self.original_headline,
            "generated_headline": self.generated_headline,
            "article": self.article,
            "analysis": self.analysis,
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
            "quality_scores": self.quality_scores,
            "review_improvements": self.review_improvements,
            "claim_graph": self.claim_graph,
        }

    def to_markdown(self) -> str:
        """Format as markdown."""
        bias_str = ", ".join(f"{k}: {v}" for k, v in self.bias_coverage.items())
        sources_str = ", ".join(self.sources_used[:10])
        if len(self.sources_used) > 10:
            sources_str += f" (+{len(self.sources_used) - 10} more)"

        return f"""## {self.generated_headline}

{self.article}

### Coverage Analysis
{self.analysis}

---
**Sources:** {sources_str}
**Bias Coverage:** {bias_str}
**Articles:** {self.article_count}
"""


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
        self._token_budget: Optional[int] = None

    @property
    def token_budget(self) -> int:
        """Lazy-fetch the token budget from the LLM client."""
        if self._token_budget is None:
            self._token_budget = self.llm.get_prompt_token_budget()
            logger.info("token_budget_resolved", budget=self._token_budget)
        return self._token_budget

    def _story_column(self, story: Story) -> str:
        """Get the column of a story."""
        return next(
            (a.get("column") for a in story.articles if a.get("column")),
            "politics",
        )

    def _is_sports_story(self, story: Story) -> bool:
        """Check if a story is from the sports column."""
        return self._story_column(story) == "sports"

    def _is_tech_story(self, story: Story) -> bool:
        """Check if a story is from the tech column."""
        return self._story_column(story) == "tech"

    def _group_articles_by_bias(self, story: Story) -> dict[str, list[dict]]:
        """Group story articles by bias rating."""
        by_bias: dict[str, list[dict]] = defaultdict(list)
        for article in story.articles:
            bias = article.get("source_bias", "unknown")
            by_bias[bias].append(article)
        return dict(by_bias)

    def _group_articles_by_region(self, story: Story) -> dict[str, list[dict]]:
        """Group story articles by geographic region."""
        by_region: dict[str, list[dict]] = defaultdict(list)
        for article in story.articles:
            region = article.get("source_region", "unknown")
            by_region[region].append(article)
        return dict(by_region)

    def _group_articles_by_perspective(self, story: Story) -> dict[str, list[dict]]:
        """Group story articles by editorial perspective."""
        by_perspective: dict[str, list[dict]] = defaultdict(list)
        for article in story.articles:
            perspective = article.get("source_perspective", "unknown")
            by_perspective[perspective].append(article)
        return dict(by_perspective)

    def _build_articles_text_by_perspective(
        self,
        by_perspective: dict[str, list[dict]],
    ) -> str:
        """Format source articles grouped by perspective into a text block."""
        sections = []
        for perspective in PERSPECTIVE_ORDER:
            articles = by_perspective.get(perspective, [])
            if not articles:
                continue
            formatted = "\n".join(self._format_article(a) for a in articles)
            label = PERSPECTIVE_LABELS.get(perspective, perspective.upper())
            sections.append(f"### {label}\n{formatted}")
        # Include any unknown-perspective articles
        unknown = by_perspective.get("unknown", [])
        if unknown:
            formatted = "\n".join(self._format_article(a) for a in unknown)
            sections.append(f"### Other\n{formatted}")
        return "\n\n".join(sections)

    def _build_articles_text_by_region(
        self,
        by_region: dict[str, list[dict]],
    ) -> str:
        """Format source articles grouped by region into a text block."""
        sections = []
        for region in REGION_ORDER:
            articles = by_region.get(region, [])
            if not articles:
                continue
            formatted = "\n".join(self._format_article(a) for a in articles)
            label = REGION_LABELS.get(region, region.upper())
            sections.append(f"### {label}\n{formatted}")
        # Include any unknown-region articles
        unknown = by_region.get("unknown", [])
        if unknown:
            formatted = "\n".join(self._format_article(a) for a in unknown)
            sections.append(f"### OTHER\n{formatted}")
        return "\n\n".join(sections)

    def _format_article(self, article: dict, max_body_chars: int = 2000) -> str:
        """Format a single article for the prompt.

        Prefers full body text (from extraction) over RSS summary.
        """
        source = article.get("source_name", "Unknown")
        headline = article.get("headline", "")
        body = article.get("body")
        summary = article.get("summary", "")

        if body:
            return f"**{source}**: {headline}\n{body[:max_body_chars]}"
        elif summary:
            return f"**{source}**: {headline}\n  {summary[:500]}"
        return f"**{source}**: {headline}"

    def _select_representative_articles(
        self,
        articles: list[dict],
        max_per_bucket: int,
    ) -> list[dict]:
        """
        Select the most representative articles from a list using centroid proximity.

        Computes the centroid of all embeddings, then picks the articles
        closest to the centroid. Articles without embeddings are kept as-is
        (up to the limit).
        """
        embeddings = [a["embedding"] for a in articles if a.get("embedding")]
        no_embedding = [a for a in articles if not a.get("embedding")]

        if len(embeddings) <= max_per_bucket:
            return articles[:max_per_bucket]

        articles_with_emb = [a for a in articles if a.get("embedding")]
        emb_matrix = np.array(embeddings)
        centroid = emb_matrix.mean(axis=0, keepdims=True)
        distances = cosine_distances(centroid, emb_matrix)[0]

        # Sort by distance to centroid (closest = most representative)
        ranked_indices = np.argsort(distances)
        selected = [articles_with_emb[i] for i in ranked_indices[:max_per_bucket]]

        # Fill remaining slots with no-embedding articles if any
        remaining = max_per_bucket - len(selected)
        if remaining > 0 and no_embedding:
            selected.extend(no_embedding[:remaining])

        return selected

    def _build_articles_text(
        self,
        by_bias: dict[str, list[dict]],
    ) -> str:
        """
        Format source articles grouped by bias into a text block.
        Used as input for both the article and analysis generation passes.
        """
        sections = []
        for bias in BIAS_ORDER:
            articles = by_bias.get(bias, [])
            if not articles:
                continue

            formatted = "\n".join(self._format_article(a) for a in articles)
            bias_label = bias.upper().replace("-", " ")
            sections.append(f"### {bias_label}\n{formatted}")

        return "\n\n".join(sections)

    def _build_prompt(self, story: Story) -> str:
        """
        Build the synthesis prompt, using all articles if they fit in the
        token budget, otherwise sampling representative articles per bucket.

        Sports stories are grouped by region; tech by perspective; all others by bias.
        """
        is_sports = self._is_sports_story(story)
        is_tech = self._is_tech_story(story)

        if is_sports:
            by_bucket = self._group_articles_by_region(story)
            build_text = self._build_articles_text_by_region
            bucket_order = REGION_ORDER
            system_prompt = SPORTS_ARTICLE_SYSTEM_PROMPT
        elif is_tech:
            by_bucket = self._group_articles_by_perspective(story)
            build_text = self._build_articles_text_by_perspective
            bucket_order = PERSPECTIVE_ORDER
            system_prompt = TECH_ARTICLE_SYSTEM_PROMPT
        else:
            by_bucket = self._group_articles_by_bias(story)
            build_text = self._build_articles_text
            bucket_order = BIAS_ORDER
            system_prompt = ARTICLE_SYSTEM_PROMPT

        # Try the full prompt first
        full_articles_text = build_text(by_bucket)
        template_overhead = 200
        full_tokens = self.llm.estimate_tokens(
            system_prompt + full_articles_text
        ) + int(template_overhead / 3.5)

        if full_tokens <= self.token_budget:
            logger.debug(
                "using_all_articles",
                story_id=story.id,
                articles=len(story.articles),
                estimated_tokens=full_tokens,
            )
            return full_articles_text

        # Over budget — downsample
        logger.info(
            "token_budget_exceeded",
            story_id=story.id,
            articles=len(story.articles),
            estimated_tokens=full_tokens,
            budget=self.token_budget,
        )

        bucket_count = sum(1 for b in bucket_order if b in by_bucket)
        max_per_bucket = max(1, self.token_budget // (bucket_count * 200))

        for cap in range(min(max_per_bucket, 20), 0, -1):
            sampled = {
                key: self._select_representative_articles(articles, cap)
                for key, articles in by_bucket.items()
            }
            articles_text = build_text(sampled)
            estimated = self.llm.estimate_tokens(
                system_prompt + articles_text
            ) + int(template_overhead / 3.5)

            if estimated <= self.token_budget:
                total_selected = sum(len(v) for v in sampled.values())
                logger.info(
                    "sampled_articles_for_budget",
                    story_id=story.id,
                    original=len(story.articles),
                    selected=total_selected,
                    per_bucket_cap=cap,
                    estimated_tokens=estimated,
                )
                return articles_text

        # Absolute fallback: 1 per bucket
        sampled = {
            key: articles[:1]
            for key, articles in by_bucket.items()
        }
        return build_text(sampled)

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
        self,
        story: Story,
        claim_graph: Optional[ClaimGraph] = None,
    ) -> Optional[SynthesizedStory]:
        """
        Generate a neutral article and coverage analysis for a story.

        Pass 1: Generate a neutral news article from all sources.
                If a claim_graph is provided, uses structured fact alignment
                instead of raw article text.
        Pass 2: Analyze how different outlets covered the story differently,
                using the generated article as context.

        Args:
            story: Story object with articles from multiple sources
            claim_graph: Optional pre-built claim graph for structured input

        Returns:
            SynthesizedStory with article and analysis, or None on error
        """
        if story.source_count < 2:
            logger.warning("story_single_source", story_id=story.id)
            return None

        is_sports = self._is_sports_story(story)
        is_tech = self._is_tech_story(story)

        # _build_prompt returns the formatted source articles text,
        # already token-budget-aware (sampled if needed).
        articles_text = self._build_prompt(story)

        # Select prompts based on column
        if is_sports:
            article_sys = SPORTS_ARTICLE_SYSTEM_PROMPT
            analysis_sys = SPORTS_ANALYSIS_SYSTEM_PROMPT
            analysis_focus = (
                "Write a coverage analysis that examines how outlets from different regions "
                "and countries covered this story.\n"
                "Focus on regional differences in emphasis, framing, which athletes or teams "
                "get prominence, and what was included or omitted.\n"
            )
        elif is_tech:
            article_sys = TECH_ARTICLE_SYSTEM_PROMPT
            analysis_sys = TECH_ANALYSIS_SYSTEM_PROMPT
            analysis_focus = (
                "Write a coverage analysis that examines how outlets from different editorial "
                "perspectives covered this technology story.\n"
                "Focus on differences between consumer, enterprise, academic, and cultural "
                "framing — what each perspective emphasizes, downplays, or omits.\n"
            )
        else:
            article_sys = ARTICLE_SYSTEM_PROMPT
            analysis_sys = ANALYSIS_SYSTEM_PROMPT
            analysis_focus = (
                "Write a coverage analysis that examines how different outlets covered this story.\n"
                "Focus on meaningful differences in framing, emphasis, language, and what was\n"
                "included or omitted by different sources.\n"
            )

        max_retries = 10

        try:
            # Pass 1: Generate neutral article (with retry on degenerate output)
            if claim_graph and (claim_graph.corroborated or claim_graph.unique_details):
                column = self._story_column(story)
                graph_text = claim_graph.to_prompt_text(column)
                article_prompt = (
                    "Below is a structured analysis of news reports covering the same story "
                    "from multiple outlets. Facts corroborated across sources are grouped together, "
                    "and unique details reported by only one source are listed separately.\n\n"
                    f"{graph_text}\n\n"
                    "Write a comprehensive news article based on this analysis. Prioritize "
                    "corroborated facts, note where sources diverge, and include unique details "
                    "with appropriate attribution.\n"
                    'Respond with a JSON object containing "headline" and "article" keys.'
                )
            else:
                article_prompt = (
                    "Below are news reports covering the same story from multiple outlets.\n\n"
                    f"{articles_text}\n\n"
                    "Write a comprehensive news article based on these sources.\n"
                    'Respond with a JSON object containing "headline" and "article" keys.'
                )

            headline = None
            article = None
            for attempt in range(1, max_retries + 1):
                article_response = self.llm.generate(
                    article_prompt, system_prompt=article_sys,
                    skip_thinking=True,
                )
                parsed = _parse_llm_json(article_response)
                headline = _ensure_str(parsed["headline"]).strip()
                article = _ensure_str(parsed["article"]).strip()

                if _is_degenerate(headline, _MIN_HEADLINE_WORDS) or _is_degenerate(article, _MIN_ARTICLE_WORDS):
                    logger.warning(
                        "degenerate_article_output",
                        story_id=story.id,
                        attempt=attempt,
                        headline=headline[:80],
                        article_preview=article[:80],
                    )
                    if attempt == max_retries:
                        logger.error(
                            "article_generation_failed_all_retries",
                            story_id=story.id,
                            retries=max_retries,
                        )
                        return None
                    continue
                break

            logger.info(
                "article_generated",
                story_id=story.id,
                headline=headline[:80],
            )

            # Pass 2: Generate coverage analysis (with retry on degenerate output)
            analysis_prompt = (
                "Here is a neutral article we produced from multiple sources:\n\n"
                "---\n"
                f"{article}\n"
                "---\n\n"
                "And here are the original source reports it was based on:\n\n"
                f"{articles_text}\n\n"
                f"{analysis_focus}"
                'Respond with a JSON object containing an "analysis" key.'
            )

            # Check if analysis prompt fits the budget
            analysis_tokens = self.llm.estimate_tokens(
                analysis_sys + analysis_prompt
            )
            if analysis_tokens > self.token_budget:
                logger.warning(
                    "analysis_prompt_over_budget",
                    story_id=story.id,
                    estimated_tokens=analysis_tokens,
                    budget=self.token_budget,
                )
                truncated_text = articles_text[: int(self.token_budget * 2.5)]
                analysis_prompt = (
                    "Here is a neutral article we produced from multiple sources:\n\n"
                    "---\n"
                    f"{article}\n"
                    "---\n\n"
                    "And here are the original source reports it was based on:\n\n"
                    f"{truncated_text}\n\n"
                    f"{analysis_focus}"
                    'Respond with a JSON object containing an "analysis" key.'
                )

            analysis = None
            for attempt in range(1, max_retries + 1):
                analysis_response = self.llm.generate(
                    analysis_prompt, system_prompt=analysis_sys,
                    max_tokens=6144,
                    skip_thinking=True,
                )
                parsed_analysis = _parse_llm_json(analysis_response)
                analysis = _ensure_str(parsed_analysis["analysis"]).strip()

                if _is_degenerate(analysis, _MIN_ANALYSIS_WORDS):
                    logger.warning(
                        "degenerate_analysis_output",
                        story_id=story.id,
                        attempt=attempt,
                        analysis_preview=analysis[:80],
                    )
                    if attempt == max_retries:
                        logger.error(
                            "analysis_generation_failed_all_retries",
                            story_id=story.id,
                            retries=max_retries,
                        )
                        return None
                    continue
                break

            logger.info(
                "analysis_generated",
                story_id=story.id,
            )

            sources_used = list(set(a.get("source_slug", "") for a in story.articles))
            similarity_edges = self._compute_similarity_edges(story.articles)
            article_refs = self._build_article_refs(story.articles)
            hero_image_url, hero_image_source = self._pick_hero_image(story.articles, is_sports)
            article_urls = sorted(str(a.get("url", "")) for a in story.articles if a.get("url"))
            timing = compute_story_timing(story.articles)

            # Use region spread for sports, bias spread otherwise
            coverage = story.coverage_spread

            result = SynthesizedStory(
                story_id=story.id,
                original_headline=story.headline,
                generated_headline=headline,
                article=article,
                analysis=analysis,
                sources_used=sources_used,
                bias_coverage=coverage,
                article_count=len(story.articles),
                articles=article_refs,
                hero_image_url=hero_image_url,
                hero_image_source=hero_image_source,
                article_urls=article_urls,
                similarity_edges=similarity_edges,
                hotness_score=timing.hotness_score,
                median_pub_date=timing.median_pub_date,
                first_pub_date=timing.first_pub_date,
                last_pub_date=timing.last_pub_date,
                quality_scores=None,
                review_improvements=None,
                claim_graph=claim_graph.to_viz_dict() if claim_graph else None,
            )

            logger.info(
                "story_synthesized",
                story_id=story.id,
                sources=story.source_count,
                articles=len(story.articles),
            )

            return result

        except (LLMError, json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
            logger.error("synthesis_failed", story_id=story.id, error=str(e))
            return None

    def synthesize_stories(
        self, stories: list[Story], limit: Optional[int] = None
    ) -> list[SynthesizedStory]:
        """
        Synthesize multiple stories.

        Args:
            stories: List of Story objects
            limit: Maximum number to process

        Returns:
            List of SynthesizedStory objects
        """
        # Filter to multi-source stories only
        multi_source = [s for s in stories if s.source_count >= 2]

        if limit:
            multi_source = multi_source[:limit]

        results = []
        for story in multi_source:
            result = self.synthesize(story)
            if result:
                results.append(result)

        logger.info(
            "synthesis_batch_complete",
            processed=len(multi_source),
            successful=len(results),
        )

        return results
