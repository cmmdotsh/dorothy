"""Story summarizer for generating balanced news synthesis."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import structlog

from src.clustering import Story
from src.synthesis.llm_client import LLMClient, LLMError

logger = structlog.get_logger(__name__)

BIAS_ORDER = ["left", "lean-left", "center", "lean-right", "right"]

# ── Pass 1: Neutral Article ──

ARTICLE_SYSTEM_PROMPT = """You are a senior wire service journalist. Your job is to write
clear, factual, comprehensive news articles from multiple source reports.

Guidelines:
- Write in standard news article style: lead paragraph with the key facts,
  then expanding detail in subsequent paragraphs
- Use neutral, precise language — no editorializing or opinion
- Attribute specific claims to their sources when appropriate
- Include relevant context and background
- Write as if this is the definitive account of the story"""

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
- Write in an analytical but accessible tone"""


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

    def _group_articles_by_bias(self, story: Story) -> dict[str, list[dict]]:
        """Group story articles by bias rating."""
        by_bias: dict[str, list[dict]] = defaultdict(list)
        for article in story.articles:
            bias = article.get("source_bias", "unknown")
            by_bias[bias].append(article)
        return dict(by_bias)

    def _format_article(self, article: dict) -> str:
        """Format a single article for the prompt."""
        source = article.get("source_name", "Unknown")
        headline = article.get("headline", "")
        summary = article.get("summary", "")

        if summary:
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
        token budget, otherwise sampling representative articles per bias bucket.
        """
        by_bias = self._group_articles_by_bias(story)

        # Try the full prompt first
        full_articles_text = self._build_articles_text(by_bias)
        # Estimate tokens for the full prompt (system + template + articles)
        template_overhead = 200  # approximate chars for prompt template wrapper
        full_tokens = self.llm.estimate_tokens(
            ARTICLE_SYSTEM_PROMPT + full_articles_text
        ) + int(template_overhead / 3.5)

        if full_tokens <= self.token_budget:
            logger.debug(
                "using_all_articles",
                story_id=story.id,
                articles=len(story.articles),
                estimated_tokens=full_tokens,
            )
            return full_articles_text

        # Over budget — need to downsample via centroid-based representative sampling.
        logger.info(
            "token_budget_exceeded",
            story_id=story.id,
            articles=len(story.articles),
            estimated_tokens=full_tokens,
            budget=self.token_budget,
        )

        # Start with a reasonable per-bucket cap and shrink if needed
        bucket_count = sum(1 for b in BIAS_ORDER if b in by_bias)
        max_per_bucket = max(1, self.token_budget // (bucket_count * 200))

        for cap in range(min(max_per_bucket, 20), 0, -1):
            sampled = {
                bias: self._select_representative_articles(articles, cap)
                for bias, articles in by_bias.items()
            }
            articles_text = self._build_articles_text(sampled)
            estimated = self.llm.estimate_tokens(
                ARTICLE_SYSTEM_PROMPT + articles_text
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

        # Absolute fallback: 1 per bucket (should always fit)
        sampled = {
            bias: articles[:1]
            for bias, articles in by_bias.items()
        }
        return self._build_articles_text(sampled)

    def _parse_article_response(self, response: str) -> tuple[str, str]:
        """Parse the article generation response into headline and article body."""
        headline = ""
        article = ""

        lines = response.strip().split("\n")
        in_article = False

        for line in lines:
            if line.startswith("HEADLINE:"):
                headline = line.replace("HEADLINE:", "").strip()
            elif line.startswith("ARTICLE:"):
                in_article = True
            elif in_article:
                article += line + "\n"

        # Fallback if format wasn't followed
        if not headline and not article:
            article = response
            headline = response.split("\n")[0][:100]

        return headline.strip(), article.strip()

    def _parse_analysis_response(self, response: str) -> str:
        """Parse the analysis generation response."""
        lines = response.strip().split("\n")
        analysis = ""
        in_analysis = False

        for line in lines:
            if line.startswith("ANALYSIS:"):
                in_analysis = True
            elif in_analysis:
                analysis += line + "\n"

        # Fallback: use the whole response
        if not analysis:
            analysis = response

        return analysis.strip()

    def _pick_hero_image(self, articles: list[dict]) -> Optional[str]:
        """Pick the best hero image from articles, preferring center sources."""
        # Prefer images from center/lean sources for neutral framing
        preference_order = ["center", "lean-left", "lean-right", "left", "right"]

        for bias in preference_order:
            for article in articles:
                if article.get("source_bias") == bias and article.get("image_url"):
                    return article["image_url"]

        # Fallback: first article with any image
        for article in articles:
            if article.get("image_url"):
                return article["image_url"]

        return None

    def _build_article_refs(self, articles: list[dict]) -> list[dict]:
        """Build article reference list for storage."""
        return [
            {
                "url": str(a.get("url", "")),
                "headline": a.get("headline", ""),
                "source_name": a.get("source_name", ""),
                "source_slug": a.get("source_slug", ""),
                "source_bias": a.get("source_bias", ""),
                "image_url": a.get("image_url"),
            }
            for a in articles
        ]

    def synthesize(self, story: Story) -> Optional[SynthesizedStory]:
        """
        Generate a neutral article and coverage analysis for a story.

        Pass 1: Generate a neutral news article from all sources.
        Pass 2: Analyze how different outlets covered the story differently,
                using the generated article as context.

        Args:
            story: Story object with articles from multiple sources

        Returns:
            SynthesizedStory with article and analysis, or None on error
        """
        if story.source_count < 2:
            logger.warning("story_single_source", story_id=story.id)
            return None

        # _build_prompt returns the formatted source articles text,
        # already token-budget-aware (sampled if needed).
        articles_text = self._build_prompt(story)

        try:
            # Pass 1: Generate neutral article
            # Use f-string to avoid issues with {curly braces} in source content
            article_prompt = (
                "Below are news reports covering the same story from multiple outlets.\n\n"
                f"{articles_text}\n\n"
                "Write a comprehensive news article based on these sources.\n\n"
                "Format your response as:\n"
                "HEADLINE: [a clear, neutral headline]\n\n"
                "ARTICLE:\n"
                "[your news article]"
            )
            article_response = self.llm.generate(
                article_prompt, system_prompt=ARTICLE_SYSTEM_PROMPT
            )
            headline, article = self._parse_article_response(article_response)

            logger.info(
                "article_generated",
                story_id=story.id,
                headline=headline[:80],
            )

            # Pass 2: Generate coverage analysis (with article as context)
            # Use % formatting to avoid issues with {curly braces} in LLM output
            analysis_prompt = (
                "Here is a neutral article we produced from multiple sources:\n\n"
                "---\n"
                f"{article}\n"
                "---\n\n"
                "And here are the original source reports it was based on:\n\n"
                f"{articles_text}\n\n"
                "Write a coverage analysis that examines how different outlets covered this story.\n"
                "Focus on meaningful differences in framing, emphasis, language, and what was\n"
                "included or omitted by different sources.\n\n"
                "Format your response as:\n"
                "ANALYSIS:\n"
                "[your coverage analysis]"
            )

            # Check if analysis prompt fits the budget (it's bigger than
            # the article prompt since it includes the generated article too)
            analysis_tokens = self.llm.estimate_tokens(
                ANALYSIS_SYSTEM_PROMPT + analysis_prompt
            )
            if analysis_tokens > self.token_budget:
                logger.warning(
                    "analysis_prompt_over_budget",
                    story_id=story.id,
                    estimated_tokens=analysis_tokens,
                    budget=self.token_budget,
                )
                # Truncate the source articles in the analysis prompt —
                # the model already has the synthesized article as primary context
                truncated_text = articles_text[: int(self.token_budget * 2.5)]
                analysis_prompt = (
                    "Here is a neutral article we produced from multiple sources:\n\n"
                    "---\n"
                    f"{article}\n"
                    "---\n\n"
                    "And here are the original source reports it was based on:\n\n"
                    f"{truncated_text}\n\n"
                    "Write a coverage analysis that examines how different outlets covered this story.\n"
                    "Focus on meaningful differences in framing, emphasis, language, and what was\n"
                    "included or omitted by different sources.\n\n"
                    "Format your response as:\n"
                    "ANALYSIS:\n"
                    "[your coverage analysis]"
                )

            analysis_response = self.llm.generate(
                analysis_prompt, system_prompt=ANALYSIS_SYSTEM_PROMPT
            )
            analysis = self._parse_analysis_response(analysis_response)

            logger.info(
                "analysis_generated",
                story_id=story.id,
            )

            sources_used = list(set(a.get("source_slug", "") for a in story.articles))
            article_refs = self._build_article_refs(story.articles)
            hero_image = self._pick_hero_image(story.articles)

            result = SynthesizedStory(
                story_id=story.id,
                original_headline=story.headline,
                generated_headline=headline,
                article=article,
                analysis=analysis,
                sources_used=sources_used,
                bias_coverage=story.bias_spread,
                article_count=len(story.articles),
                articles=article_refs,
                hero_image_url=hero_image,
            )

            logger.info(
                "story_synthesized",
                story_id=story.id,
                sources=story.source_count,
                articles=len(story.articles),
            )

            return result

        except LLMError as e:
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
