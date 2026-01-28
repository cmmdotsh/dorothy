"""Story summarizer for generating balanced news synthesis."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.clustering import Story
from src.synthesis.llm_client import LLMClient, LLMError

logger = structlog.get_logger(__name__)

BIAS_ORDER = ["left", "lean-left", "center", "lean-right", "right"]

SYSTEM_PROMPT = """You are a balanced news synthesizer. Your job is to write neutral,
factual summaries that incorporate perspectives from across the political spectrum.

Guidelines:
- Present core facts that all sources agree on first
- Note where sources differ in framing, emphasis, or interpretation
- Use neutral, non-editorial language
- Attribute specific claims or perspectives to their sources
- Do not take sides or express opinions
- Be concise but comprehensive"""

SYNTHESIS_PROMPT_TEMPLATE = """Below are news articles covering the same story from sources across the political spectrum.

{articles_by_bias}

Based on these sources, write:
1. A neutral headline (one line)
2. A balanced 2-3 paragraph summary that:
   - States the core facts all sources agree on
   - Notes any differences in how sources frame or interpret events
   - Attributes perspectives to their sources when appropriate

Format your response as:
HEADLINE: [your headline]

SUMMARY:
[your balanced summary]"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SynthesizedStory:
    """A story with LLM-generated balanced summary."""

    story_id: str
    original_headline: str
    generated_headline: str
    summary: str
    sources_used: list[str] = field(default_factory=list)
    bias_coverage: dict[str, int] = field(default_factory=dict)
    article_count: int = 0
    generated_at: datetime = field(default_factory=_utcnow)
    articles: list[dict] = field(default_factory=list)
    hero_image_url: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "story_id": self.story_id,
            "original_headline": self.original_headline,
            "generated_headline": self.generated_headline,
            "summary": self.summary,
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

{self.summary}

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
        max_articles_per_bias: int = 3,
    ):
        self.llm = llm_client
        self.max_articles_per_bias = max_articles_per_bias

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

    def _build_prompt(self, story: Story) -> str:
        """Build the synthesis prompt from story articles."""
        by_bias = self._group_articles_by_bias(story)

        sections = []
        for bias in BIAS_ORDER:
            articles = by_bias.get(bias, [])
            if not articles:
                continue

            # Limit articles per bias category
            selected = articles[: self.max_articles_per_bias]
            formatted = "\n".join(self._format_article(a) for a in selected)

            bias_label = bias.upper().replace("-", " ")
            sections.append(f"### {bias_label}\n{formatted}")

        articles_text = "\n\n".join(sections)
        return SYNTHESIS_PROMPT_TEMPLATE.format(articles_by_bias=articles_text)

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response into headline and summary."""
        headline = ""
        summary = ""

        lines = response.strip().split("\n")
        in_summary = False

        for line in lines:
            if line.startswith("HEADLINE:"):
                headline = line.replace("HEADLINE:", "").strip()
            elif line.startswith("SUMMARY:"):
                in_summary = True
            elif in_summary:
                summary += line + "\n"

        # Fallback if format wasn't followed
        if not headline and not summary:
            # Just use the whole response as summary
            summary = response
            headline = response.split("\n")[0][:100]

        return headline.strip(), summary.strip()

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
        Generate a balanced summary for a story.

        Args:
            story: Story object with articles from multiple sources

        Returns:
            SynthesizedStory with generated headline and summary, or None on error
        """
        if story.source_count < 2:
            logger.warning("story_single_source", story_id=story.id)
            return None

        prompt = self._build_prompt(story)

        try:
            response = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
            headline, summary = self._parse_response(response)

            sources_used = list(set(a.get("source_slug", "") for a in story.articles))
            article_refs = self._build_article_refs(story.articles)
            hero_image = self._pick_hero_image(story.articles)

            result = SynthesizedStory(
                story_id=story.id,
                original_headline=story.headline,
                generated_headline=headline,
                summary=summary,
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
