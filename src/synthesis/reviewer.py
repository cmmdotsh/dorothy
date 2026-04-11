"""Article quality reviewer using a separate model (gemma4) as editor."""

from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.synthesis.json_utils import parse_llm_json, ensure_str, is_degenerate
from src.synthesis.ollama_client import OllamaClient, OllamaError

logger = structlog.get_logger(__name__)


REVIEW_SYSTEM_PROMPT = """You are a senior news editor at a wire service. Your job is to review
synthesized news articles against their source material and improve them when needed.

You evaluate articles on four dimensions (1-10 each):
- **Factuality**: Are all claims supported by the source material? No hallucinated details?
- **Neutrality**: Is the language free of editorial bias, loaded words, or one-sided framing?
- **Completeness**: Are important facts, perspectives, and context from the sources included?
- **Structure**: Is the article well-organized with a strong lead, logical flow, and clear writing?

If any dimension scores below 7, rewrite the article to address the issues.
If all dimensions score 7 or above, return the article unchanged.

Respond with a JSON object containing:
- "scores": {"factuality": N, "neutrality": N, "completeness": N, "structure": N}
- "improvements": ["list of specific changes made"] (empty list if no changes)
- "headline": "improved or original headline"
- "article": "improved or original article text"
"""


@dataclass
class ReviewResult:
    """Result of an article quality review."""

    improved_headline: str
    improved_article: str
    improvements_made: list[str] = field(default_factory=list)
    quality_scores: dict[str, float] = field(default_factory=dict)
    was_improved: bool = False


class ArticleReviewer:
    """Reviews and improves synthesized articles using a separate model."""

    def __init__(self, ollama_client: OllamaClient):
        self.llm = ollama_client

    def review_and_improve(
        self,
        headline: str,
        article: str,
        source_articles_text: str,
        column: str,
    ) -> ReviewResult:
        """Review a synthesized article against source material.

        Args:
            headline: The generated headline
            article: The generated article text
            source_articles_text: Formatted source articles (same text used for synthesis)
            column: The column type (politics, tech, sports, etc.)

        Returns:
            ReviewResult with scores, improvements, and potentially rewritten article
        """
        prompt = (
            f"## Generated Article\n"
            f"**Headline:** {headline}\n\n"
            f"{article}\n\n"
            f"## Source Material\n"
            f"{source_articles_text}\n\n"
            f"## Column\n"
            f"This is a {column} article.\n\n"
            f"Review this article against the source material. Score each dimension "
            f"and improve the article if any score is below 7.\n"
            f'Respond with JSON containing "scores", "improvements", "headline", and "article" keys.'
        )

        # Truncate source text if it would blow the context window
        max_prompt_chars = int(self.llm.get_prompt_token_budget() * CHARS_PER_TOKEN)
        if len(prompt) > max_prompt_chars:
            overhead = len(prompt) - len(source_articles_text)
            max_source_chars = max_prompt_chars - overhead - 200
            source_articles_text = source_articles_text[:max_source_chars]
            prompt = (
                f"## Generated Article\n"
                f"**Headline:** {headline}\n\n"
                f"{article}\n\n"
                f"## Source Material (truncated)\n"
                f"{source_articles_text}\n\n"
                f"## Column\n"
                f"This is a {column} article.\n\n"
                f"Review this article against the source material. Score each dimension "
                f"and improve the article if any score is below 7.\n"
                f'Respond with JSON containing "scores", "improvements", "headline", and "article" keys.'
            )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = self.llm.generate(
                    prompt,
                    system_prompt=REVIEW_SYSTEM_PROMPT,
                )
                parsed = parse_llm_json(response)

                scores = parsed.get("scores", {})
                improvements = parsed.get("improvements", [])
                reviewed_headline = ensure_str(parsed.get("headline", headline)).strip()
                reviewed_article = ensure_str(parsed.get("article", article)).strip()

                # Validate scores
                quality_scores = {}
                for dim in ("factuality", "neutrality", "completeness", "structure"):
                    try:
                        quality_scores[dim] = float(scores.get(dim, 0))
                    except (ValueError, TypeError):
                        quality_scores[dim] = 0.0

                # Determine if the article was materially changed
                was_improved = bool(improvements) and not is_degenerate(reviewed_article, 30)

                # If the reviewer returned garbage, fall back to original
                if is_degenerate(reviewed_headline, 3) or is_degenerate(reviewed_article, 30):
                    logger.warning(
                        "reviewer_degenerate_output",
                        attempt=attempt,
                        headline_preview=reviewed_headline[:80],
                        article_preview=reviewed_article[:80],
                    )
                    if attempt < max_retries:
                        continue
                    return ReviewResult(
                        improved_headline=headline,
                        improved_article=article,
                        quality_scores=quality_scores,
                        was_improved=False,
                    )

                if isinstance(improvements, list):
                    improvements = [str(i) for i in improvements]
                else:
                    improvements = []

                return ReviewResult(
                    improved_headline=reviewed_headline,
                    improved_article=reviewed_article,
                    improvements_made=improvements,
                    quality_scores=quality_scores,
                    was_improved=was_improved,
                )

            except (OllamaError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    "review_attempt_failed",
                    attempt=attempt,
                    error=str(e),
                )
                if attempt == max_retries:
                    raise

        # Should not reach here, but return original as fallback
        return ReviewResult(
            improved_headline=headline,
            improved_article=article,
            was_improved=False,
        )


# Import constant from ollama_client for truncation calculation
CHARS_PER_TOKEN = 3.5
