"""Radio script generation for Dorothy podcast briefings."""

import json

import structlog

from src.synthesis.llm_client import LLMClient, LLMError
from src.synthesis.summarizer import _parse_llm_json

logger = structlog.get_logger(__name__)

SCRIPT_SYSTEM_PROMPT = """You are a veteran public radio news anchor writing a top-of-the-hour \
news briefing script. Your style is clear, authoritative, and conversational — like NPR's \
hourly news updates.

Guidelines:
- Use short, declarative sentences in active voice and present tense
- Write for the ear, not the eye: no parentheses, abbreviations, or complex clauses
- Spell out EVERYTHING — this script is read by text-to-speech. Never use abbreviations, \
acronyms, initialisms, or symbols. Write "percent" not "%", "dollars" not "$", \
"United States" not "US" or "U.S.", "versus" not "vs.", "million" not "M", \
"Chief Executive Officer" on first use then "C.E.O." with periods between letters, etc. \
Numbers under 100 should be written as words. Dates should be fully written out.
- Each story body should be 120-150 words (about 50-60 seconds at broadcast pace)
- The body must NOT repeat or rephrase the headline_read — dive straight into the details
- Use natural transitions between stories ("Turning to...", "In other news...", "Meanwhile...")
- Do not mention the number of outlets or sources covering a story
- The intro and outro should be brief (1-2 sentences each)
- Respond with a JSON object only, no other text"""

SCRIPT_USER_PROMPT_TEMPLATE = """Write a radio news briefing script for the following {count} stories.

{stories_block}

Respond with this exact JSON structure:
{{
  "intro": "From Dorothy, it's {dateline}. Here are the latest headlines.",
  "stories": [
    {{
      "headline_read": "A one-sentence spoken headline for the anchor to read",
      "body": "The 120-150 word broadcast script for this story"
    }}
  ],
  "outro": "That's your Dorothy news update. For full coverage and source links, visit dorothy dot C-M-M dot S-H."
}}"""


def _format_story_for_prompt(story: dict, index: int) -> str:
    """Format a synthesis for inclusion in the script prompt."""
    headline = story.get("generated_headline", "Untitled")
    article = story.get("article", "")
    article_count = story.get("article_count", 0)
    bias_coverage = story.get("bias_coverage", {})

    # Truncate article to ~300 words for prompt budget
    words = article.split()
    if len(words) > 300:
        article = " ".join(words[:300]) + "..."

    REGION_LABELS = {
        "us": "US", "canada": "Canada", "mexico": "Mexico",
        "uk": "UK", "australia": "Australia", "india": "India",
        "japan": "Japan", "korea": "Korea", "international": "Intl",
    }

    column = story.get("_column", story.get("column", ""))
    if column == "sports":
        coverage_str = ", ".join(
            f"{REGION_LABELS.get(k, k)}: {v}" for k, v in bias_coverage.items() if v
        )
        coverage_label = "Regions"
    else:
        coverage_str = ", ".join(f"{k}: {v}" for k, v in bias_coverage.items() if v)
        coverage_label = "Coverage"

    return (
        f"### Story {index + 1}: {headline}\n"
        f"Article: {article}\n"
        f"Sources: {article_count} outlets | {coverage_label}: {coverage_str}"
    )


NEW_FACTS_THRESHOLD = 3  # URLs not in previous manifest to skip penalty


def _get_story_urls(story: dict) -> set[str]:
    """Extract article URLs from a synthesis story."""
    urls = set()
    for article in story.get("articles", []):
        url = article.get("url", "")
        if url:
            urls.add(url)
    return urls


def _apply_recency_penalties(
    stories: list[dict], manifests: list[dict]
) -> list[dict]:
    """Penalize stories that were covered in recent episodes.

    Penalty tiers (by manifest recency, newest first):
    - Last episode (index 0): 0.15x hotness
    - 2-3 episodes ago (index 1-2): 0.4x hotness

    Stories with 3+ new article URLs since last covered skip the penalty.
    """
    # Build lookup: story_id -> (manifest_index, covered_urls)
    covered: dict[str, tuple[int, set[str]]] = {}
    for i, manifest in enumerate(manifests):
        urls_by_story = manifest.get("article_urls_by_story", {})
        for story_id in manifest.get("story_ids", []):
            if story_id not in covered:
                covered[story_id] = (i, set(urls_by_story.get(story_id, [])))

    penalties = {0: 0.15, 1: 0.4, 2: 0.4}

    for story in stories:
        hotness = story.get("hotness_score", 0)
        story_id = story.get("story_id", "")

        if story_id and story_id in covered:
            manifest_idx, prev_urls = covered[story_id]
            current_urls = _get_story_urls(story)
            new_urls = current_urls - prev_urls

            if len(new_urls) >= NEW_FACTS_THRESHOLD:
                logger.info(
                    "story_penalty_skipped_new_facts",
                    story_id=story_id,
                    new_url_count=len(new_urls),
                )
                story["_effective_hotness"] = hotness
            else:
                penalty = penalties.get(manifest_idx, 0.4)
                story["_effective_hotness"] = hotness * penalty
                logger.info(
                    "story_penalized",
                    story_id=story_id,
                    penalty=penalty,
                    original=round(hotness, 3),
                    effective=round(hotness * penalty, 3),
                )
        else:
            story["_effective_hotness"] = hotness

    return stories


class ScriptWriter:
    """Generates radio-style news scripts from synthesized stories."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select_top_stories(
        self,
        syntheses_by_column: dict[str, list[dict]],
        count: int = 5,
        recent_manifests: list[dict] | None = None,
    ) -> list[dict]:
        """Select top stories across all columns by penalized hotness score.

        Stories covered in recent episodes get a hotness penalty unless they
        have 3+ new article URLs since last covered (indicating new facts).

        Args:
            syntheses_by_column: Dict mapping column name to list of synthesis dicts.
            count: Number of stories to select.
            recent_manifests: List of manifest dicts (newest first) from recent episodes.

        Returns:
            Top N stories sorted by penalized hotness.
        """
        all_stories = []
        for column, stories in syntheses_by_column.items():
            for story in stories:
                story["_column"] = column
                all_stories.append(story)

        # Apply recency penalties from episode manifests
        if recent_manifests:
            all_stories = _apply_recency_penalties(all_stories, recent_manifests)

        # Sort by effective_hotness descending
        all_stories.sort(key=lambda s: s.get("_effective_hotness", s.get("hotness_score", 0)), reverse=True)
        return all_stories[:count]

    def generate_script(self, stories: list[dict], dateline: str = "") -> dict:
        """Generate a radio news script from selected stories.

        Args:
            stories: List of synthesis dicts (from select_top_stories).
            dateline: Date string for the briefing (e.g. "February 18, 2026").

        Returns:
            Parsed JSON dict with intro, stories, outro keys.

        Raises:
            LLMError: If LLM generation fails.
            json.JSONDecodeError: If response can't be parsed.
            KeyError: If expected fields are missing.
        """
        stories_block = "\n\n".join(
            _format_story_for_prompt(s, i) for i, s in enumerate(stories)
        )

        prompt = SCRIPT_USER_PROMPT_TEMPLATE.format(
            count=len(stories),
            stories_block=stories_block,
            dateline=dateline or "today",
        )

        response = self.llm.generate(
            prompt,
            system_prompt=SCRIPT_SYSTEM_PROMPT,
            max_tokens=4096,
            skip_thinking=True,
        )

        script = _parse_llm_json(response)

        # Validate structure
        if "intro" not in script or "stories" not in script or "outro" not in script:
            raise KeyError(f"Script missing required fields. Got keys: {list(script.keys())}")

        logger.info(
            "script_generated",
            story_count=len(script["stories"]),
            intro_len=len(script["intro"]),
        )

        return script
