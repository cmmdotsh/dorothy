# Extractive Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Dorothy's generative synthesis pipeline with extractive synthesis — the LLM orders claim graph clusters and writes transitions, code assembles articles from real source passages.

**Architecture:** The claim graph identifies corroborated facts as text chunks with source attribution. A single LLM call orders the clusters and writes a headline + transition sentences (~200 tokens). Code assembles the article by inserting attributed source passages in the LLM's specified order. The analysis tab is replaced by the claim graph D3 visualization.

**Tech Stack:** Python 3.13, OpenSearch, LMStudio (MLX), Jinja2, D3.js

**Spec:** `docs/superpowers/specs/2026-04-18-extractive-synthesis-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/synthesis/summarizer.py` | Rewrite | New extractive synthesizer: ordering prompt, article assembly |
| `src/synthesis/__init__.py` | Modify | Remove reviewer/ollama exports |
| `src/synthesis/reviewer.py` | Delete | No longer needed |
| `src/synthesis/ollama_client.py` | Delete | No longer needed |
| `src/config.py` | Modify | Remove ReviewerSettings, remove synthesis_concurrency |
| `scripts/run_pipeline.py` | Modify | Remove reviewer setup, simplify run_synthesis |
| `src/storage/opensearch.py` | Modify | Drop quality_scores/review_improvements from store methods |
| `src/web/templates/story.html` | Modify | Restructure tabs: Article, Evidence, Sources |
| `tests/test_article_assembler.py` | Create | Tests for article assembly from claim graph data |

---

### Task 1: Strip Reviewer Infrastructure

**Files:**
- Delete: `src/synthesis/reviewer.py`
- Delete: `src/synthesis/ollama_client.py`
- Modify: `src/synthesis/__init__.py`
- Modify: `src/config.py`
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Delete reviewer and ollama client**

```bash
rm src/synthesis/reviewer.py src/synthesis/ollama_client.py
```

- [ ] **Step 2: Update src/synthesis/__init__.py**

Replace the current exports with:

```python
"""Synthesis package for Dorothy."""

from src.synthesis.llm_client import LLMClient, LLMError
from src.synthesis.summarizer import StorySummarizer, SynthesizedStory

__all__ = [
    "LLMClient",
    "LLMError",
    "StorySummarizer",
    "SynthesizedStory",
]
```

- [ ] **Step 3: Remove ReviewerSettings from config.py**

Remove the `ReviewerSettings` class (lines 71-81) and remove `synthesis_concurrency` from `LLMSettings` (line 65). Remove `self.reviewer = ReviewerSettings()` from `DorothyConfig.__init__`.

The `LLMSettings` class should become:

```python
class LLMSettings(BaseSettings):
    """LLM service settings for story synthesis (via LMStudio)."""

    base_url: str = "http://192.168.0.149:1234"
    model: str = "mlx-community/qwen3.5-35b-a3b"
    temperature: float = 0.3
    max_tokens: int = 4096
    context_length: int = 32768

    class Config:
        env_prefix = "LLM_"
```

In `DorothyConfig.__init__`, remove the `self.reviewer` line.

- [ ] **Step 4: Strip reviewer from run_pipeline.py**

Remove these imports:

```python
from src.synthesis.ollama_client import OllamaClient
from src.synthesis.reviewer import ArticleReviewer
```

Delete the `_review_story()` function (lines 172-208).

In `run_synthesis()`, remove the `reviewer` parameter and the reviewer call. The loop in Pass 2 becomes:

```python
        results = []

        for story in stories_to_synthesize:
            try:
                synthesized, story = _write_story(summarizer, story, edition, graph_builder)
                if synthesized:
                    results.append(synthesized)
            except Exception as e:
                logger.error("story_synthesis_error", story_id=story.id, column=column, error=str(e))
```

Remove `reviewer` parameter from both `run_synthesis()` and `run_pipeline_cycle()` signatures.

In `daemon_mode()` (lines 560-574), remove the entire reviewer setup block (creating `LLMClient` for reviewer, `ArticleReviewer`, health check, console print).

In `main()` / `--once` path (lines 667-681), remove the reviewer setup block and the `reviewer_client.close()` in the finally block.

Remove `reviewer=reviewer` from all `run_pipeline_cycle()` and `run_synthesis()` calls.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "strip reviewer infrastructure

Remove reviewer.py, ollama_client.py, ReviewerSettings, and all
reviewer wiring from pipeline. Extractive synthesis eliminates
the need for a separate review model."
```

---

### Task 2: Rewrite SynthesizedStory Dataclass

**Files:**
- Modify: `src/synthesis/summarizer.py`

- [ ] **Step 1: Remove unused fields from SynthesizedStory**

Remove `analysis`, `quality_scores`, and `review_improvements` fields. Change `claim_graph` from `Optional[dict]` to `dict`. The dataclass becomes:

```python
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
```

- [ ] **Step 2: Remove the to_markdown() method**

Delete the `to_markdown()` method and the `summary` property. They reference the removed `analysis` field.

- [ ] **Step 3: Commit**

```bash
git add src/synthesis/summarizer.py
git commit -m "simplify SynthesizedStory dataclass

Remove analysis, quality_scores, review_improvements fields.
Make claim_graph required. Drop to_markdown() and summary property."
```

---

### Task 3: Write Article Assembler

**Files:**
- Create: `src/synthesis/assembler.py`
- Create: `tests/test_article_assembler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_article_assembler.py`:

```python
"""Tests for extractive article assembly."""

from src.synthesis.assembler import assemble_article


def _make_claim_graph(clusters, unique_details=None):
    """Build a minimal claim_graph viz dict for testing."""
    return {
        "corroborated": clusters,
        "unique_details": unique_details or [],
        "chunk_count": sum(len(c["sources"]) for c in clusters),
        "edge_count": 0,
    }


def test_assemble_basic_ordering():
    graph = _make_claim_graph([
        {
            "representative_text": "The president signed the bill into law.",
            "source_count": 3,
            "source_names": ["AP", "Reuters", "BBC"],
            "avg_similarity": 0.9,
            "sources": [
                {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "President signed the bill Tuesday."},
                {"source_name": "Reuters", "source_slug": "reuters", "source_bias": "center", "text": "The legislation was signed into law."},
                {"source_name": "BBC", "source_slug": "bbc", "source_bias": "center", "text": "Bill signed by president on Tuesday."},
            ],
        },
        {
            "representative_text": "Opposition lawmakers criticized the move.",
            "source_count": 2,
            "source_names": ["NYT", "Fox News"],
            "avg_similarity": 0.85,
            "sources": [
                {"source_name": "NYT", "source_slug": "nyt", "source_bias": "lean-left", "text": "Democrats in the Senate objected."},
                {"source_name": "Fox News", "source_slug": "foxnews", "source_bias": "lean-right", "text": "Republican leaders praised the decision."},
            ],
        },
    ])

    ordering = {
        "headline": "President Signs Bill Into Law",
        "ordering": [
            {"cluster": 0, "transition": ""},
            {"cluster": 1, "transition": "The decision drew mixed reactions."},
        ],
    }

    article = assemble_article(graph, ordering)

    # Lead cluster has no transition
    assert article.startswith("President signed the bill Tuesday.")
    # Attribution present
    assert "(AP)" in article
    # Transition present before second cluster
    assert "The decision drew mixed reactions." in article
    # Second cluster's passage present
    assert "Democrats in the Senate objected." in article
    assert "(NYT)" in article


def test_assemble_with_unique_details():
    graph = _make_claim_graph(
        clusters=[{
            "representative_text": "Markets fell sharply.",
            "source_count": 2,
            "source_names": ["AP", "Reuters"],
            "avg_similarity": 0.88,
            "sources": [
                {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "The S&P 500 dropped 3%."},
                {"source_name": "Reuters", "source_slug": "reuters", "source_bias": "center", "text": "Markets plunged on the news."},
            ],
        }],
        unique_details=[
            {"source_name": "The Intercept", "source_slug": "intercept", "source_bias": "left", "text": "Internal documents show the policy was drafted months ago."},
        ],
    )

    ordering = {
        "headline": "Markets Drop",
        "ordering": [{"cluster": 0, "transition": ""}],
    }

    article = assemble_article(graph, ordering)

    assert "The S&P 500 dropped 3%." in article
    assert "Internal documents show" in article
    assert "The Intercept" in article


def test_assemble_skips_invalid_cluster_index():
    graph = _make_claim_graph([{
        "representative_text": "Something happened.",
        "source_count": 2,
        "source_names": ["AP", "BBC"],
        "avg_similarity": 0.9,
        "sources": [
            {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "An event occurred."},
        ],
    }])

    ordering = {
        "headline": "Test",
        "ordering": [
            {"cluster": 0, "transition": ""},
            {"cluster": 99, "transition": "This cluster does not exist."},
        ],
    }

    article = assemble_article(graph, ordering)
    assert "An event occurred." in article
    # Invalid cluster 99 is silently skipped
    assert "This cluster does not exist." not in article


def test_assemble_picks_best_source_per_cluster():
    """The representative source (first in list) is used as the passage."""
    graph = _make_claim_graph([{
        "representative_text": "Lead text.",
        "source_count": 2,
        "source_names": ["AP", "Fox News"],
        "avg_similarity": 0.9,
        "sources": [
            {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "The definitive AP version of events."},
            {"source_name": "Fox News", "source_slug": "foxnews", "source_bias": "lean-right", "text": "Fox take on it."},
        ],
    }])

    ordering = {
        "headline": "Test",
        "ordering": [{"cluster": 0, "transition": ""}],
    }

    article = assemble_article(graph, ordering)
    # First source in list (closest to centroid) is the passage used
    assert "The definitive AP version" in article
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/cmur/projects/dorothy && source .venv/bin/activate && python -m pytest tests/test_article_assembler.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.synthesis.assembler'`

- [ ] **Step 3: Implement assembler**

Create `src/synthesis/assembler.py`:

```python
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
        # Use the first source's text (closest to centroid per graph builder)
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

    # Unique details section
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/cmur/projects/dorothy && source .venv/bin/activate && python -m pytest tests/test_article_assembler.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/synthesis/assembler.py tests/test_article_assembler.py
git commit -m "add extractive article assembler with tests

Assembles articles from claim graph clusters in LLM-specified order.
Each passage is attributed to its source with corroboration noted.
Unique details appended as a separate section."
```

---

### Task 4: Rewrite StorySummarizer

**Files:**
- Modify: `src/synthesis/summarizer.py`

- [ ] **Step 1: Replace system prompts and synthesize() method**

Remove all 6 system prompt constants (`ARTICLE_SYSTEM_PROMPT`, `ANALYSIS_SYSTEM_PROMPT`, and the sports/tech variants).

Replace with a single ordering prompt:

```python
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
```

Rewrite `synthesize()` to:

```python
    def synthesize(
        self, story: Story, claim_graph: ClaimGraph,
    ) -> Optional[SynthesizedStory]:
        """Synthesize a story using extractive assembly.

        1. Check body text threshold (3+ articles with body from 2+ sources)
        2. Build ordering prompt from claim graph clusters
        3. LLM orders clusters + writes headline + transitions
        4. Assemble article from claim graph + ordering
        """
        if not claim_graph or not claim_graph.corroborated:
            logger.info("skipping_no_corroborated_facts", story_id=story.id)
            return None

        # Body text threshold check
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

        # Build ordering prompt from cluster representative texts
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
            + "\n\nReturn JSON with \"headline\" and \"ordering\" keys."
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

            # Assemble article
            viz_dict = claim_graph.to_viz_dict()
            article = assemble_article(viz_dict, ordering)

            if not article or len(article.split()) < 20:
                logger.warning("degenerate_article", story_id=story.id)
                return None

            # Build metadata (reuse existing helpers)
            is_sports = self._story_column(story) == "sports"
            sources_used = list(set(
                a.get("source_slug", "") for a in story.articles
            ))
            similarity_edges = self._compute_similarity_edges(story.articles)
            article_refs = self._build_article_refs(story.articles)
            hero_url, hero_src = self._pick_hero_image(story.articles, is_sports)
            article_urls = sorted(
                str(a.get("url", "")) for a in story.articles if a.get("url")
            )
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
```

Add the import at the top of the file:

```python
from src.synthesis.assembler import assemble_article
```

- [ ] **Step 2: Remove dead code**

Delete these methods/functions that are no longer called:
- `_build_prompt()` (was used to build the full source text for generative prompt)
- `synthesize_stories()` (batch wrapper, not used by pipeline)
- The `StoryTiming` dataclass and `compute_story_timing()` stay — they're still used.
- `_story_column()`, `_compute_similarity_edges()`, `_pick_hero_image()`, `_build_article_refs()` all stay — used by the new `synthesize()`.

Remove unused imports: `defaultdict`, `np`, `cosine_distances` (if `_compute_similarity_edges` still uses them, keep them).

- [ ] **Step 3: Commit**

```bash
git add src/synthesis/summarizer.py
git commit -m "rewrite synthesizer for extractive pipeline

Single LLM call for ordering + headline + transitions (~200 tokens).
Article assembled programmatically from claim graph passages.
Removes all generative system prompts and analysis pass."
```

---

### Task 5: Update Pipeline

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Simplify _write_story and run_synthesis**

The `_write_story` function changes to pass `claim_graph` directly to `synthesize()` instead of as an optional:

```python
def _write_story(
    summarizer: StorySummarizer,
    story,
    edition: int,
    graph_builder: ClaimGraphBuilder,
):
    """Build claim graph and synthesize a single story."""
    try:
        claim_graph = graph_builder.build(story)
    except Exception as e:
        logger.warning("claim_graph_failed", story_id=story.id, error=str(e))
        return None, story

    synthesized = summarizer.synthesize(story, claim_graph=claim_graph)
    if not synthesized:
        return None, story

    synthesized.edition = edition
    return synthesized, story
```

In `run_synthesis()`, make `graph_builder` required (not Optional):

```python
def run_synthesis(
    os_client: OpenSearchClient,
    llm_client: LLMClient,
    column: str,
    edition: int = 1,
    limit: Optional[int] = None,
    graph_builder: ClaimGraphBuilder = None,
) -> int:
```

Remove the `reviewer` parameter entirely.

The synthesis loop stays the same (sequential write per story), but without the reviewer call.

- [ ] **Step 2: Make claim graph builder required in run_pipeline_cycle**

In `run_pipeline_cycle()`, the claim graph builder is no longer conditional. Remove the `if config.claim_graph.enabled:` check — it's always built:

```python
    graph_builder = ClaimGraphBuilder(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        similarity_threshold=config.claim_graph.similarity_threshold,
        min_sources_corroborated=config.claim_graph.min_sources_corroborated,
        embedding_concurrency=config.claim_graph.embedding_concurrency,
        min_chunk_chars=config.claim_graph.min_chunk_chars,
        max_chunk_chars=config.claim_graph.max_chunk_chars,
    )
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "update pipeline for extractive synthesis

Claim graph builder is now required, not optional. Remove reviewer
from synthesis loop. Simplify _write_story."
```

---

### Task 6: Update OpenSearch Storage

**Files:**
- Modify: `src/storage/opensearch.py`

- [ ] **Step 1: Remove quality_scores and review_improvements from store methods**

In `store_synthesis()` (line ~548-549), remove:
```python
            "quality_scores": synthesis.get("quality_scores"),
            "review_improvements": synthesis.get("review_improvements"),
```

In `bulk_store_syntheses()` (line ~598-599), remove the same two lines.

Keep `analysis` in the doc as an empty string for backward compatibility with old stories that have it. Add a default:
```python
            "analysis": synthesis.get("analysis", ""),
```

Leave the `SYNTHESIS_MAPPING` unchanged — the mapping tolerates missing fields, and old stories may still have `quality_scores` and `analysis` populated.

- [ ] **Step 2: Commit**

```bash
git add src/storage/opensearch.py
git commit -m "remove quality_scores and review_improvements from synthesis storage

These fields are no longer produced by the extractive pipeline.
Keep analysis field for backward compat with old stories."
```

---

### Task 7: Update Story Page Template

**Files:**
- Modify: `src/web/templates/story.html`

- [ ] **Step 1: Restructure tabs to Article / Evidence / Sources**

Replace the tab structure. The key changes:
- Remove the Analysis tab (radio input, label, panel)
- Rename Claims tab to Evidence
- Keep Article and Sources tabs

```html
    <div class="story-tabs">
        <input type="radio" name="story-tab" id="tab-article" class="tab-radio" checked>
        {% if story.claim_graph and story.claim_graph.corroborated %}
        <input type="radio" name="story-tab" id="tab-evidence" class="tab-radio">
        {% endif %}
        <input type="radio" name="story-tab" id="tab-sources" class="tab-radio">

        <nav class="tab-bar">
            <label for="tab-article" class="tab-label">Article</label>
            {% if story.claim_graph and story.claim_graph.corroborated %}
            <label for="tab-evidence" class="tab-label">Evidence</label>
            {% endif %}
            <label for="tab-sources" class="tab-label">Sources</label>
        </nav>
```

- [ ] **Step 2: Replace the Analysis panel with Evidence panel**

Remove the entire `panel-analysis` div (the analysis text + bias charts).

The Evidence panel replaces the old Claims panel:

```html
        {% if story.claim_graph and story.claim_graph.corroborated %}
        <div class="tab-panel" id="panel-evidence">
            <section class="claim-graph-section">
                <h3>Fact Corroboration</h3>
                <p class="claim-graph-caption">Which sources independently confirm the same facts. Hover a claim to see its sources, or a source to see what it corroborates.</p>
                <div id="claim-graph"></div>
                <script id="claim-graph-data" type="application/json">{{ story.claim_graph|tojson }}</script>
            </section>

            {% if story.bias_coverage %}
            <section class="bias-section">
                {% if story.column == 'sports' %}
                <h3>Coverage by Region</h3>
                <div class="bias-chart">
                    {% for region in ['us', 'canada', 'mexico', 'uk', 'australia', 'india', 'japan', 'korea', 'international'] %}
                        {% if region in story.bias_coverage %}
                        <div class="bias-bar">
                            <span class="bias-label" style="color: {{ region_colors.get(region, '#888') }}">{{ region_labels.get(region, region|title) }}</span>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: {{ (story.bias_coverage[region] / story.article_count * 100)|round }}%; background: {{ region_colors.get(region, '#888') }}"></div>
                            </div>
                            <span class="bias-count">{{ story.bias_coverage[region] }}</span>
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>
                {% elif story.column == 'tech' %}
                <h3>Coverage by Perspective</h3>
                <div class="bias-chart">
                    {% for perspective in ['consumer', 'enterprise', 'academic', 'culture'] %}
                        {% if perspective in story.bias_coverage %}
                        <div class="bias-bar">
                            <span class="bias-label" style="color: {{ perspective_colors.get(perspective, '#888') }}">{{ perspective_labels.get(perspective, perspective|title) }}</span>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: {{ (story.bias_coverage[perspective] / story.article_count * 100)|round }}%; background: {{ perspective_colors.get(perspective, '#888') }}"></div>
                            </div>
                            <span class="bias-count">{{ story.bias_coverage[perspective] }}</span>
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>
                {% else %}
                <h3>Coverage by Perspective</h3>
                <div class="bias-chart">
                    {% for bias in ['left', 'lean-left', 'center', 'lean-right', 'right'] %}
                        {% if bias in story.bias_coverage %}
                        <div class="bias-bar">
                            <span class="bias-label" style="color: {{ bias_colors[bias] }}">{{ bias|title }}</span>
                            <div class="bar-container">
                                <div class="bar-fill" style="width: {{ (story.bias_coverage[bias] / story.article_count * 100)|round }}%; background: {{ bias_colors[bias] }}"></div>
                            </div>
                            <span class="bias-count">{{ story.bias_coverage[bias] }}</span>
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>
                {% endif %}
            </section>
            {% endif %}
        </div>
        {% endif %}
```

- [ ] **Step 3: Update CSS tab selectors**

In `src/web/static/style.css`, update the tab selectors. Replace the old `tab-analysis` and `tab-claims` selectors:

```css
/* Active tab highlighting via :checked sibling selectors */
#tab-article:checked ~ .tab-bar label[for="tab-article"],
#tab-evidence:checked ~ .tab-bar label[for="tab-evidence"],
#tab-sources:checked ~ .tab-bar label[for="tab-sources"] {
    color: var(--text-primary);
    border-bottom-color: var(--text-primary);
}

/* Tab panel visibility */
.tab-panel {
    display: none;
}

#tab-article:checked ~ #panel-article,
#tab-evidence:checked ~ #panel-evidence,
#tab-sources:checked ~ #panel-sources {
    display: block;
}
```

- [ ] **Step 4: Update script includes at bottom of story.html**

Replace the script block:

```html
{% if story.similarity_edges or (story.claim_graph and story.claim_graph.corroborated) %}
<script src="https://d3js.org/d3.v7.min.js"></script>
{% endif %}
{% if story.similarity_edges %}
<script src="/static/{{ assets['similarity-web.js'] if assets is defined and 'similarity-web.js' in assets else 'similarity-web.js' }}"></script>
{% endif %}
{% if story.claim_graph and story.claim_graph.corroborated %}
<script src="/static/{{ assets['claim-graph.js'] if assets is defined and 'claim-graph.js' in assets else 'claim-graph.js' }}"></script>
{% endif %}
```

- [ ] **Step 5: Commit**

```bash
git add src/web/templates/story.html src/web/static/style.css
git commit -m "restructure story page: Article / Evidence / Sources

Remove Analysis tab. Rename Claims to Evidence. Move bias coverage
charts into Evidence tab alongside claim graph visualization."
```

---

### Task 8: End-to-End Test

**Files:**
- No new files. Manual verification.

- [ ] **Step 1: Run a single-story synthesis test**

```bash
source .venv/bin/activate && OPENSEARCH_HOST=100.64.0.8 python3 -c "
from src.storage import OpenSearchClient
from src.synthesis import LLMClient, StorySummarizer
from src.clustering import StoryGrouper
from src.claim_graph import ClaimGraphBuilder
from src.config import config

c = OpenSearchClient(host='100.64.0.8')
grouper = StoryGrouper(c, min_cluster_size=3, min_samples=2)
stories = grouper.get_stories_for_column('politics', size=500)
multi = [s for s in stories if s.source_count >= 2]
story = multi[0]

print('Story: %s' % story.headline[:80])
print('Articles: %d, Sources: %d' % (len(story.articles), story.source_count))

builder = ClaimGraphBuilder(
    base_url=config.embedding.base_url,
    model=config.embedding.model,
    similarity_threshold=config.claim_graph.similarity_threshold,
    min_sources_corroborated=config.claim_graph.min_sources_corroborated,
    embedding_concurrency=config.claim_graph.embedding_concurrency,
    min_chunk_chars=config.claim_graph.min_chunk_chars,
    max_chunk_chars=config.claim_graph.max_chunk_chars,
)
graph = builder.build(story)
print('Claim graph: %d corroborated, %d unique' % (len(graph.corroborated), len(graph.unique_details)))

llm = LLMClient(
    base_url=config.llm.base_url,
    model=config.llm.model,
    temperature=config.llm.temperature,
    max_tokens=config.llm.max_tokens,
    context_length=config.llm.context_length,
)
summarizer = StorySummarizer(llm)
import time
t0 = time.time()
result = summarizer.synthesize(story, claim_graph=graph)
elapsed = time.time() - t0

if result:
    print('Headline: %s' % result.generated_headline)
    print('Article: %d chars' % len(result.article))
    print('Time: %.1fs' % elapsed)
    print()
    print(result.article)
else:
    print('SYNTHESIS FAILED')
"
```

Expected: Article assembled from source passages in ~5-10 seconds. No hallucinated names. Each paragraph attributed to a source.

- [ ] **Step 2: Run the full pipeline**

```bash
source .venv/bin/activate && OPENSEARCH_HOST=100.64.0.8 python -m scripts.run_pipeline --once --stories 5 --publish
```

Expected: Completes in a few minutes. Stories rendered and deployed.

- [ ] **Step 3: Verify the deployed site**

Open https://dorothy.cmm.sh and check:
- Front page shows stories with headlines
- Story pages have Article / Evidence / Sources tabs
- Article tab shows attributed passages
- Evidence tab shows claim graph D3 visualization
- Sources tab shows original article links

- [ ] **Step 4: Commit any fixes**

If any adjustments were needed during testing, commit them:

```bash
git add -A
git commit -m "fix: adjustments from end-to-end testing"
```
