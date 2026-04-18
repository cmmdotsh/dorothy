# Extractive Synthesis Pipeline

**Date:** 2026-04-18
**Status:** Approved

## Problem

Dorothy's generative synthesis pipeline asks local LLMs to write full news articles from source material. Even with anti-hallucination prompts and a reviewer pass, models consistently fabricate expert names, quotes, and facts not present in sources. The reviewer was further undermined by a bug that passed zero source material to the review model. Local models (3B-active MoE and 31B dense) lack the capacity to reliably follow "only use source material" constraints during open-ended generation.

## Solution

Replace generative synthesis with extractive synthesis. The claim graph already identifies corroborated facts as text chunks with source attribution. The model's role changes from "write an article" to "order these facts and write transitions between them." The article becomes 90% real source text with 10% structural connective tissue.

## Architecture

### Pipeline Flow (per story)

```
1. StoryGrouper clusters articles (unchanged)
2. Filter: skip if < 3 articles with body text from 2+ sources
3. ClaimGraphBuilder.build(story) -> ClaimGraph
4. Filter: skip if 0 corroborated clusters
5. LLM call: order clusters + write headline + transitions (~200 tokens)
6. Assemble article programmatically from claim graph + LLM ordering
7. Store in OpenSearch with claim_graph data
```

Step 5 is the only LLM call. Total LLM output per story: ~100-200 tokens.

### The LLM Prompt (Step 5)

The model receives only the representative text from each corroborated cluster (not full passages). It returns:

```json
{
  "headline": "Iran Reopens Strait of Hormuz But U.S. Blockade Remains",
  "ordering": [
    {"cluster": 0, "transition": ""},
    {"cluster": 2, "transition": "The announcement drew immediate international response."},
    {"cluster": 1, "transition": "However, the situation remains contested."},
    {"cluster": 3, "transition": "Financial markets reacted swiftly."}
  ]
}
```

- First cluster gets no transition (it's the lead)
- Transitions are structural, not factual ("Meanwhile...", "The situation was further complicated by...")
- The model never sees or rewrites source text

### Article Assembly (Step 6)

Code assembles the article by iterating the LLM's ordering:

1. For each cluster in order:
   - Insert transition sentence (if any)
   - Insert the representative passage from the cluster
   - Append source attribution
2. Append unique details section (grouped by source, with single-source attribution)

The article is stored as assembled text in the `article` field for backward compatibility with templates and static site generation.

### Body Text Requirement

Stories are skipped if they don't meet both conditions:
- At least 3 articles with non-empty `body` field (extracted full text)
- Those articles come from at least 2 unique sources

The claim graph needs real article passages to build meaningful chunks. Stories with only headline + summary produce empty or degenerate graphs and are not synthesized.

## Data Model Changes

### SynthesizedStory

| Field | Change |
|-------|--------|
| `article` | Stays. Contains assembled extractive article |
| `analysis` | Remove. Claim graph viz replaces prose analysis |
| `generated_headline` | Stays. LLM-generated from cluster ordering call |
| `claim_graph` | Required (was optional). Backbone of the article |
| `quality_scores` | Remove. No reviewer needed |
| `review_improvements` | Remove. No reviewer needed |
| `similarity_edges` | Keep. Powers source similarity web |

### OpenSearch Mapping

- Drop `quality_scores` and `review_improvements` from mapping
- `claim_graph` becomes a required field (keep `"enabled": false` for passthrough storage)
- `analysis` field can remain in mapping for backward compat but will be empty on new stories

## Components Removed

| Component | Reason |
|-----------|--------|
| `src/synthesis/reviewer.py` | No reviewer needed. Articles are extracted, not generated |
| `src/synthesis/ollama_client.py` | Legacy. Reviewer was last consumer |
| `ReviewerSettings` in config | No reviewer |
| `ARTICLE_SYSTEM_PROMPT` (all 3 variants) | No generative article writing |
| `ANALYSIS_SYSTEM_PROMPT` (all 3 variants) | Claim graph viz replaces prose analysis |
| Pass 2 analysis LLM call | Replaced by claim graph visualization |
| `synthesis_concurrency` setting | Pipeline is fast enough without it |

## Components Modified

| Component | Change |
|-----------|--------|
| `src/synthesis/summarizer.py` | Rewrite. Two-pass generative flow becomes single ordering call + assembly |
| `src/config.py` | Remove `ReviewerSettings`. Simplify `LLMSettings` |
| `scripts/run_pipeline.py` | Remove reviewer setup, remove reviewer thread/queue code |

## Components Unchanged or More Important

| Component | Note |
|-----------|------|
| Claim graph builder | Becomes core of pipeline, not optional |
| Chunk embedder | Runs on every story |
| `to_viz_dict()` | Primary data format for article assembly and visualization |
| Body text extractor | Critical. Chunks need real article text |
| `claim-graph.js` | Main analytical visualization |
| `_compute_similarity_edges()` | Still powers source similarity web |

## Story Page Tabs

| Tab | Content |
|-----|---------|
| **Article** | Extractive article: ordered corroborated passages with attribution + transitions |
| **Evidence** | Claim graph D3 visualization. Corroborated clusters as central nodes, sources as bias-colored satellites. Unique details as isolated nodes |
| **Sources** | Original article links with bias pills + source similarity web |

The "Analysis" tab with LLM-generated prose is removed. The claim graph visualization IS the analysis -- it shows which sources corroborate which facts, and what's reported exclusively by one outlet. The reader draws their own conclusions from the structure.

## Unique Details Handling

Unique details (facts reported by only one source) appear in a clearly labeled sidebar/section below the main article. They are not mixed into the corroborated article body. Each unique detail shows its single-source attribution so readers can assess credibility themselves.

## Performance

| Metric | Current (Generative) | New (Extractive) |
|--------|---------------------|-------------------|
| LLM tokens per story | ~8000 (article + analysis + review) | ~200 (ordering + headline + transitions) |
| LLM calls per story | 3 (write + analyze + review) | 1 (ordering) |
| Time per story (qwen MoE) | ~90s | ~5-10s |
| Time per 100-story edition | ~2.5 hours | ~20 minutes |
| Hallucination risk | High (model invents names, quotes, facts) | Near zero (model only writes structural transitions) |

## Edition Cadence

With ~20 minute generation time, Dorothy can run morning and evening editions comfortably. The pipeline schedule changes from hourly to twice daily, reducing compute and producing cleaner editions with more accumulated articles to cluster from.
