# Dorothy Clustering Fixes

## Problems Identified

### 1. Column Bleeding (HIGH PRIORITY)
Sports articles cluster with politics, tech bleeds into money, etc.

**Root Cause:** `knn_search()` in `src/storage/opensearch.py` searches the entire index without filtering by column. The column filter is applied *after* the k-NN results come back, but by then you've already lost relevant same-column matches.

**Fix:** Add a pre-filter to the k-NN query so OpenSearch only considers articles from the same column.

### 2. Short Synthesis Output (MEDIUM PRIORITY)
LLM summaries are shorter than desired.

**Root Cause:** Prompt says "2-3 paragraphs" - model is following instructions. Also depends on which model is running in LMStudio.

**Fix:** Update prompt in `src/synthesis/summarizer.py` to request longer output with more specific guidance.

### 3. Duplicate/Near-Duplicate Articles (LOW PRIORITY)
Syndicated content (AP, Reuters) appears multiple times from different outlets with different URLs.

**Root Cause:** Deduplication is URL-based only. Same story syndicated across outlets = multiple articles.

**Fix:** Add headline similarity check before indexing, or let clustering handle it (it should, if column filtering works).

---

## Implementation Plan

### Task 1: Fix k-NN Column Filtering

**File:** `src/storage/opensearch.py`

**Change:** Modify `knn_search()` to accept an optional `column` parameter and add a filter to the query.

**Before:**
```python
def knn_search(
    self,
    embedding: list[float],
    k: int = 10,
    index_name: Optional[str] = None,
) -> list[dict]:
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": k,
                }
            }
        },
    }
```

**After:**
```python
def knn_search(
    self,
    embedding: list[float],
    k: int = 10,
    index_name: Optional[str] = None,
    column: Optional[str] = None,
) -> list[dict]:
    knn_query = {
        "vector": embedding,
        "k": k,
    }
    
    if column:
        knn_query["filter"] = {"term": {"column": column}}
    
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": knn_query
            }
        },
    }
```

**File:** `src/clustering/story_grouper.py`

**Change:** Pass `column` through to `knn_search()` in `find_similar_articles()`.

```python
def find_similar_articles(
    self,
    embedding: list[float],
    index_name: Optional[str] = None,
    column: Optional[str] = None,  # Add this parameter
) -> list[dict]:
    results = self.os_client.knn_search(
        embedding=embedding,
        k=self.k_neighbors,
        index_name=index_name,
        column=column,  # Pass it through
    )
    return [r for r in results if r.get("score", 0) >= self.similarity_threshold]
```

**Also update** the call site in `group_articles()` to pass `column` to `find_similar_articles()`.

---

### Task 2: Improve Synthesis Length

**File:** `src/synthesis/summarizer.py`

**Change:** Update `SYNTHESIS_PROMPT_TEMPLATE` to request more detailed output.

**Current:**
```
2. A balanced 2-3 paragraph summary that:
```

**Proposed:**
```
2. A comprehensive 4-6 paragraph summary (400-600 words) that:
   - Opens with the core facts all sources agree on (who, what, when, where)
   - Explores the context and background of the story
   - Details how different sources frame or emphasize different aspects
   - Notes any disputed facts or conflicting claims with attribution
   - Closes with the current status or next expected developments
```

---

### Task 3: (Optional) Headline-Based Deduplication

**File:** `src/storage/opensearch.py`

**Change:** Add a method to check for similar headlines using fuzzy matching or embedding similarity.

```python
def find_similar_headlines(
    self,
    headline: str,
    threshold: float = 0.9,
    index_name: Optional[str] = None,
) -> list[dict]:
    """Find articles with very similar headlines (potential duplicates)."""
    # Option A: Use OpenSearch's more_like_this query
    # Option B: Embed the headline and do k-NN with high threshold
    pass
```

Then call this in the fetch pipeline before indexing.

**Note:** This may not be necessary if Task 1 fixes clustering quality. Evaluate after Task 1.

---

## Testing

After implementing Task 1:
1. Clear existing syntheses: `python -c "from src.storage import OpenSearchClient; OpenSearchClient().clear_syntheses()"`
2. Run synthesis for politics only: `python -m scripts.run_synthesis --column politics --limit 5`
3. Check if sports/tech articles are bleeding in
4. Repeat for other columns

After implementing Task 2:
1. Re-run synthesis and compare output length
2. Adjust word count targets if needed

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/storage/opensearch.py` | Add `column` param to `knn_search()` |
| `src/clustering/story_grouper.py` | Pass `column` through to k-NN calls |
| `src/synthesis/summarizer.py` | Update prompt for longer output |

---

## Success Criteria

- [ ] Politics syntheses contain only politics articles
- [ ] Sports syntheses contain only sports articles  
- [ ] No cross-column contamination in story clusters
- [ ] Synthesis output is 400-600 words instead of 100-200
