# Daily Window + Event Threads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Dorothy a daily paper (72h clustering window, honest dateline) and add recurrence-born event threads that track developing stories across months, surfaced via a "Developments" section.

**Architecture:** Part 1 threads a `since` window + per-source cap through the existing clustering path and clamps story dates. Part 2 adds a `dorothy-events` index, stores a summary embedding on each new synthesis, and matches new stories against threads (and recent threadless stories) via python-side cosine shortlist + small-LLM yes/no confirm. Matching failure never blocks publishing.

**Tech Stack:** Python 3.13, pydantic BaseSettings, opensearch-py, structlog, httpx, Jinja2. No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-16-daily-window-and-event-threads-design.md`. On conflict, the spec wins.
- Logging: structlog event-name-first (`logger.info("event_name", key=value)`); match each file's existing logger.
- Config: pydantic `BaseSettings` subclasses with `env_prefix`, registered on `DorothyConfig` (`src/config.py`).
- Tests: `.venv/bin/python -m pytest tests/ -q`. Unit tests MUST NOT require OpenSearch or an LLM — stub clients.
- Thread-matching similarity is computed **python-side** (cosine over fetched candidate embeddings), NOT via OpenSearch k-NN. Thread counts are small (<1000); boring wins.
- Embeddings: 1024-d, same model/client as articles (`src/embeddings/client.py` `EmbeddingClient.embed(texts: list[str]) -> list[list[float]]` — verify exact name before use and adapt).
- Timezones: all datetimes UTC-aware. Use each module's existing `utcnow`/`_utcnow` helper.
- Do NOT run formatters/linters. Commit after each task with the message given.

---

### Task 1: ClusteringSettings + recency window + per-source cap

**Files:**
- Modify: `src/config.py` (add `ClusteringSettings`, register on `DorothyConfig`)
- Modify: `src/storage/opensearch.py:275-310` (`search_articles`: accept `index_name: Optional[str | list[str]]`)
- Modify: `src/clustering/story_grouper.py:105-143,346-377` (constructor + `get_stories_for_column`)
- Modify: `scripts/run_pipeline.py:199-204` (wire config into `StoryGrouper`)
- Test: `tests/test_clustering_window.py` (new)

**Interfaces:**
- Produces: `config.clustering` with `window_hours: int = 72`, `max_per_source: int = 40`, `min_cluster_size: int = 3`, `min_samples: int = 2` (env prefix `CLUSTERING_`).
- Produces: `StoryGrouper.__init__(..., window_hours: int = 72, max_per_source: int = 40)`;
  `get_stories_for_column` computes `since`, queries current + previous monthly index when the window crosses the month boundary, and applies the per-source cap.
- Produces: `StoryGrouper._cap_per_source(articles: list[dict], max_per_source: int) -> list[dict]` (pure, testable).

- [ ] **Step 1: Write failing tests** (`tests/test_clustering_window.py`)

```python
"""Recency window + per-source cap for clustering input."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.clustering.story_grouper import StoryGrouper


def _article(source, pub_date, i):
    return {"id": f"a{i}", "source_slug": source, "pub_date": pub_date.isoformat(),
            "embedding": [0.1] * 8, "column": "lifestyle"}


def test_cap_per_source_keeps_newest():
    now = datetime.now(timezone.utc)
    arts = [_article("npr-lifekit", now - timedelta(hours=h), h) for h in range(10)]
    arts += [_article("fox-lifestyle", now - timedelta(hours=h), 100 + h) for h in range(3)]
    capped = StoryGrouper._cap_per_source(arts, max_per_source=5)
    npr = [a for a in capped if a["source_slug"] == "npr-lifekit"]
    assert len(npr) == 5
    # newest 5 kept (hours 0-4)
    assert {a["id"] for a in npr} == {"a0", "a1", "a2", "a3", "a4"}
    assert len([a for a in capped if a["source_slug"] == "fox-lifestyle"]) == 3


def test_get_stories_passes_since_window():
    client = MagicMock()
    client.search_articles.return_value = []
    client.get_current_index_name.return_value = "dorothy-articles-2026-08"
    g = StoryGrouper(client, window_hours=72)
    g.get_stories_for_column("politics", size=2000)
    kwargs = client.search_articles.call_args.kwargs
    assert kwargs["since"] is not None
    age = datetime.now(timezone.utc) - kwargs["since"]
    assert timedelta(hours=71) < age < timedelta(hours=73)


def test_month_boundary_queries_both_indices():
    client = MagicMock()
    client.search_articles.return_value = []
    client.get_current_index_name.return_value = "dorothy-articles-2026-08"
    g = StoryGrouper(client, window_hours=72)
    # 2026-08-02T00:00Z minus 72h crosses into July
    g.get_stories_for_column("politics", size=2000,
                             now=datetime(2026, 8, 2, tzinfo=timezone.utc))
    kwargs = client.search_articles.call_args.kwargs
    assert kwargs["index_name"] == ["dorothy-articles-2026-07", "dorothy-articles-2026-08"]


def test_mid_month_queries_single_index():
    client = MagicMock()
    client.search_articles.return_value = []
    client.get_current_index_name.return_value = "dorothy-articles-2026-08"
    g = StoryGrouper(client, window_hours=72)
    g.get_stories_for_column("politics", size=2000,
                             now=datetime(2026, 8, 16, tzinfo=timezone.utc))
    kwargs = client.search_articles.call_args.kwargs
    assert kwargs["index_name"] is None  # default = current index
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_clustering_window.py -q`
Expected: FAIL (`_cap_per_source` not defined; unexpected kwargs).

- [ ] **Step 3: Implement**

`src/config.py` — after `ClaimGraphSettings`:

```python
class ClusteringSettings(BaseSettings):
    """Story clustering settings."""
    # Only articles published within this window enter clustering (daily paper).
    window_hours: int = 72
    # Per-cycle cap of articles per source per column (archive-dump guard).
    max_per_source: int = 40
    min_cluster_size: int = 3
    min_samples: int = 2

    class Config:
        env_prefix = "CLUSTERING_"
```

Register in `DorothyConfig.__init__` alongside the other settings objects
(follow the existing pattern, attribute name `clustering`).

`src/storage/opensearch.py` `search_articles`: change signature to
`index_name: Optional[str | list[str]] = None`; when a list is passed, join:
`index = ",".join(index_name) if isinstance(index_name, list) else index_name`
(OpenSearch accepts comma-separated indices) — resolve `None` to
`self.get_current_index_name()` first.

`src/clustering/story_grouper.py`:
- Constructor: add `window_hours: int = 72, max_per_source: int = 40`, store on self. Keep deprecated-param warnings untouched.
- Add:

```python
    @staticmethod
    def _cap_per_source(articles: list[dict], max_per_source: int) -> list[dict]:
        """Keep at most max_per_source newest articles per source_slug."""
        by_source: dict[str, list[dict]] = {}
        for a in articles:
            by_source.setdefault(a.get("source_slug", ""), []).append(a)
        kept: list[dict] = []
        dropped = 0
        for source, group in by_source.items():
            group.sort(key=lambda a: a.get("pub_date") or "", reverse=True)
            kept.extend(group[:max_per_source])
            dropped += max(0, len(group) - max_per_source)
        if dropped:
            logger.info("per_source_cap_applied", dropped=dropped,
                        max_per_source=max_per_source)
        return kept
```

- `get_stories_for_column(self, column, size=100, index_name=None, now=None)`:

```python
        if now is None:
            now = utcnow()  # add module-level: from datetime import datetime, timezone; def utcnow(): return datetime.now(timezone.utc) — reuse an existing helper if the module has one
        since = now - timedelta(hours=self.window_hours)
        if index_name is None and since.strftime("%Y-%m") != now.strftime("%Y-%m"):
            index_name = [
                f"dorothy-articles-{since.strftime('%Y-%m')}",
                self.os_client.get_current_index_name(),
            ]
        articles = self.os_client.search_articles(
            column=column, size=size, since=since, index_name=index_name,
        )
        articles = self._cap_per_source(articles, self.max_per_source)
```

(then the existing embedding filter + `group_articles` call, unchanged).
Note: when passing a list of index names, a missing previous-month index must not
error — add `ignore_unavailable=True` to the `client.search` call in
`search_articles`.

`scripts/run_pipeline.py:199` (and the extraction-pass twin at :134):

```python
        grouper = StoryGrouper(
            os_client,
            min_cluster_size=config.clustering.min_cluster_size,
            min_samples=config.clustering.min_samples,
            window_hours=config.clustering.window_hours,
            max_per_source=config.clustering.max_per_source,
        )
```

(`config` is already imported in the script; verify.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_clustering_window.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py src/storage/opensearch.py src/clustering/story_grouper.py scripts/run_pipeline.py tests/test_clustering_window.py
git commit -m "feat: 72h clustering window + per-source cap (daily paper)"
```

---

### Task 2: Date hygiene — window-clamped story timing

**Files:**
- Modify: `src/synthesis/summarizer.py:101-149` (`compute_story_timing`)
- Test: `tests/test_story_timing.py` (new)

**Interfaces:**
- Produces: `compute_story_timing(articles, now=None, window_hours: int = 72) -> StoryTiming` — every parsed pub_date is clamped into `[now - window_hours, now]` before median/first/last/hotness are computed. Raw `pub_date` on article docs is untouched (data preserved).
- Consumes: callers pass `window_hours=config.clustering.window_hours` (wire the call in `summarizer.synthesize` / wherever `compute_story_timing` is invoked — grep callers first).

- [ ] **Step 1: Write failing tests** (`tests/test_story_timing.py`)

```python
from datetime import datetime, timedelta, timezone

from src.synthesis.summarizer import compute_story_timing

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _art(pub, bias="center"):
    return {"pub_date": pub.isoformat(), "column": "politics", "source_bias": bias}


def test_ancient_date_clamps_to_window_floor():
    arts = [_art(datetime(2022, 2, 8, tzinfo=timezone.utc)),
            _art(NOW - timedelta(hours=2)), _art(NOW - timedelta(hours=3))]
    t = compute_story_timing(arts, now=NOW, window_hours=72)
    floor = (NOW - timedelta(hours=72)).isoformat()
    assert t.first_pub_date == floor          # 2022 date floored, not shown
    assert t.median_pub_date >= floor


def test_future_date_clamps_to_now():
    arts = [_art(NOW + timedelta(days=2)), _art(NOW - timedelta(hours=1)),
            _art(NOW - timedelta(hours=2))]
    t = compute_story_timing(arts, now=NOW, window_hours=72)
    assert t.last_pub_date == NOW.isoformat()
    # future date can no longer pin hotness via a 1.0h clamp on a "future median"
    assert t.hotness_score <= len(arts) * 2.0


def test_in_window_dates_unchanged():
    d1, d2, d3 = (NOW - timedelta(hours=h) for h in (5, 10, 20))
    t = compute_story_timing([_art(d1), _art(d2), _art(d3)], now=NOW, window_hours=72)
    assert t.first_pub_date == d3.isoformat()
    assert t.last_pub_date == d1.isoformat()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_story_timing.py -q`
Expected: FAIL (`window_hours` unexpected kwarg / clamping absent).

- [ ] **Step 3: Implement**

In `compute_story_timing`, after the tz-normalization inside the parse loop
(`summarizer.py:119-121`), clamp:

```python
        floor = now - timedelta(hours=window_hours)
        # inside the loop, after pd is tz-aware:
        pd = min(max(pd, floor), now)
```

(`timedelta` import exists or add it.) Signature gains
`window_hours: int = 72`. Grep for `compute_story_timing(` callers and pass
`window_hours=config.clustering.window_hours` where config is available;
default keeps other callers working.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_story_timing.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/synthesis/summarizer.py tests/test_story_timing.py
git commit -m "feat: clamp story dates to the publishing window"
```

---

### Task 3: Honest dateline — render-time freshness filter

**Files:**
- Modify: `scripts/render_static.py:277-290,319-331,345-356` (`get_stories_for_column`, `get_all_stories` untouched — story pages keep full history)
- Modify: `src/web/templates/front_page.html`, `src/web/templates/column.html` (quiet-day note)
- Test: `tests/test_render_freshness.py` (new)

**Interfaces:**
- Produces: `StaticRenderer.get_stories_for_column(column, limit=20, max_age_hours: int | None = 72)` — drops syntheses with `generated_at` older than `max_age_hours`; `None` disables (used by story-page path to keep archives reachable).
- Front/column templates render `{% if not stories %}<p class="quiet-day">A quiet day on the {{ column }} desk.</p>{% endif %}` (adapt copy/class to existing template style).

- [ ] **Step 1: Write failing test** (`tests/test_render_freshness.py`)

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from scripts.render_static import StaticRenderer  # verify class name at :250ish; adapt import if it differs

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _synth(story_id, age_hours):
    return {"story_id": story_id, "article_urls": [f"https://x/{story_id}"],
            "generated_headline": story_id,
            "generated_at": (NOW - timedelta(hours=age_hours)).isoformat()}


def test_stale_syntheses_filtered(monkeypatch):
    r = StaticRenderer.__new__(StaticRenderer)   # skip filesystem/Jinja setup
    r.os_client = MagicMock()
    r.os_client.get_syntheses.return_value = [_synth("fresh", 5), _synth("june", 1340)]
    monkeypatch.setattr("scripts.render_static._utcnow", lambda: NOW, raising=False)
    got = r.get_stories_for_column("politics", limit=20, max_age_hours=72)
    assert [s["story_id"] for s in got] == ["fresh"]


def test_filter_disabled_with_none():
    r = StaticRenderer.__new__(StaticRenderer)
    r.os_client = MagicMock()
    r.os_client.get_syntheses.return_value = [_synth("june", 1340)]
    got = r.get_stories_for_column("politics", limit=20, max_age_hours=None)
    assert len(got) == 1
```

(The implementer MUST check the real class name and `_dedup_stories` /
`_backfill_image_credit` flow at `render_static.py:277-281` and keep them; the
freshness filter is applied to the `get_syntheses` result before dedup. Add a
module-level `_utcnow()` helper if none exists so the test can patch time.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_render_freshness.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

    def get_stories_for_column(self, column: str, limit: int = 20,
                               max_age_hours: int | None = 72) -> list[dict]:
        stories = self.os_client.get_syntheses(column=column, limit=max(limit * 5, 50))
        if max_age_hours is not None:
            cutoff = _utcnow() - timedelta(hours=max_age_hours)
            stories = [s for s in stories
                       if s.get("generated_at") and
                       dateutil_parser.isoparse(s["generated_at"]) >= cutoff]
        stories = _dedup_stories(stories)[:limit]
        return [self._backfill_image_credit(s) for s in stories]
```

`render_story_pages` (`:364-371`) keeps full history: it calls
`get_syntheses` directly today — leave that path unfiltered. Front page and
column pages use the default `max_age_hours=72`.

Templates: add the empty-state block to `column.html` where stories iterate,
and to each column section of `front_page.html` (follow existing markup;
minimal styling, reuse an existing muted-text class if present).

- [ ] **Step 4: Run tests + render smoke**

Run: `.venv/bin/python -m pytest tests/test_render_freshness.py tests/ -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_static.py src/web/templates/front_page.html src/web/templates/column.html tests/test_render_freshness.py
git commit -m "feat: front/column pages only render syntheses fresh within 72h"
```

---

### Task 4: Events storage — index, model, CRUD

**Files:**
- Create: `src/events/__init__.py` (exports `EventStore`, `Event`)
- Create: `src/events/store.py`
- Modify: `src/storage/opensearch.py` (add `EVENTS_MAPPING`, `SYNTHESIS_MAPPING` gains `event_id` keyword + `summary_embedding` float array (plain `"type": "float"` list, NOT knn_vector — similarity is python-side); add `ensure_events_index()`, extend `bulk_store_syntheses` doc construction to pass through `event_id`/`summary_embedding` when present)
- Test: `tests/test_event_store.py` (new)

**Interfaces:**
- Produces (consumed by Tasks 5, 6, 7, 8):

```python
@dataclass
class Event:
    event_id: str                 # "event-" + 12 hex
    title: str
    summary: str
    summary_embedding: list[float]
    status: str                   # "active" | "dormant"
    chapters: list[dict]          # {story_id, generated_headline, generated_at, article_count}
    columns: list[str]
    first_seen: str               # isoformat
    last_seen: str

class EventStore:
    def __init__(self, os_client: OpenSearchClient): ...
    INDEX = "dorothy-events"
    def ensure_index(self) -> None: ...
    def create_event(self, title, summary, summary_embedding, chapters, columns) -> Event: ...
    def attach_chapter(self, event_id, chapter: dict, new_summary: str,
                       new_embedding: list[float], column: str) -> None:
        """Append chapter, update summary/embedding/last_seen/columns, set status=active."""
    def get_all_events(self) -> list[Event]: ...          # active + dormant
    def get_event(self, event_id) -> Optional[Event]: ...
    def mark_dormant_older_than(self, days: int = 14) -> int:
        """status=dormant where status=active and last_seen < now-days. Returns count."""
```

- `event_id = "event-" + hashlib.sha256(first_chapter_story_id.encode()).hexdigest()[:12]` — deterministic, idempotent re-runs.
- All writes `refresh=True` (thread counts are tiny; read-after-write matters within a cycle).

- [ ] **Step 1: Write failing tests** (`tests/test_event_store.py`)

```python
from unittest.mock import MagicMock

from src.events.store import Event, EventStore


def test_event_id_deterministic():
    client = MagicMock()
    store = EventStore(client)
    e1 = store.create_event("T", "S", [0.1], [{"story_id": "story-abc", "generated_headline": "h", "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 5}], ["politics"])
    e2 = store.create_event("T", "S", [0.1], [{"story_id": "story-abc", "generated_headline": "h", "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 5}], ["politics"])
    assert e1.event_id == e2.event_id
    assert e1.event_id.startswith("event-") and len(e1.event_id) == 18


def test_create_event_indexes_doc():
    client = MagicMock()
    store = EventStore(client)
    e = store.create_event("Iran strikes", "sum", [0.1] * 4,
                           [{"story_id": "story-1", "generated_headline": "h",
                             "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 5}],
                           ["politics"])
    call = client.client.index.call_args.kwargs
    assert call["index"] == "dorothy-events"
    assert call["id"] == e.event_id
    assert call["body"]["status"] == "active"
    assert call["body"]["chapters"][0]["story_id"] == "story-1"


def test_attach_chapter_updates_and_reactivates():
    client = MagicMock()
    store = EventStore(client)
    store.attach_chapter("event-abc", {"story_id": "story-2", "generated_headline": "h2",
                                       "generated_at": "2026-08-16T01:00:00+00:00", "article_count": 3},
                         new_summary="s2", new_embedding=[0.2] * 4, column="politics")
    call = client.client.update.call_args.kwargs
    assert call["id"] == "event-abc"
    script_or_doc = call["body"]
    assert "story-2" in str(script_or_doc)
    assert "active" in str(script_or_doc)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_event_store.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** `src/events/store.py` per the interface block. Use
`self.os.client.index(index=..., id=..., body=..., refresh=True)` and
`client.update` with a partial `doc` body (append chapters read-modify-write:
`get_event` → mutate → full re-index is acceptable and simpler than painless
scripts; keep it that way). `mark_dormant_older_than` uses `update_by_query`
with a range on `last_seen` and `term status=active`. `EVENTS_MAPPING` mirrors
the interface fields (`keyword` for ids/status/columns, `text` for
title/summary, `float` for embedding, `date` for seen fields, `object`
(enabled) for chapters). `ensure_events_index` follows `ensure_index`'s
create-if-missing pattern (`opensearch.py:184-196`).

- [ ] **Step 4: Run tests** — `pytest tests/test_event_store.py tests/ -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/events/ src/storage/opensearch.py tests/test_event_store.py
git commit -m "feat: dorothy-events index + EventStore CRUD"
```

---

### Task 5: Event matcher — shortlist + LLM confirm + recurrence birth

**Files:**
- Create: `src/events/matcher.py`
- Modify: `src/config.py` (add `EventSettings`: `enabled: bool = True`, `shortlist_threshold: float = 0.60`, `shortlist_k: int = 3`, `dormant_after_days: int = 14`, `threadless_window_days: int = 14`; env prefix `EVENTS_`)
- Test: `tests/test_event_matcher.py` (new)

**Interfaces:**
- Consumes: `EventStore` (Task 4), `LLMClient.generate(prompt, system_prompt=..., max_tokens=...) -> str` (`src/synthesis/llm_client.py:66`), embedding client (same one the pipeline builds — accept any object with the article-embedding call; take it as a constructor dep and call through a small `_embed(text) -> list[float]` wrapper the implementer adapts to the real client API).
- Produces (consumed by Task 6):

```python
class EventMatcher:
    def __init__(self, store: EventStore, llm_client, embedding_client, settings): ...
    def match_story(self, synthesis: dict) -> Optional[str]:
        """Returns event_id if the story attaches (or founds) a thread, else None.
        synthesis: a stored synthesis dict (story_id, column, generated_headline,
        summary, article_urls, generated_at, summary_embedding)."""
```

Matching order inside `match_story` (spec §2.2–2.3):
1. **Jaccard fast-path**: caller (Task 6) handles this — when the pipeline's
   existing `find_overlapping_synthesis` fires (`run_pipeline.py:247-274`) and
   the superseded synthesis has an `event_id`, inherit it without calling
   `match_story`. The matcher itself only does semantic matching.
2. **Thread shortlist**: cosine(new `summary_embedding`, each event's
   `summary_embedding`) over `store.get_all_events()`; keep top
   `shortlist_k` with score ≥ `shortlist_threshold`; for each, one LLM yes/no
   (prompt below); first `yes` → LLM-refresh the rolling summary
   (second prompt below) → `store.attach_chapter(...)` → return event_id.
3. **Recurrence birth**: same shortlist over *threadless* syntheses
   (`event_id` missing) from the last `threadless_window_days` days OR flagged
   `thread_candidate: true` (bootstrap, Task 8), excluding self and
   same-cluster duplicates (Jaccard(article_urls) > 0.3 → skip: that is the
   same story, not a development). On a confirmed `yes`:
   `store.create_event(...)` with the OLD story as chapter 1 and the new story
   as chapter 2; initial summary LLM-generated from both; tag both synthesis
   docs with the event_id (`os.client.update` on `dorothy-synthesis`).
4. Nothing confirmed → return None.

Confirm prompt (bias to "no", strict token):

```python
CONFIRM_SYSTEM = (
    "You decide whether a news story is a development of an ongoing event. "
    "Answer with exactly one word: yes or no. When unsure, answer no. "
    "Same topic is NOT enough - it must be the same specific ongoing event, "
    "conflict, case, or storyline."
)
CONFIRM_TEMPLATE = (
    "ONGOING EVENT:\n{thread_summary}\n\n"
    "NEW STORY ({date}):\n{headline}\n{summary}\n\n"
    "Is the new story a development of the ongoing event? yes or no:"
)
# accept only: response.strip().lower().split()[0] == "yes"
```

Summary-refresh prompt:

```python
SUMMARY_SYSTEM = (
    "You maintain a neutral running summary of an ongoing news event. "
    "Rewrite the summary to incorporate the new development. "
    "Maximum 120 words. No commentary, no markdown."
)
SUMMARY_TEMPLATE = (
    "CURRENT SUMMARY:\n{summary}\n\nNEW DEVELOPMENT ({date}):\n{headline}\n{story_summary}\n\n"
    "UPDATED SUMMARY:"
)
```

Every decision logs: `logger.info("event_match_candidate", story_id=..., event_id=..., score=..., verdict=...)`, `event_attached`, `event_born`, `event_match_none`.
Any exception inside `match_story` is caught, logged `event_match_failed`, returns None (spec: matching MUST NOT block publishing).

- [ ] **Step 1: Write failing tests** (`tests/test_event_matcher.py`) — stub LLM returns scripted "yes"/"no"; stub embedding returns fixed vectors; stub store returns fixed events. Cover: (a) high-cosine + LLM yes → attach called with refreshed summary; (b) high-cosine + LLM no → no attach; (c) below threshold → LLM never called; (d) recurrence birth from a threadless candidate: create_event called, chapters ordered old→new; (e) same-cluster Jaccard>0.3 candidate skipped; (f) LLM exception → returns None, no raise.

```python
import math
from unittest.mock import MagicMock

from src.events.matcher import EventMatcher


def _settings(**kw):
    s = MagicMock()
    s.shortlist_threshold = kw.get("threshold", 0.60)
    s.shortlist_k = 3
    s.threadless_window_days = 14
    return s


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _synth(sid, emb, urls=("https://a",)):
    return {"story_id": sid, "column": "politics", "generated_headline": sid,
            "summary": "s", "article_urls": list(urls),
            "generated_at": "2026-08-16T00:00:00+00:00",
            "summary_embedding": _unit(emb)}


def test_attach_on_yes():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.event_id = "event-1"; ev.summary = "old"; ev.status = "active"
    ev.summary_embedding = _unit([1.0, 0.0]); store.get_all_events.return_value = [ev]
    llm.generate.side_effect = ["yes", "updated summary"]
    emb_vec = _unit([0.9, 0.1])
    m = EventMatcher(store, llm, emb, _settings())
    got = m.match_story(_synth("story-9", emb_vec))
    assert got == "event-1"
    assert store.attach_chapter.called


def test_no_attach_on_no():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.event_id = "event-1"; ev.summary = "old"
    ev.summary_embedding = _unit([1.0, 0.0]); store.get_all_events.return_value = [ev]
    store.search_threadless.return_value = []
    llm.generate.return_value = "no"
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-9", _unit([0.9, 0.1]))) is None
    assert not store.attach_chapter.called


def test_below_threshold_skips_llm():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.summary_embedding = _unit([1.0, 0.0])
    store.get_all_events.return_value = [ev]
    store.search_threadless.return_value = []
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-9", _unit([0.0, 1.0]))) is None
    assert not llm.generate.called


def test_recurrence_birth():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    store.get_all_events.return_value = []
    old = _synth("story-old", [1.0, 0.05], urls=("https://old",))
    store.search_threadless.return_value = [old]
    llm.generate.side_effect = ["yes", "seed summary"]
    created = MagicMock(); created.event_id = "event-n"; store.create_event.return_value = created
    m = EventMatcher(store, llm, emb, _settings())
    got = m.match_story(_synth("story-new", _unit([0.95, 0.1]), urls=("https://new",)))
    assert got == "event-n"
    chapters = store.create_event.call_args.kwargs["chapters"]
    assert [c["story_id"] for c in chapters] == ["story-old", "story-new"]


def test_same_cluster_candidate_skipped():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    store.get_all_events.return_value = []
    dup = _synth("story-old", [1.0, 0.05], urls=("https://a", "https://b"))
    store.search_threadless.return_value = [dup]
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-new", _unit([1.0, 0.05]),
                                urls=("https://a", "https://b", "https://c"))) is None
    assert not llm.generate.called


def test_llm_error_returns_none():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.event_id = "event-1"; ev.summary = "old"
    ev.summary_embedding = _unit([1.0, 0.0]); store.get_all_events.return_value = [ev]
    llm.generate.side_effect = RuntimeError("down")
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-9", _unit([0.9, 0.1]))) is None
```

(This fixes `EventStore` needing `search_threadless(window_days) -> list[dict]`
— ADD it to Task 4's store: a `dorothy-synthesis` query for docs without
`event_id` and (`generated_at >= now-window` OR `thread_candidate: true`),
`_source` includes `summary_embedding`. The Task 4 implementer sees this
note via the plan; if Task 4 already merged, add it here.)

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_event_matcher.py -q` → FAIL.

- [ ] **Step 3: Implement** `src/events/matcher.py` per the interface + order above. Cosine:

```python
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
```

- [ ] **Step 4: Run tests** — `pytest tests/test_event_matcher.py tests/ -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/events/matcher.py src/events/store.py src/config.py tests/test_event_matcher.py
git commit -m "feat: event matcher - cosine shortlist + LLM confirm, recurrence birth"
```

---

### Task 6: Pipeline integration

**Files:**
- Modify: `scripts/run_pipeline.py` (`run_synthesis` at :189-320, cycle function, daemon setup)
- Modify: wherever the synthesis doc is built/stored after `_write_story` (grep `bulk_store_syntheses` callers): add `summary_embedding` (embed `generated_headline + "\n" + summary` via the pipeline's embedding client) before storing.
- Test: `tests/test_pipeline_events.py` (new; stub-level)

**Interfaces:**
- Consumes: `EventMatcher.match_story(synthesis) -> Optional[str]`, `EventStore.mark_dormant_older_than(days)`, `config.events`.
- Produces: synthesis docs stored with `summary_embedding` always, `event_id` when matched.

Wiring (all guarded by `config.events.enabled`):
1. Build `EventStore` + `EventMatcher` once per cycle (next to `graph_builder` construction, `run_pipeline.py:477-485`); call `store.ensure_index()`.
2. In `run_synthesis`, at the existing `story_evolved` branch (`:264-274`): after `mark_synthesis_historical(existing_id, new_id)`, record `inherited_event = existing.get("event_id")` and pass it through so the new synthesis doc gets that `event_id` and `store.attach_chapter` is called with the new chapter (Jaccard fast-path from spec §2.2). `find_overlapping_synthesis` must include `event_id` in its `_source` — extend it.
3. After each successful `_write_story` + store: if no inherited event, call `matcher.match_story(synthesis_doc)`; on an event_id, update the stored synthesis doc's `event_id`.
4. End of cycle: `store.mark_dormant_older_than(config.events.dormant_after_days)`; log count.
5. Any exception in 1–4 logs `events_stage_failed` and never fails the cycle.

- [ ] **Step 1: Write failing test** — `tests/test_pipeline_events.py` covering the inherit-on-evolve helper: extract the decision into a pure function `resolve_event_id(existing: Optional[dict], matcher_result: Optional[str]) -> Optional[str]` in `src/events/matcher.py` (inherited id wins over matcher result; both None → None) and test the three cases directly.

```python
from src.events.matcher import resolve_event_id


def test_inherited_event_wins():
    assert resolve_event_id({"event_id": "event-a"}, "event-b") == "event-a"


def test_matcher_used_when_no_inheritance():
    assert resolve_event_id({"event_id": None}, "event-b") == "event-b"
    assert resolve_event_id(None, "event-b") == "event-b"


def test_none_when_nothing():
    assert resolve_event_id(None, None) is None
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement** the helper + wiring per above. Embedding of the synthesis summary happens where the doc is built (one call per story; on embedding failure log `synthesis_embedding_failed` and store without — matcher skips docs lacking embeddings).
- [ ] **Step 4: Run** `pytest tests/ -q` → PASS; `.venv/bin/python -m py_compile scripts/run_pipeline.py` → OK.
- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py src/events/matcher.py src/storage/opensearch.py tests/test_pipeline_events.py
git commit -m "feat: wire event matching into publisher cycle"
```

---

### Task 7: Rendering — Developments section + event pages

**Files:**
- Modify: `scripts/render_static.py` (new methods `render_event_pages()`, `render_events_index()`; front-page context gains `developments`)
- Create: `src/web/templates/event.html`, `src/web/templates/events_index.html`
- Modify: `src/web/templates/front_page.html` (Developments section), `src/web/templates/story.html` ("The story so far" link when `event_id`)
- Test: visual/manual (Step 4); template-logic unit test not required.

**Interfaces:**
- Consumes: `EventStore.get_all_events()`, `get_event(event_id)`; syntheses carry `event_id`.
- Output paths: `output/event/<event_id>/index.html`, `output/events/index.html`.

Behavior:
- **Developments** (front page): syntheses from the last 72h that have an `event_id`, newest first, capped at 6. Each entry: headline → story page link, thread title link, "Previously: <prev chapter headline>, <date>" (prev = chapter before this story in `chapters[]`), and a "Last covered <Month Year>" badge when the gap to the previous chapter exceeds 14 days.
- **event.html**: title, rolling summary, status line ("Active — last development <date>" / "Dormant since <date>"), chapter timeline newest-first, each linking to its story page (`/story/<story_id>/`; verify actual story URL pattern in `render_story_pages` and match it).
- **events_index.html**: active threads then dormant, with title, one-line summary (truncate 160 chars), chapter count, last_seen.
- Follow existing template conventions (`base.html` blocks, existing CSS classes; broadsheet styling — check `column.html` for list markup to mimic). No new CSS files; reuse classes, add at most a few rules to the existing stylesheet if unavoidable.
- `render_static.py` main flow (grep for where `render_front_page`/`render_column_pages` are invoked) calls the two new renderers; events renderers no-op gracefully when the events index is absent (store call wrapped in try/except logging `events_render_skipped`).

- [ ] **Step 1: Implement** per above.
- [ ] **Step 2: Compile + full tests** — `python -m py_compile scripts/render_static.py`; `pytest tests/ -q` → PASS.
- [ ] **Step 3: Local render smoke** — cannot run without OpenSearch locally; rely on Jinja compile: add a tiny test that loads both new templates via `jinja2.Environment(loader=FileSystemLoader("src/web/templates")).get_template("event.html")` and renders with representative context (no OpenSearch needed). Include it as `tests/test_event_templates.py`:

```python
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader


ENV = Environment(loader=FileSystemLoader("src/web/templates"))
CTX = {
    "columns": ["politics"], "bias_colors": {}, "region_colors": {},
    "region_labels": {}, "perspective_colors": {}, "perspective_labels": {},
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "dateline": "Sunday, August 16, 2026", "edition": 1,
}


def test_event_page_renders():
    ev = {"event_id": "event-abc", "title": "T", "summary": "S", "status": "active",
          "last_seen": "2026-08-16T00:00:00+00:00", "first_seen": "2026-06-01T00:00:00+00:00",
          "chapters": [{"story_id": "story-1", "generated_headline": "H",
                        "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 4}]}
    html = ENV.get_template("event.html").render(event=ev, **CTX)
    assert "T" in html and "story-1" in html


def test_events_index_renders():
    html = ENV.get_template("events_index.html").render(active=[], dormant=[], **CTX)
    assert "html" in html.lower()
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit**

```bash
git add scripts/render_static.py src/web/templates/ tests/test_event_templates.py
git commit -m "feat: Developments section, event thread pages, events index"
```

---

### Task 8: Bootstrap — seed June syntheses as thread candidates

**Files:**
- Create: `scripts/backfill_event_candidates.py`
- Test: manual (run against CT 110 in Task 9); `python -m py_compile` locally.

**Behavior:** For every synthesis doc lacking `summary_embedding`: embed
`generated_headline + "\n" + (summary or "")[:500]`, update the doc with
`summary_embedding` and `thread_candidate: true`. Idempotent (skips docs that
already have embeddings). Uses `config` for the embedding client, prints a
count. ~35 docs today; batch of 8 with the existing embedding client API.
CLI: `python -m scripts.backfill_event_candidates [--dry-run]`.

- [ ] **Step 1: Implement** (follow `scripts/backfill_similarity.py` for the update-by-id pattern — read it first).
- [ ] **Step 2: Verify** — `python -m py_compile scripts/backfill_event_candidates.py`; `--dry-run` shape-check happens on CT 110 in Task 9.
- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_event_candidates.py
git commit -m "feat: backfill script - seed existing syntheses as thread candidates"
```

---

### Task 9: Deploy + live smoke (CT 110)

Performed by the orchestrator (not a subagent), after all tasks merge and the full suite passes.

- [ ] Full suite: `.venv/bin/python -m pytest tests/ -q` → all PASS.
- [ ] `git push origin main`; on CT 110: `git pull --ff-only`, `docker compose build fetcher extractor publisher`, `docker compose up -d`.
- [ ] Run `python -m scripts.backfill_event_candidates` inside the publisher container (or a one-off `docker compose run --rm pipeline-once python -m scripts.backfill_event_candidates`).
- [ ] Watch one publisher cycle. Acceptance:
  - `articles_grouped` shows NO cluster > ~100 and lifestyle input article_count reflects the 72h window (hundreds, not 920).
  - New syntheses exist with `generated_at` = today; front page dateline honest; no June stories on front/column pages.
  - `event_match_*` log lines present; `dorothy-events` index exists.
  - Site deploys; spot-check `/events/` renders (likely near-empty on day 1 — fine).

---

## Self-Review Notes

- Spec §1.1–1.4 → Tasks 1–3; §1.5 → explicitly untouched. §2.1 → Task 4; §2.2–2.3 → Task 5 (+ fast-path in Task 6); §2.4 → Task 7; §2.5 → Task 8; error handling → Task 5 catch-all + Task 6 guard; testing section → per-task tests + Task 9 smoke.
- Interface drift risks called out inline: embedding client method name (Tasks 5/6/8), `StaticRenderer` class name (Task 3), story URL pattern (Task 7), `find_overlapping_synthesis` `_source` extension (Task 6), `EventStore.search_threadless` added by Task 5's needs (noted in both Task 4 and Task 5).
