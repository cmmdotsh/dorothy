# Daily Window + Event Threads — Design

**Date:** 2026-08-16
**Status:** Approved in discussion; pending spec review

## Motivation

Post-revival evidence (2026-08-16 diagnostics):

- Clustering has no recency window: `get_stories_for_column` never passes the
  `since` filter that `search_articles` already supports. The NPR Life Kit feed
  re-emits its 2018–2026 evergreen archive; 663/920 lifestyle articles came from
  that one feed and 657 of them formed a single 672-article HDBSCAN cluster.
- Story dates are `MIN(pub_date)` over the cluster with no clamp: one 2022-dated
  article rewrites a story's displayed date. Future dates clamp `hours_since` to
  1.0 and pin stories to maximum hotness forever.
- When a cycle yields zero syntheses, stale `is_current` rows re-render under a
  fresh masthead dateline (a June paper printed under an August dateline).
- There is no cross-cycle story identity: `story_id` is a hash of the member URL
  set, so an evolving story becomes a chain of unrelated docs, and nothing can
  reconnect a development to a story from months ago.

Owner intent (settled in brainstorming):

1. The front page is a **daily paper**: 24–72h clustering window.
2. **Events** are long-lived threads (weeks/months) that daily stories attach
   to — e.g. an ongoing conflict, or a resolution landing months after chapter 1.
3. Threads surface via a **hybrid** layout: daily paper unchanged, plus a
   "Developments" section for thread movement (including dormant-thread wakes).
4. **Recurrence earns thread status**: no LLM triage at birth; a thread is born
   retroactively when a later story matches a prior one.

## Part 1 — Daily-layer tuning

### 1.1 Recency window
- New `ClusteringSettings` in `src/config.py` (env prefix `CLUSTERING_`):
  `window_hours: int = 72`, plus the currently hardcoded HDBSCAN params
  (`min_cluster_size=3`, `min_samples=2`) promoted to config.
- `StoryGrouper.get_stories_for_column` passes `since = now - window_hours` to
  `search_articles` (`src/clustering/story_grouper.py:352`,
  `src/storage/opensearch.py:296`).
- Near month boundaries (`now - window_hours` crosses into the previous month),
  query both monthly indices (extend `search_articles` to take multiple index
  names; OpenSearch accepts comma-separated indices natively).

### 1.2 Date hygiene
- Ingest keeps the feed's raw `pub_date` verbatim (data preserved), but
  `compute_story_timing` (`src/synthesis/summarizer.py`) computes story dates
  from **window-clamped** pub_dates: values older than the window floor to the
  window start; future values clamp to `fetched_at`.
- Ranking (`get_syntheses` script score, `src/storage/opensearch.py:655`) can no
  longer see future dates, removing the "infinitely hot" pin.

### 1.3 Feed-dominance cap
- Per cycle, per column: at most `CLUSTERING_MAX_PER_SOURCE` (default 40)
  articles per `source_slug` enter clustering, newest `pub_date` first.
- Applied in `get_stories_for_column` after fetch, before the distance matrix.

### 1.4 Honest dateline
- Front page and column pages render only syntheses with
  `generated_at >= now - 72h` (render-time filter in `scripts/render_static.py`;
  no data mutation — old rows stay `is_current` for thread history).
- A column with nothing fresh renders thin with a small "quiet day" note in the
  template. Never re-print stale syntheses under today's dateline.

### 1.5 Explicitly unchanged
- Body-text synthesis gate (≥3 bodies from ≥2 sources): starved during the
  outage, not wrong. Re-evaluate after several live cycles.
- HDBSCAN algorithm/selection method: the megacluster was an input problem
  (archive dump + no window), not clustering math (0 merge events; healthy June
  cycles topped out at 41-article clusters).

## Part 2 — Event threads

### 2.1 Model
- New index `dorothy-events`:
  - `event_id` (stable id, e.g. `event-<12 hex>`), `title`,
  - `summary` — rolling prose summary, LLM-refreshed on each attach; this is
    the text future matching embeds against,
  - `summary_embedding` (1024-d knn_vector, same model as articles),
  - `status`: `active` | `dormant` (dormant after 14 quiet days, checked at
    cycle end; nothing auto-resolves — a "resolution" is just a chapter that
    ends coverage),
  - `chapters[]`: `{story_id, generated_headline, generated_at, article_count}`,
  - `columns[]`, `first_seen`, `last_seen`.
- `dorothy-synthesis` docs gain optional `event_id`.

### 2.2 Matching (at synthesis time, per new story)
Ordered, first match wins:
1. **Jaccard fast-path** (existing overlap machinery): same-week URL overlap →
   attach to that story's thread (or found one — see 2.3).
2. **Embedding shortlist + LLM confirm**: embed new story's
   `generated_headline + summary`; k-NN against `summary_embedding` of all
   threads, active and dormant (dormant matching is the point);
   take top-3 above a loose cosine threshold (start 0.60); for each candidate,
   one small-model yes/no: "Is this story a development of this ongoing event?
   <thread summary> / <new story headline+summary>". Bias the prompt toward
   "no"; require a strict `yes` token to attach.
3. No match → story stays threadless. No thread is created for first chapters.

Every decision (candidates, scores, LLM verdicts) is logged for audit.

### 2.3 Thread birth (recurrence rule)
- When step 1 or 2 matches a story that is itself **threadless**, a new thread
  is created containing both stories (the old one becomes chapter 1
  retroactively); its initial summary is LLM-generated from both syntheses.
- Wrong "no" (split threads) is recoverable: two threads whose new chapters
  keep matching each other can be merged manually or by the same recurrence
  logic later. Wrong "yes" (polluted thread) is worse — hence the "no" bias.

### 2.4 Rendering
- **Front page**: "Developments" section listing this cycle's thread-attached
  stories with "Previously: <chapter N-1 headline>, <date>", plus a visible
  badge when the thread was dormant ("last covered <month>").
- **`/event/<event_id>/`**: rolling summary + chapter timeline (newest first).
- **`/events/`**: index of active and dormant threads.
- Story pages link to their thread ("The story so far").

### 2.5 Bootstrap
- One-off script seeds threads from the 35 existing June syntheses: they are
  inserted as candidate chapters (threadless stories with embeddings), so
  August news can wake them via the normal recurrence rule. No hand-curated
  threads.

## Error handling
- Thread matching failure (LLM down, embedding error) MUST NOT block
  publishing: the story publishes threadless; a later cycle can still attach it
  (matching considers threadless stories from the last 14 days, not only the
  current cycle's).
- `dorothy-events` writes are idempotent by `event_id`; attach operations
  re-run safely.

## Testing
- Unit: window clamp math; per-source cap selection; recurrence birth logic
  (threadless+threadless → new thread; threadless+threaded → attach); dormancy
  transition.
- Integration (against live LMStudio, manual): matching yes/no on a handful of
  constructed pairs (same event vs same topic different event — e.g. two
  distinct Iran stories must NOT merge).
- Smoke: one full publisher cycle on CT 110; verify window (no pre-window
  articles clustered), no Life Kit blob, Developments section renders.

## Out of scope
- Podcast leg (tabled — owner has other plans).
- Body-gate relaxation, HDBSCAN retuning (revisit with live-cycle evidence).
- Repo/CT litter cleanup happens after this work lands (owner-sequenced).
