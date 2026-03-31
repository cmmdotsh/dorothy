# Dorothy — Web & Static Site

Dorothy has two modes for serving content:

1. **FastAPI dev server** — live queries OpenSearch, for development
2. **Static site generator** — renders everything to HTML files, deployed to S3 for production

## FastAPI server

**Module:** `src/web/`
**Entry point:** `scripts/run_server.py`

### Running the server

```bash
# Default: localhost:8000
python -m scripts.run_server

# With options
python -m scripts.run_server --host 0.0.0.0 --port 8080 --reload
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port |
| `--reload` | off | Enable auto-reload on file changes (development) |

In Docker (dev profile):
```bash
docker-compose --profile dev up web
# → http://localhost:8000
```

### Routes

#### HTML pages

| Route | Template | Description |
|-------|----------|-------------|
| `GET /` | `front_page.html` | Front page: top 3 stories per column |
| `GET /column/{column}` | `column.html` | Column page: up to 20 stories |
| `GET /story/{story_id}` | `story.html` | Story detail: full article + analysis + sources |
| `GET /about` | `about.html` | About page |

#### JSON API

| Route | Query params | Description |
|-------|-------------|-------------|
| `GET /api/columns` | — | List valid column names |
| `GET /api/stories` | `column`, `limit` | Get stories (all columns or filtered) |
| `GET /api/stories/{story_id}` | — | Get single synthesized story by ID |

#### Static assets

Mounted at `/static` → `src/web/static/`

### Template context

All page templates receive these common context variables:

| Variable | Type | Description |
|----------|------|-------------|
| `columns` | `list[str]` | All column names (for nav) |
| `bias_colors` | `dict` | Hex colors per bias tier |
| `dateline` | `str` | Current UTC date (e.g., "Friday, March 13, 2026") |
| `edition` | `int` | Current pipeline run number |

Page-specific variables:

**Front page:**
- `stories_by_column` — dict of `{column: [story, ...]}`, 3 stories per column

**Column page:**
- `column` — the column name
- `stories` — list of up to 20 synthesized stories

**Story page:**
- `story` — full synthesized story dict with all fields

### Templates

| File | Description |
|------|-------------|
| `base.html` | Base layout: nav, header, footer |
| `front_page.html` | Front page grid |
| `column.html` | Column article list |
| `story.html` | Story detail with tabs (article / analysis / sources) |
| `about.html` | About page |
| `podcast.html` | Podcast episode list (used by static renderer) |

Templates use Jinja2. A `markdown` filter is registered to render markdown in templates (via `markdown-it`).

## Static site generator

**Module:** `scripts/render_static.py`
**Class:** `StaticSiteGenerator`

The static generator renders all pages to files in an output directory (default: `./output/`). This is what gets deployed to S3 for production.

### Running the renderer

```bash
# Render to ./output/
python -m scripts.render_static

# Clean output directory first
python -m scripts.render_static --clean

# Custom output directory
python -m scripts.render_static --output /tmp/dorothy-site
```

| Flag | Description |
|------|-------------|
| `--output PATH` | Output directory (default: `./output`) |
| `--clean` | Delete and recreate the output directory before rendering |

In Docker (manual profile):
```bash
docker-compose --profile manual run render
```

### What gets generated

```
output/
├── index.html                    # Front page
├── about/
│   └── index.html                # About page
├── column/
│   ├── politics/index.html
│   ├── tech/index.html
│   ├── money/index.html
│   ├── sports/index.html
│   └── lifestyle/index.html
├── story/
│   └── {story_id}/index.html     # One directory per story
├── podcast/
│   └── index.html                # Podcast episode list (if episodes exist)
└── static/
    ├── style.[hash].css          # Version-stamped CSS
    ├── app.[hash].js             # Version-stamped JS
    └── ...                       # Other static assets (SVGs, fonts, etc.)
```

Static assets (CSS, JS) are version-stamped with a content hash so CDN caches are busted automatically on changes.

### Per-column deduplication

Before rendering column pages, the static generator runs a final deduplication pass:

- For each column, sort stories by hotness score (descending)
- For each story, compute Jaccard index of article URLs against all previously included stories
- Skip stories with Jaccard > 0.30 (more than 30% URL overlap with an already-included story)

This prevents near-duplicate stories from appearing side-by-side on column pages.

### Color scheme

The renderer provides these color maps to templates:

**Bias colors:**
```python
{
    "left":       "#3b82f6",   # blue
    "lean-left":  "#60a5fa",   # light blue
    "center":     "#a855f7",   # purple
    "lean-right": "#f97316",   # orange
    "right":      "#ef4444",   # red
}
```

**Region colors (sports):**
```python
{
    "us":            "#3b82f6",
    "canada":        "#ef4444",
    "mexico":        "#22c55e",
    "uk":            "#6366f1",
    "australia":     "#eab308",
    "india":         "#f97316",
    "japan":         "#ec4899",
    "korea":         "#14b8a6",
    "international": "#8b5cf6",
}
```

**Perspective colors (tech):**
```python
{
    "consumer":   "#3b82f6",
    "enterprise": "#f97316",
    "academic":   "#a855f7",
    "culture":    "#22c55e",
}
```

### Podcast integration

If a podcast feed exists at `output/podcast/feed.xml`, the static renderer reads it and renders a `podcast/index.html` listing recent episodes. Episode metadata (title, duration, URL) is extracted from the RSS feed XML.

## Story data structure (used by templates)

Stories returned from OpenSearch and passed to templates include:

```json
{
  "story_id": "abc123...",
  "column": "politics",
  "headline": "Senate Passes Budget Bill",
  "generated_headline": "Senate Approves Bipartisan Budget Agreement",
  "article_text": "The Senate passed...",
  "analysis": "Left-leaning outlets focused on...",
  "sources_used": ["ap", "nyt", "foxnews"],
  "bias_coverage": {"center": 2, "lean-left": 3, "right": 1},
  "article_count": 6,
  "articles": [
    {"url": "https://...", "source_name": "AP", "headline": "Senate votes...", "source_bias": "center"}
  ],
  "hero_image_url": "https://...",
  "hero_image_source": "Reuters",
  "similarity_edges": [
    {"source": "ap", "target": "nyt", "weight": 0.82}
  ],
  "hotness_score": 4.2,
  "median_pub_date": "2026-03-13T14:00:00Z",
  "edition": 42,
  "is_current": true
}
```
