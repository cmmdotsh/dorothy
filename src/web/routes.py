"""Route handlers for Dorothy web server."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse

import structlog

from src.web.app import get_os_client

logger = structlog.get_logger(__name__)

COLUMNS = ["politics", "tech", "money", "sports", "lifestyle"]
BIAS_COLORS = {
    "left": "#3b82f6",       # blue
    "lean-left": "#60a5fa",  # light blue
    "center": "#a855f7",     # purple
    "lean-right": "#f97316", # orange
    "right": "#ef4444",      # red
}


def get_stories_for_column(column: str, limit: int = 10) -> list[dict]:
    """Get pre-synthesized stories for a column from OpenSearch."""
    os_client = get_os_client()
    return os_client.get_syntheses(column=column, limit=limit)


def register_routes(app: FastAPI) -> None:
    """Register all routes on the app."""

    @app.get("/", response_class=HTMLResponse)
    async def front_page(request: Request):
        """Front page with top stories from all columns."""
        templates = request.app.state.templates

        all_stories = {}
        for column in COLUMNS:
            try:
                # Get top 3 stories per column from pre-synthesized storage
                stories = get_stories_for_column(column, limit=3)
                all_stories[column] = stories
            except Exception as e:
                logger.error("column_fetch_error", column=column, error=str(e))
                all_stories[column] = []

        return templates.TemplateResponse(
            "front_page.html",
            {
                "request": request,
                "columns": COLUMNS,
                "stories_by_column": all_stories,
                "bias_colors": BIAS_COLORS,
                "dateline": datetime.now(timezone.utc).strftime("%A, %B %-d, %Y"),
            },
        )

    @app.get("/column/{column}", response_class=HTMLResponse)
    async def column_page(request: Request, column: str):
        """Column page with all stories for a category."""
        if column not in COLUMNS:
            raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

        templates = request.app.state.templates

        try:
            stories = get_stories_for_column(column, limit=20)
        except Exception as e:
            logger.error("column_fetch_error", column=column, error=str(e))
            stories = []

        return templates.TemplateResponse(
            "column.html",
            {
                "request": request,
                "column": column,
                "columns": COLUMNS,
                "stories": stories,
                "bias_colors": BIAS_COLORS,
                "dateline": datetime.now(timezone.utc).strftime("%A, %B %-d, %Y"),
            },
        )

    @app.get("/story/{story_id}", response_class=HTMLResponse)
    async def story_page(request: Request, story_id: str):
        """Individual story detail page."""
        templates = request.app.state.templates

        os_client = get_os_client()
        story = os_client.get_synthesis_by_id(story_id)

        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        return templates.TemplateResponse(
            "story.html",
            {
                "request": request,
                "story": story,
                "columns": COLUMNS,
                "bias_colors": BIAS_COLORS,
                "dateline": datetime.now(timezone.utc).strftime("%A, %B %-d, %Y"),
            },
        )

    @app.get("/about", response_class=HTMLResponse)
    async def about_page(request: Request):
        """About this site page."""
        templates = request.app.state.templates

        return templates.TemplateResponse(
            "about.html",
            {
                "request": request,
                "columns": COLUMNS,
                "bias_colors": BIAS_COLORS,
                "page": "about",
                "dateline": datetime.now(timezone.utc).strftime("%A, %B %-d, %Y"),
            },
        )

    # API Endpoints

    @app.get("/api/columns")
    async def api_columns():
        """List available columns."""
        return {"columns": COLUMNS}

    @app.get("/api/stories")
    async def api_stories(column: Optional[str] = None, limit: int = 10):
        """Get synthesized stories."""
        if column and column not in COLUMNS:
            raise HTTPException(status_code=400, detail=f"Invalid column: {column}")

        if column:
            stories = get_stories_for_column(column, limit=limit)
            return {"column": column, "stories": stories}

        # All columns
        all_stories = {}
        for col in COLUMNS:
            all_stories[col] = get_stories_for_column(col, limit=3)

        return {"stories_by_column": all_stories}

    @app.get("/api/stories/{story_id}")
    async def api_story(story_id: str):
        """Get a single story by ID."""
        os_client = get_os_client()
        story = os_client.get_synthesis_by_id(story_id)

        if not story:
            raise HTTPException(status_code=404, detail="Story not found")

        return story
