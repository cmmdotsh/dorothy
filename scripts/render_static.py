#!/usr/bin/env python3
"""
Static Site Generator for Dorothy

Renders all pages to static HTML files for S3 hosting.

Usage:
    python -m scripts.render_static                    # Render to ./output/
    python -m scripts.render_static --output /path     # Custom output directory
    python -m scripts.render_static --clean            # Clean output before render
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone
from xml.etree.ElementTree import parse as parse_xml

import structlog
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup
from markdown_it import MarkdownIt
from rich.console import Console

from src.config import config
from src.storage import OpenSearchClient

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()

COLUMNS = ["politics", "tech", "money", "sports", "lifestyle"]
BIAS_COLORS = {
    "left": "#3b82f6",
    "lean-left": "#60a5fa",
    "center": "#a855f7",
    "lean-right": "#f97316",
    "right": "#ef4444",
}

REGION_COLORS = {
    "us": "#3b82f6",
    "canada": "#ef4444",
    "mexico": "#22c55e",
    "uk": "#6366f1",
    "australia": "#eab308",
    "india": "#f97316",
    "japan": "#ec4899",
    "korea": "#14b8a6",
    "international": "#8b5cf6",
}

REGION_LABELS = {
    "us": "US",
    "canada": "Canada",
    "mexico": "Mexico",
    "uk": "UK",
    "australia": "Australia",
    "india": "India",
    "japan": "Japan",
    "korea": "Korea",
    "international": "Intl",
}

PERSPECTIVE_COLORS = {
    "consumer": "#3b82f6",
    "enterprise": "#f97316",
    "academic": "#a855f7",
    "culture": "#22c55e",
}

PERSPECTIVE_LABELS = {
    "consumer": "Consumer",
    "enterprise": "Enterprise",
    "academic": "Academic",
    "culture": "Culture",
}


def _dedup_stories(stories: list[dict], threshold: float = 0.3) -> list[dict]:
    """Remove duplicates based on article URL overlap (Jaccard > threshold).

    Keeps the story with more articles (input order breaks ties, so stories
    pre-sorted by hotness are preferred).
    """
    kept: list[dict] = []
    kept_url_sets: list[set[str]] = []

    for story in stories:
        urls = set(story.get("article_urls", []))
        if not urls:
            kept.append(story)
            kept_url_sets.append(set())
            continue

        is_dup = False
        for prev_urls in kept_url_sets:
            if not prev_urls:
                continue
            intersection = len(urls & prev_urls)
            union = len(urls | prev_urls)
            if union > 0 and intersection / union > threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(story)
            kept_url_sets.append(urls)

    return kept


def get_podcast_episodes(output_dir: Path) -> list[dict]:
    """Parse podcast/feed.xml and return episode metadata sorted newest first."""
    feed_path = output_dir / "podcast" / "feed.xml"
    if not feed_path.exists():
        return []

    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    tree = parse_xml(str(feed_path))
    episodes = []

    for item in tree.findall(".//item"):
        title = item.findtext("title", "")
        pub_date = item.findtext("pubDate", "")
        duration = item.findtext("itunes:duration", "", ns)
        enclosure = item.find("enclosure")
        url = enclosure.get("url", "") if enclosure is not None else ""
        filename = url.rsplit("/", 1)[-1] if url else ""

        episodes.append({
            "title": title,
            "url": url,
            "pub_date": pub_date,
            "duration": duration,
            "filename": filename,
        })

    return episodes


class StaticSiteGenerator:
    """Generates static HTML from OpenSearch data."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.templates_dir = Path("src/web/templates")
        self.static_dir = Path("src/web/static")

        # Initialize Jinja2
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True,
        )
        md = MarkdownIt()
        self.env.filters["markdown"] = lambda text: Markup(md.render(text)) if text else ""

        # Initialize OpenSearch client
        auth_kwargs = {}
        if config.opensearch.username and config.opensearch.password:
            auth_kwargs["username"] = config.opensearch.username
            auth_kwargs["password"] = config.opensearch.password

        self.os_client = OpenSearchClient(
            host=config.opensearch.host,
            port=config.opensearch.port,
            use_ssl=config.opensearch.use_ssl,
            **auth_kwargs,
        )

    def clean(self) -> None:
        """Remove contents of the output directory (safe for volume mounts).

        Preserves the podcast/ subdirectory which is managed independently,
        and any hidden directories (e.g. .tmp used by podcast TTS).
        """
        if self.output_dir.exists():
            for child in self.output_dir.iterdir():
                if child.name == "podcast" or child.name.startswith("."):
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            console.print(f"[dim]Cleaned {self.output_dir}[/dim]")

    def setup_output(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "column").mkdir(exist_ok=True)
        (self.output_dir / "story").mkdir(exist_ok=True)
        (self.output_dir / "static").mkdir(exist_ok=True)

    def copy_static_assets(self) -> None:
        """Copy static files to output with content-hashed filenames for CSS/JS."""
        self.assets = {}
        hash_files = {"style.css", "app.js", "similarity-web.js"}

        if not self.static_dir.exists():
            return

        # Hash CSS and JS files
        for name in hash_files:
            src = self.static_dir / name
            if not src.exists():
                continue
            content = src.read_bytes()
            file_hash = hashlib.md5(content).hexdigest()[:8]
            stem, ext = name.rsplit(".", 1)
            hashed_name = f"{stem}.{file_hash}.{ext}"
            shutil.copy(src, self.output_dir / "static" / hashed_name)
            self.assets[name] = hashed_name

        # Build a combined hash for the cache name
        cache_hash = hashlib.md5("".join(sorted(self.assets.values())).encode()).hexdigest()[:8]

        # Process sw.js — inject hashed CACHE_NAME and PRECACHE_URLS
        sw_src = self.static_dir / "sw.js"
        if sw_src.exists():
            sw_content = sw_src.read_text()
            sw_content = re.sub(
                r"const CACHE_NAME = '[^']*'",
                f"const CACHE_NAME = 'dorothy-{cache_hash}'",
                sw_content,
            )
            precache = [
                "'/'",
                f"'/static/{self.assets.get('style.css', 'style.css')}'",
                f"'/static/{self.assets.get('app.js', 'app.js')}'",
                "'/static/manifest.json'",
            ]
            sw_content = re.sub(
                r"const PRECACHE_URLS = \[.*?\];",
                "const PRECACHE_URLS = [\n  " + ",\n  ".join(precache) + "\n];",
                sw_content,
                flags=re.DOTALL,
            )
            (self.output_dir / "static" / "sw.js").write_text(sw_content)

        # Copy remaining static files as-is (skip already-handled ones)
        handled = hash_files | {"sw.js"}
        for file in self.static_dir.iterdir():
            if file.is_file() and file.name not in handled:
                shutil.copy(file, self.output_dir / "static" / file.name)

        # Make assets available to templates
        self.env.globals["assets"] = self.assets

        console.print(f"[green]  Copied static assets (hashed: {', '.join(self.assets.values())})[/green]")

    def _backfill_image_credit(self, story: dict) -> dict:
        """Derive hero_image_source from articles if not already set."""
        if story.get("hero_image_source") or not story.get("hero_image_url"):
            return story
        hero_url = story["hero_image_url"]
        for article in story.get("articles", []):
            if article.get("image_url") == hero_url:
                story["hero_image_source"] = article.get("source_name", "")
                break
        return story

    def get_stories_for_column(self, column: str, limit: int = 20) -> list[dict]:
        """Get synthesized stories for a column (deduped)."""
        stories = self.os_client.get_syntheses(column=column, limit=max(limit * 5, 50))
        stories = _dedup_stories(stories)[:limit]
        return [self._backfill_image_credit(s) for s in stories]

    def get_all_stories(self) -> list[dict]:
        """Get all synthesized stories across all columns."""
        all_stories = []
        for column in COLUMNS:
            stories = self.os_client.get_syntheses(column=column, limit=200)
            stories = _dedup_stories(stories)[:100]
            all_stories.extend(stories)
        return all_stories

    def get_edition(self) -> int:
        """Get the current edition number from OpenSearch."""
        return self.os_client.get_edition()

    def render_template(self, template_name: str, context: dict) -> str:
        """Render a template with context."""
        template = self.env.get_template(template_name)
        # Add common context
        now = datetime.now(timezone.utc)
        context.update({
            "columns": COLUMNS,
            "bias_colors": BIAS_COLORS,
            "region_colors": REGION_COLORS,
            "region_labels": REGION_LABELS,
            "perspective_colors": PERSPECTIVE_COLORS,
            "perspective_labels": PERSPECTIVE_LABELS,
            "generated_at": now.isoformat(),
            "dateline": now.strftime("%A, %B %-d, %Y"),
            "edition": context.get("edition") or self.get_edition() or 1,
        })
        return template.render(**context)

    def write_page(self, path: Path, content: str) -> None:
        """Write rendered HTML to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def render_front_page(self, episodes: list[dict] | None = None) -> None:
        """Render the front page."""
        stories_by_column = {}
        for column in COLUMNS:
            stories_by_column[column] = self.get_stories_for_column(column, limit=3)

        html = self.render_template("front_page.html", {
            "stories_by_column": stories_by_column,
            "latest_episode": episodes[0] if episodes else None,
        })

        self.write_page(self.output_dir / "index.html", html)
        console.print("[green]  Rendered index.html[/green]")

    def render_podcast_page(self, episodes: list[dict]) -> None:
        """Render the podcast archive page."""
        if not episodes:
            return
        html = self.render_template("podcast.html", {
            "episodes": episodes,
            "latest_episode": episodes[0],
            "page": "podcast",
        })
        self.write_page(self.output_dir / "podcast.html", html)
        console.print(f"[green]  Rendered podcast.html ({len(episodes)} episodes)[/green]")

    def render_column_pages(self) -> None:
        """Render all column pages."""
        for column in COLUMNS:
            stories = self.get_stories_for_column(column, limit=20)

            html = self.render_template("column.html", {
                "column": column,
                "stories": stories,
            })

            self.write_page(self.output_dir / "column" / column / "index.html", html)
            console.print(f"[green]  Rendered column/{column}/index.html ({len(stories)} stories)[/green]")

    def render_about_page(self) -> None:
        """Render the about page."""
        html = self.render_template("about.html", {"page": "about"})
        self.write_page(self.output_dir / "about" / "index.html", html)
        console.print("[green]  Rendered about/index.html[/green]")

    def render_story_pages(self) -> int:
        """Render all individual story pages with navigation. Returns count."""
        # Group stories by column for navigation
        stories_by_column = {}
        for column in COLUMNS:
            stories = self.os_client.get_syntheses(column=column, limit=200)
            stories = _dedup_stories(stories)[:100]
            stories_by_column[column] = [self._backfill_image_credit(s) for s in stories]

        rendered = 0

        # Render each story with prev/next navigation within its column
        for column, stories in stories_by_column.items():
            for i, story in enumerate(stories):
                story_id = story.get("story_id")
                if not story_id:
                    continue

                # Get prev/next story IDs within the same column
                prev_story = stories[i - 1].get("story_id") if i > 0 else None
                next_story = stories[i + 1].get("story_id") if i < len(stories) - 1 else None

                html = self.render_template("story.html", {
                    "story": story,
                    "prev_story": prev_story,
                    "next_story": next_story,
                })

                self.write_page(self.output_dir / "story" / story_id / "index.html", html)
                rendered += 1

        console.print(f"[green]  Rendered {rendered} story pages[/green]")
        return rendered

    def generate(self) -> dict:
        """Generate the complete static site."""
        start = datetime.now(timezone.utc)

        console.print("[bold blue]Generating static site...[/bold blue]")

        self.setup_output()
        self.copy_static_assets()
        episodes = get_podcast_episodes(self.output_dir)
        self.render_front_page(episodes)
        self.render_column_pages()
        self.render_about_page()
        self.render_podcast_page(episodes)
        story_count = self.render_story_pages()

        duration = (datetime.now(timezone.utc) - start).total_seconds()

        console.print(f"\n[bold green]Static site generated in {duration:.1f}s[/bold green]")
        console.print(f"[dim]Output: {self.output_dir.absolute()}[/dim]")

        return {
            "output_dir": str(self.output_dir.absolute()),
            "story_count": story_count,
            "column_count": len(COLUMNS),
            "duration_seconds": duration,
        }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy Static Site Generator")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help="Clean output directory before rendering",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    generator = StaticSiteGenerator(output_dir)

    if args.clean:
        generator.clean()

    generator.generate()


if __name__ == "__main__":
    main()
