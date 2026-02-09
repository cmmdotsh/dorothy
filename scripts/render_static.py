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
import shutil
from pathlib import Path
from datetime import datetime, timezone

import structlog
from jinja2 import Environment, FileSystemLoader
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
        """Remove existing output directory."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            console.print(f"[dim]Cleaned {self.output_dir}[/dim]")

    def setup_output(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "column").mkdir(exist_ok=True)
        (self.output_dir / "story").mkdir(exist_ok=True)
        (self.output_dir / "static").mkdir(exist_ok=True)

    def copy_static_assets(self) -> None:
        """Copy static files to output."""
        if self.static_dir.exists():
            for file in self.static_dir.iterdir():
                if file.is_file():
                    shutil.copy(file, self.output_dir / "static" / file.name)
            console.print(f"[green]  Copied static assets[/green]")

    def get_stories_for_column(self, column: str, limit: int = 20) -> list[dict]:
        """Get synthesized stories for a column."""
        return self.os_client.get_syntheses(column=column, limit=limit)

    def get_all_stories(self) -> list[dict]:
        """Get all synthesized stories across all columns."""
        all_stories = []
        for column in COLUMNS:
            stories = self.os_client.get_syntheses(column=column, limit=100)
            all_stories.extend(stories)
        return all_stories

    def render_template(self, template_name: str, context: dict) -> str:
        """Render a template with context."""
        template = self.env.get_template(template_name)
        # Add common context
        now = datetime.now(timezone.utc)
        context.update({
            "columns": COLUMNS,
            "bias_colors": BIAS_COLORS,
            "generated_at": now.isoformat(),
            "dateline": now.strftime("%A, %B %-d, %Y"),
        })
        return template.render(**context)

    def write_page(self, path: Path, content: str) -> None:
        """Write rendered HTML to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def render_front_page(self) -> None:
        """Render the front page."""
        stories_by_column = {}
        for column in COLUMNS:
            stories_by_column[column] = self.get_stories_for_column(column, limit=3)

        html = self.render_template("front_page.html", {
            "stories_by_column": stories_by_column,
        })

        self.write_page(self.output_dir / "index.html", html)
        console.print("[green]  Rendered index.html[/green]")

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
        """Render all individual story pages. Returns count."""
        all_stories = self.get_all_stories()
        rendered = 0

        for story in all_stories:
            story_id = story.get("story_id")
            if not story_id:
                continue

            html = self.render_template("story.html", {
                "story": story,
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
        self.render_front_page()
        self.render_column_pages()
        self.render_about_page()
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
