"""Article fetchers for Dorothy."""

from .rss import RSSFetcher, fetch_all_sources

__all__ = ["RSSFetcher", "fetch_all_sources"]
