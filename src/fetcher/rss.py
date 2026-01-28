"""RSS feed fetcher and parser."""

import re
from datetime import datetime, timezone
from typing import Iterator, Optional

import feedparser
import httpx
import structlog
from dateutil import parser as date_parser

from src.models import Article, Source

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class RSSFetchError(Exception):
    """Error fetching or parsing RSS feed."""

    pass


class RSSFetcher:
    """
    Fetches and parses RSS feeds from configured sources.

    Uses httpx for fetching (better timeout/error handling) and
    feedparser for parsing (handles RSS/Atom/edge cases).
    """

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = "Dorothy/0.1 (news aggregator)",
    ):
        self.timeout = timeout
        self.user_agent = user_agent
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def fetch_feed(self, source: Source) -> Iterator[Article]:
        """
        Fetch and parse a single RSS feed.

        Args:
            source: Source configuration with RSS URL

        Yields:
            Article instances for each valid feed entry
        """
        if not source.rss_url:
            logger.warning("source_no_rss_url", source=source.slug)
            return

        try:
            response = self.client.get(str(source.rss_url))
            response.raise_for_status()

            feed = feedparser.parse(response.text)

            if feed.bozo and feed.bozo_exception:
                logger.warning(
                    "feed_parse_warning",
                    source=source.slug,
                    error=str(feed.bozo_exception),
                )

            fetched_at = _utcnow()

            for entry in feed.entries:
                try:
                    article = self._entry_to_article(entry, source, fetched_at)
                    if article:
                        yield article
                except Exception as e:
                    logger.warning(
                        "entry_parse_error",
                        source=source.slug,
                        entry_title=getattr(entry, "title", "unknown"),
                        error=str(e),
                    )

        except httpx.HTTPError as e:
            logger.error("feed_fetch_error", source=source.slug, error=str(e))
        except Exception as e:
            logger.error("feed_unexpected_error", source=source.slug, error=str(e))

    def _entry_to_article(
        self,
        entry: feedparser.FeedParserDict,
        source: Source,
        fetched_at: datetime,
    ) -> Optional[Article]:
        """Convert feedparser entry to Article model."""
        headline = entry.get("title", "").strip()
        if not headline:
            return None

        url = entry.get("link", "").strip()
        if not url:
            return None

        summary = None
        if "summary" in entry:
            summary = self._clean_html(entry.summary)
        elif "description" in entry:
            summary = self._clean_html(entry.description)

        pub_date = self._parse_date(entry)
        if not pub_date:
            pub_date = fetched_at

        image_url = self._extract_image(entry)

        return Article(
            source_name=source.name,
            source_slug=source.slug,
            source_bias=source.bias,
            column=source.column,
            headline=headline[:500],
            summary=summary[:2000] if summary else None,
            url=url,
            pub_date=pub_date,
            fetched_at=fetched_at,
            image_url=image_url,
        )

    def _parse_date(self, entry: feedparser.FeedParserDict) -> Optional[datetime]:
        """Parse publication date from various feed formats."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                # Create timezone-aware UTC datetime
                return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass

        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass

        for field in ["published", "updated", "created"]:
            if field in entry and entry[field]:
                try:
                    parsed = date_parser.parse(entry[field])
                    # Make timezone-aware if naive
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed
                except (ValueError, TypeError):
                    continue

        return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _extract_image(self, entry: feedparser.FeedParserDict) -> Optional[str]:
        """Extract image URL from RSS entry."""
        # Try media:thumbnail first (BBC, etc)
        if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
            url = entry.media_thumbnail[0].get("url")
            if url:
                return url

        # Try media:content (Guardian, Fox, etc)
        if hasattr(entry, "media_content") and entry.media_content:
            for media in entry.media_content:
                # Prefer ones with explicit image type
                if media.get("medium") == "image" or media.get("type", "").startswith(
                    "image/"
                ):
                    return media.get("url")
            # Fallback: take first media_content with a URL (Guardian doesn't set type)
            for media in entry.media_content:
                url = media.get("url", "")
                if url:
                    return url

        # Try enclosures (standard RSS)
        if hasattr(entry, "enclosures") and entry.enclosures:
            for enc in entry.enclosures:
                if enc.get("type", "").startswith("image/"):
                    return enc.get("url")

        return None


def fetch_all_sources(sources: list[Source]) -> Iterator[Article]:
    """
    Fetch articles from all active RSS sources.

    Args:
        sources: List of source configurations

    Yields:
        Article instances from all sources
    """
    fetcher = RSSFetcher()

    try:
        for source in sources:
            if not source.active:
                continue
            if source.fetch_method.value != "rss":
                continue

            logger.info("fetching_source", source=source.slug)

            count = 0
            for article in fetcher.fetch_feed(source):
                count += 1
                yield article

            logger.info("source_fetched", source=source.slug, articles=count)

    finally:
        fetcher.close()
