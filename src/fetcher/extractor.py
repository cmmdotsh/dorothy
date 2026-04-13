"""Full article text extraction via trafilatura (HTML -> Markdown)."""

import time
from typing import Optional

import httpx
import structlog
import trafilatura

from src.storage import OpenSearchClient

logger = structlog.get_logger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _resolve_google_news_url(url: str) -> Optional[str]:
    """Resolve a Google News redirect URL to the real article URL.

    Google News RSS feeds use opaque redirect URLs that encode the real
    destination. Uses googlenewsdecoder to extract the actual article URL.
    Returns None if resolution fails (expired link, rate limited, etc).
    """
    try:
        from googlenewsdecoder.new_decoderv1 import decode_google_news_url

        result = decode_google_news_url(url)
        if result.get("status") and result.get("decoded_url"):
            logger.debug(
                "google_news_resolved",
                decoded_url=result["decoded_url"][:100],
            )
            return result["decoded_url"]
        else:
            logger.debug(
                "google_news_resolve_failed",
                url=url[:80],
                message=result.get("message", "")[:100],
            )
            return None
    except ImportError:
        logger.warning("googlenewsdecoder_not_installed")
        return None
    except Exception as e:
        logger.debug("google_news_resolve_error", url=url[:80], error=str(e))
        return None


class ArticleExtractor:
    """Extracts full article body text from URLs and stores as Markdown."""

    def __init__(
        self,
        timeout: float = 30.0,
        delay: float = 1.0,
    ):
        self.timeout = timeout
        self.delay = delay
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers=_BROWSER_HEADERS,
                follow_redirects=True,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _resolve_url(self, url: str) -> str:
        """Resolve redirects (Google News, etc) to the real article URL."""
        if "news.google.com" in url:
            resolved = _resolve_google_news_url(url)
            if resolved:
                return resolved
        return url

    def extract(self, url: str) -> Optional[str]:
        """Fetch a URL and extract article body as Markdown.

        Uses httpx with browser headers to avoid bot blocking, then
        trafilatura for content extraction. Resolves Google News
        redirects before fetching.

        Returns Markdown text on success, None on failure.
        """
        try:
            fetch_url = self._resolve_url(url)

            response = self.client.get(fetch_url)
            if response.status_code != 200:
                logger.debug(
                    "extractor_http_error",
                    url=fetch_url[:100],
                    status=response.status_code,
                )
                return None

            body = trafilatura.extract(
                response.text,
                output_format="markdown",
                favor_recall=True,
                include_links=False,
                include_images=False,
                include_comments=False,
            )

            if not body or len(body.strip()) < 50:
                logger.debug(
                    "extractor_body_too_short",
                    url=fetch_url[:100],
                    length=len(body) if body else 0,
                )
                return None

            return body.strip()

        except Exception as e:
            logger.warning("extractor_error", url=url[:100], error=str(e))
            return None

    def extract_batch(
        self,
        articles: list[dict],
        os_client: OpenSearchClient,
        index_name: Optional[str] = None,
    ) -> dict:
        """Extract body text for a batch of articles, updating OpenSearch.

        Returns stats dict with counts of processed, success, failed.
        """
        stats = {"processed": 0, "success": 0, "failed": 0}

        for article in articles:
            article_id = article["id"]
            url = article["url"]
            source = article.get("source_name", "unknown")

            body = self.extract(url)

            if body:
                os_client.update_article_body(article_id, body, index_name)
                stats["success"] += 1
                logger.debug(
                    "body_extracted",
                    source=source,
                    url=url[:80],
                    body_length=len(body),
                )
            else:
                os_client.mark_body_extraction_failed(article_id, index_name)
                stats["failed"] += 1

            stats["processed"] += 1

            if self.delay > 0 and stats["processed"] < len(articles):
                time.sleep(self.delay)

        logger.info("extraction_batch_complete", **stats)
        return stats
