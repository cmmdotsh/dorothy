"""Full article text extraction via trafilatura (HTML -> Markdown).

Supports parallel extraction across domains — one request per domain
at a time, but many domains concurrently.
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urlparse

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


def _resolve_google_news_url(url: str, timeout: float = 10.0) -> Optional[str]:
    """Resolve a Google News redirect URL to the real article URL."""
    import concurrent.futures

    def _decode():
        from googlenewsdecoder.new_decoderv1 import decode_google_news_url
        return decode_google_news_url(url)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_decode)
            result = future.result(timeout=timeout)

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
    except concurrent.futures.TimeoutError:
        logger.debug("google_news_resolve_timeout", url=url[:80])
        return None
    except ImportError:
        logger.warning("googlenewsdecoder_not_installed")
        return None
    except Exception as e:
        logger.debug("google_news_resolve_error", url=url[:80], error=str(e))
        return None


class ArticleExtractor:
    """Extracts full article body text from URLs and stores as Markdown.

    Supports parallel extraction: groups articles by domain, then runs
    one worker per domain concurrently. Each worker processes its domain's
    articles sequentially with a polite delay between requests.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        delay: float = 1.0,
        max_workers: int = 10,
    ):
        self.timeout = timeout
        self.delay = delay
        self.max_workers = max_workers

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            headers=_BROWSER_HEADERS,
            follow_redirects=True,
            timeout=self.timeout,
        )

    def close(self) -> None:
        pass  # clients are now per-thread

    def _resolve_url(self, url: str) -> str:
        """Resolve redirects (Google News, etc) to the real article URL."""
        if "news.google.com" in url:
            resolved = _resolve_google_news_url(url)
            if resolved:
                return resolved
        return url

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for grouping."""
        try:
            parsed = urlparse(self._resolve_url(url))
            return parsed.netloc or "unknown"
        except Exception:
            return "unknown"

    def extract(self, url: str, client: Optional[httpx.Client] = None) -> Optional[str]:
        """Fetch a URL and extract article body as Markdown.

        Returns Markdown text on success, None on failure.
        """
        own_client = False
        if client is None:
            client = self._make_client()
            own_client = True

        try:
            fetch_url = self._resolve_url(url)

            response = client.get(fetch_url)
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
                favor_precision=True,
                include_links=False,
                include_images=False,
                include_comments=False,
                include_tables=False,
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
        finally:
            if own_client:
                client.close()

    def _extract_domain_batch(
        self,
        domain: str,
        articles: list[dict],
        os_client: OpenSearchClient,
        index_name: Optional[str],
    ) -> dict:
        """Extract articles for a single domain sequentially with delay."""
        stats = {"processed": 0, "success": 0, "failed": 0}
        client = self._make_client()

        try:
            for article in articles:
                article_id = article["id"]
                url = article["url"]
                source = article.get("source_name", "unknown")

                body = self.extract(url, client=client)

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
        finally:
            client.close()

        return stats

    def extract_batch(
        self,
        articles: list[dict],
        os_client: OpenSearchClient,
        index_name: Optional[str] = None,
    ) -> dict:
        """Extract body text for a batch of articles, parallelized by domain.

        Groups articles by domain, then runs one worker per domain
        concurrently. Polite per-domain (sequential with delay), fast
        overall (many domains at once).

        Returns stats dict with counts of processed, success, failed.
        """
        # Group by domain
        by_domain: dict[str, list[dict]] = defaultdict(list)
        for article in articles:
            domain = self._get_domain(article["url"])
            by_domain[domain].append(article)

        logger.info(
            "extraction_parallel_start",
            articles=len(articles),
            domains=len(by_domain),
            workers=min(self.max_workers, len(by_domain)),
        )

        total_stats = {"processed": 0, "success": 0, "failed": 0}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(by_domain))) as pool:
            futures = {
                pool.submit(
                    self._extract_domain_batch, domain, domain_articles, os_client, index_name,
                ): domain
                for domain, domain_articles in by_domain.items()
            }

            for future in as_completed(futures):
                domain = futures[future]
                try:
                    stats = future.result()
                    total_stats["processed"] += stats["processed"]
                    total_stats["success"] += stats["success"]
                    total_stats["failed"] += stats["failed"]
                    logger.info(
                        "domain_extraction_complete",
                        domain=domain,
                        **stats,
                    )
                except Exception as e:
                    logger.error("domain_extraction_error", domain=domain, error=str(e))

        logger.info("extraction_batch_complete", **total_stats)
        return total_stats
