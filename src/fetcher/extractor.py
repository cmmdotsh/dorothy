"""Full article text extraction via trafilatura (HTML -> Markdown)."""

import time
from typing import Optional

import structlog
import trafilatura

from src.storage import OpenSearchClient

logger = structlog.get_logger(__name__)


class ArticleExtractor:
    """Extracts full article body text from URLs and stores as Markdown."""

    def __init__(
        self,
        timeout: float = 30.0,
        delay: float = 1.0,
        user_agent: str = "Dorothy/0.1 (news aggregator)",
    ):
        self.timeout = timeout
        self.delay = delay
        self.user_agent = user_agent

    def extract(self, url: str) -> Optional[str]:
        """Fetch a URL and extract article body as Markdown.

        Returns Markdown text on success, None on failure.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                logger.debug("extractor_fetch_empty", url=url)
                return None

            body = trafilatura.extract(
                downloaded,
                output_format="markdown",
                favor_precision=True,
                include_links=False,
                include_images=False,
                include_comments=False,
            )

            if not body or len(body.strip()) < 50:
                logger.debug("extractor_body_too_short", url=url, length=len(body) if body else 0)
                return None

            return body.strip()

        except Exception as e:
            logger.warning("extractor_error", url=url, error=str(e))
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
