"""Batch embedding generation for articles."""

from typing import Iterator, Optional

import structlog

from src.embeddings.client import EmbeddingClient, EmbeddingError
from src.storage import OpenSearchClient

logger = structlog.get_logger(__name__)


def _prepare_text(article: dict, max_chars: int = 500) -> str:
    """
    Prepare article text for embedding.

    Concatenates headline and summary, truncated to fit the embedding model's
    context window. mxbai-embed-large has a 512 token limit (~1800 chars) but
    Ollama counts the entire batch against context, so we keep each input short.
    """
    headline = article.get("headline", "")
    summary = article.get("summary", "") or ""

    if summary:
        text = f"{headline}\n\n{summary}"
    else:
        text = headline
    return text[:max_chars]


def generate_embeddings(
    os_client: OpenSearchClient,
    embed_client: EmbeddingClient,
    batch_size: int = 32,
    limit: Optional[int] = None,
    index_name: Optional[str] = None,
) -> Iterator[dict]:
    """
    Generate embeddings for articles that don't have them.

    Args:
        os_client: OpenSearch client
        embed_client: Embedding client
        batch_size: Number of articles to process at once
        limit: Maximum number of articles to process (None for all)
        index_name: OpenSearch index name (uses current month if None)

    Yields:
        Stats dict after each batch
    """
    if index_name is None:
        index_name = os_client.get_current_index_name()

    total_processed = 0
    total_success = 0
    total_errors = 0

    while True:
        articles = os_client.get_articles_without_embeddings(
            size=batch_size, index_name=index_name
        )

        if not articles:
            logger.info("no_more_articles_without_embeddings")
            break

        if limit and total_processed >= limit:
            break

        texts = [_prepare_text(a) for a in articles]
        article_ids = [a["id"] for a in articles]

        try:
            embeddings = embed_client.embed(texts)

            updates = list(zip(article_ids, embeddings))
            success, errors = os_client.bulk_update_embeddings(updates, index_name)

            total_processed += len(articles)
            total_success += success
            total_errors += errors

            batch_stats = {
                "batch_size": len(articles),
                "batch_success": success,
                "batch_errors": errors,
                "total_processed": total_processed,
                "total_success": total_success,
                "total_errors": total_errors,
            }

            logger.info("embedding_batch_complete", **batch_stats)
            yield batch_stats

        except EmbeddingError as e:
            logger.warning("embedding_batch_failed_trying_individually", error=str(e), batch_size=len(articles))
            # Fall back to embedding one at a time to skip bad articles
            batch_success = 0
            batch_errors = 0
            for article_id, text in zip(article_ids, texts):
                try:
                    single_embedding = embed_client.embed([text])
                    os_client.bulk_update_embeddings([(article_id, single_embedding[0])], index_name)
                    batch_success += 1
                except EmbeddingError:
                    # Store a zero-vector so this article isn't retried forever
                    zero_vec = [0.0] * 1024
                    os_client.bulk_update_embeddings([(article_id, zero_vec)], index_name)
                    batch_errors += 1

            total_processed += len(articles)
            total_success += batch_success
            total_errors += batch_errors

            yield {
                "batch_size": len(articles),
                "batch_success": batch_success,
                "batch_errors": batch_errors,
                "total_processed": total_processed,
                "total_success": total_success,
                "total_errors": total_errors,
            }

        if limit and total_processed >= limit:
            break

    logger.info(
        "embedding_generation_complete",
        total_processed=total_processed,
        total_success=total_success,
        total_errors=total_errors,
    )


def generate_embeddings_for_articles(
    articles: list[dict],
    embed_client: EmbeddingClient,
) -> list[tuple[str, list[float]]]:
    """
    Generate embeddings for a specific list of articles.

    Args:
        articles: List of article dicts (must have 'id', 'headline', optionally 'summary')
        embed_client: Embedding client

    Returns:
        List of (article_id, embedding) tuples
    """
    if not articles:
        return []

    texts = [_prepare_text(a) for a in articles]
    article_ids = [a["id"] for a in articles]

    try:
        embeddings = embed_client.embed(texts)
        return list(zip(article_ids, embeddings))
    except EmbeddingError as e:
        logger.error("generate_embeddings_for_articles_failed", error=str(e))
        return []
