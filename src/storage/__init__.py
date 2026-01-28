"""Storage backends for Dorothy."""

from .opensearch import ARTICLE_MAPPING, OpenSearchClient

__all__ = ["OpenSearchClient", "ARTICLE_MAPPING"]
