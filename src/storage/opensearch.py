"""OpenSearch storage client for Dorothy."""

import os
from datetime import datetime, timezone
from typing import Optional

from opensearchpy import OpenSearch, helpers
import structlog

from src.models import Article

logger = structlog.get_logger(__name__)

INDEX_PREFIX = "dorothy-articles"
SYNTHESIS_INDEX = "dorothy-synthesis"


def utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)

SYNTHESIS_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
    "mappings": {
        "properties": {
            "story_id": {"type": "keyword"},
            "column": {"type": "keyword"},
            "original_headline": {"type": "text"},
            "generated_headline": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 500}},
            },
            "summary": {"type": "text"},
            "sources_used": {"type": "keyword"},
            "bias_coverage": {"type": "object", "enabled": True},
            "article_count": {"type": "integer"},
            "source_count": {"type": "integer"},
            "generated_at": {"type": "date"},
            "hero_image_url": {"type": "keyword"},
            "articles": {
                "type": "nested",
                "properties": {
                    "url": {"type": "keyword"},
                    "headline": {"type": "text"},
                    "source_name": {"type": "keyword"},
                    "source_slug": {"type": "keyword"},
                    "source_bias": {"type": "keyword"},
                    "image_url": {"type": "keyword"},
                },
            },
        }
    },
}

ARTICLE_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "5s",
            "knn": True,
        }
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "source_name": {"type": "keyword"},
            "source_slug": {"type": "keyword"},
            "source_bias": {"type": "keyword"},
            "column": {"type": "keyword"},
            "headline": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 500}},
            },
            "summary": {"type": "text", "analyzer": "standard"},
            "url": {"type": "keyword"},
            "pub_date": {"type": "date"},
            "fetched_at": {"type": "date"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
        }
    },
}


class OpenSearchClient:
    """OpenSearch client for article storage."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        verify_certs: bool = False,
    ):
        self.host = host
        self.port = port

        # Only use auth if explicitly provided
        http_auth = None
        if username and password:
            http_auth = (username, password)
        elif os.getenv("OPENSEARCH_USERNAME") and os.getenv("OPENSEARCH_PASSWORD"):
            http_auth = (os.getenv("OPENSEARCH_USERNAME"), os.getenv("OPENSEARCH_PASSWORD"))

        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=http_auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,
        )

    def health_check(self) -> bool:
        """Check if OpenSearch is reachable."""
        try:
            info = self.client.info()
            logger.info("opensearch_connected", version=info["version"]["number"])
            return True
        except Exception as e:
            logger.error("opensearch_connection_failed", error=str(e))
            return False

    def get_current_index_name(self) -> str:
        """Get index name for current month."""
        return f"{INDEX_PREFIX}-{utcnow().strftime('%Y-%m')}"

    def ensure_index(self, index_name: Optional[str] = None) -> str:
        """Create index if it doesn't exist. Returns the index name."""
        if index_name is None:
            index_name = self.get_current_index_name()

        if not self.client.indices.exists(index=index_name):
            logger.info("creating_index", index=index_name)
            self.client.indices.create(index=index_name, body=ARTICLE_MAPPING)
            logger.info("index_created", index=index_name)
        else:
            logger.debug("index_exists", index=index_name)

        return index_name

    def index_article(self, article: Article, index_name: Optional[str] = None) -> bool:
        """Index a single article."""
        if index_name is None:
            index_name = self.get_current_index_name()

        try:
            self.client.index(
                index=index_name,
                id=str(article.id),
                body=article.to_opensearch_doc(),
            )
            return True
        except Exception as e:
            logger.error("index_article_failed", article_id=str(article.id), error=str(e))
            return False

    def bulk_index_articles(
        self, articles: list[Article], index_name: Optional[str] = None
    ) -> tuple[int, int]:
        """Bulk index multiple articles. Returns (success_count, error_count)."""
        if index_name is None:
            index_name = self.get_current_index_name()

        if not articles:
            return (0, 0)

        actions = [
            {
                "_index": index_name,
                "_id": str(article.id),
                "_source": article.to_opensearch_doc(),
            }
            for article in articles
        ]

        success, errors = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            stats_only=False,
        )

        error_count = len(errors) if isinstance(errors, list) else 0

        logger.info("bulk_index_complete", success=success, errors=error_count, index=index_name)

        return (success, error_count)

    def article_exists(self, url: str, index_name: Optional[str] = None) -> bool:
        """Check if article with URL already exists (deduplication)."""
        if index_name is None:
            index_name = self.get_current_index_name()

        try:
            response = self.client.search(
                index=index_name,
                body={
                    "query": {"term": {"url": url}},
                    "size": 1,
                    "_source": False,
                },
            )
            return response["hits"]["total"]["value"] > 0
        except Exception:
            return False

    def get_article_count(self, index_name: Optional[str] = None) -> int:
        """Get total article count in index."""
        if index_name is None:
            index_name = self.get_current_index_name()

        try:
            response = self.client.count(index=index_name)
            return response["count"]
        except Exception:
            return 0

    def search_articles(
        self,
        column: Optional[str] = None,
        bias: Optional[str] = None,
        query: Optional[str] = None,
        since: Optional[datetime] = None,
        size: int = 100,
        index_name: Optional[str] = None,
    ) -> list[dict]:
        """Search articles with optional filters."""
        if index_name is None:
            index_name = self.get_current_index_name()

        must_clauses = []

        if column:
            must_clauses.append({"term": {"column": column}})
        if bias:
            must_clauses.append({"term": {"source_bias": bias}})
        if query:
            must_clauses.append({"match": {"headline": query}})
        if since:
            must_clauses.append({"range": {"pub_date": {"gte": since.isoformat()}}})

        body = {
            "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
            "size": size,
            "sort": [{"pub_date": {"order": "desc"}}],
        }

        try:
            response = self.client.search(index=index_name, body=body)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error("search_failed", error=str(e))
            return []

    def get_articles_without_embeddings(
        self, size: int = 100, index_name: Optional[str] = None
    ) -> list[dict]:
        """Get articles that don't have embeddings yet."""
        if index_name is None:
            index_name = self.get_current_index_name()

        body = {
            "query": {"bool": {"must_not": [{"exists": {"field": "embedding"}}]}},
            "size": size,
            "sort": [{"pub_date": {"order": "desc"}}],
        }

        try:
            response = self.client.search(index=index_name, body=body)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error("get_articles_without_embeddings_failed", error=str(e))
            return []

    def update_article_embedding(
        self, article_id: str, embedding: list[float], index_name: Optional[str] = None
    ) -> bool:
        """Update an article's embedding."""
        if index_name is None:
            index_name = self.get_current_index_name()

        try:
            self.client.update(
                index=index_name,
                id=article_id,
                body={"doc": {"embedding": embedding}},
            )
            return True
        except Exception as e:
            logger.error("update_embedding_failed", article_id=article_id, error=str(e))
            return False

    def bulk_update_embeddings(
        self,
        updates: list[tuple[str, list[float]]],
        index_name: Optional[str] = None,
    ) -> tuple[int, int]:
        """Bulk update embeddings. Updates is a list of (article_id, embedding) tuples."""
        if index_name is None:
            index_name = self.get_current_index_name()

        if not updates:
            return (0, 0)

        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": article_id,
                "doc": {"embedding": embedding},
            }
            for article_id, embedding in updates
        ]

        success, errors = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            stats_only=False,
        )

        error_count = len(errors) if isinstance(errors, list) else 0
        logger.info("bulk_update_embeddings_complete", success=success, errors=error_count)
        return (success, error_count)

    def knn_search(
        self,
        embedding: list[float],
        k: int = 10,
        index_name: Optional[str] = None,
        column: Optional[str] = None,
    ) -> list[dict]:
        """Find k most similar articles using k-NN search.

        Args:
            embedding: Query embedding vector
            k: Number of neighbors to retrieve
            index_name: OpenSearch index name
            column: If provided, only search within this column (pre-filter)
        """
        if index_name is None:
            index_name = self.get_current_index_name()

        if column:
            # Use script_score for filtered k-NN (NMSLIB doesn't support filter param)
            # This filters first, then computes cosine similarity scores
            body = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"term": {"column": column}},
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": "embedding",
                                "query_value": embedding,
                                "space_type": "cosinesimil",
                            },
                        },
                    }
                },
            }
        else:
            # Standard k-NN query without filtering
            body = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": k,
                        }
                    }
                },
            }

        try:
            response = self.client.search(index=index_name, body=body)
            return [
                {"score": hit["_score"], **hit["_source"]}
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.error("knn_search_failed", error=str(e))
            return []

    # Synthesis storage methods

    def ensure_synthesis_index(self) -> str:
        """Create synthesis index if it doesn't exist."""
        if not self.client.indices.exists(index=SYNTHESIS_INDEX):
            logger.info("creating_synthesis_index", index=SYNTHESIS_INDEX)
            self.client.indices.create(index=SYNTHESIS_INDEX, body=SYNTHESIS_MAPPING)
            logger.info("synthesis_index_created", index=SYNTHESIS_INDEX)
        return SYNTHESIS_INDEX

    def store_synthesis(self, synthesis: dict, column: str) -> bool:
        """Store a synthesized story."""
        self.ensure_synthesis_index()

        doc = {
            "story_id": synthesis["story_id"],
            "column": column,
            "original_headline": synthesis.get("original_headline", ""),
            "generated_headline": synthesis.get("generated_headline", ""),
            "summary": synthesis.get("summary", ""),
            "sources_used": synthesis.get("sources_used", []),
            "bias_coverage": synthesis.get("bias_coverage", {}),
            "article_count": synthesis.get("article_count", 0),
            "source_count": len(synthesis.get("sources_used", [])),
            "generated_at": synthesis.get("generated_at", utcnow().isoformat()),
            "hero_image_url": synthesis.get("hero_image_url"),
            "articles": synthesis.get("articles", []),
        }

        try:
            self.client.index(
                index=SYNTHESIS_INDEX,
                id=synthesis["story_id"],
                body=doc,
            )
            logger.info("synthesis_stored", story_id=synthesis["story_id"], column=column)
            return True
        except Exception as e:
            logger.error("store_synthesis_failed", story_id=synthesis["story_id"], error=str(e))
            return False

    def bulk_store_syntheses(self, syntheses: list[dict], column: str) -> tuple[int, int]:
        """Bulk store synthesized stories. Returns (success_count, error_count)."""
        self.ensure_synthesis_index()

        if not syntheses:
            return (0, 0)

        actions = []
        for synthesis in syntheses:
            doc = {
                "story_id": synthesis["story_id"],
                "column": column,
                "original_headline": synthesis.get("original_headline", ""),
                "generated_headline": synthesis.get("generated_headline", ""),
                "summary": synthesis.get("summary", ""),
                "sources_used": synthesis.get("sources_used", []),
                "bias_coverage": synthesis.get("bias_coverage", {}),
                "article_count": synthesis.get("article_count", 0),
                "source_count": len(synthesis.get("sources_used", [])),
                "generated_at": synthesis.get("generated_at", utcnow().isoformat()),
                "hero_image_url": synthesis.get("hero_image_url"),
                "articles": synthesis.get("articles", []),
            }
            actions.append({
                "_index": SYNTHESIS_INDEX,
                "_id": synthesis["story_id"],
                "_source": doc,
            })

        success, errors = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            stats_only=False,
        )

        error_count = len(errors) if isinstance(errors, list) else 0
        logger.info("bulk_store_syntheses_complete", success=success, errors=error_count)
        return (success, error_count)

    def get_syntheses(
        self,
        column: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get synthesized stories, optionally filtered by column."""
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return []
        except Exception:
            return []

        query = {"term": {"column": column}} if column else {"match_all": {}}

        body = {
            "query": query,
            "size": limit,
            "sort": [{"generated_at": {"order": "desc"}}],
        }

        try:
            response = self.client.search(index=SYNTHESIS_INDEX, body=body)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error("get_syntheses_failed", error=str(e))
            return []

    def get_synthesis_by_id(self, story_id: str) -> Optional[dict]:
        """Get a single synthesis by story ID."""
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return None

            response = self.client.get(index=SYNTHESIS_INDEX, id=story_id)
            return response["_source"]
        except Exception:
            return None

    def get_synthesis_count(self, column: Optional[str] = None) -> int:
        """Get count of syntheses, optionally by column."""
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return 0

            if column:
                body = {"query": {"term": {"column": column}}}
                response = self.client.count(index=SYNTHESIS_INDEX, body=body)
            else:
                response = self.client.count(index=SYNTHESIS_INDEX)
            return response["count"]
        except Exception:
            return 0

    def clear_syntheses(self, column: Optional[str] = None) -> int:
        """Delete syntheses, optionally by column. Returns count deleted."""
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return 0

            if column:
                body = {"query": {"term": {"column": column}}}
            else:
                body = {"query": {"match_all": {}}}

            response = self.client.delete_by_query(index=SYNTHESIS_INDEX, body=body)
            deleted = response.get("deleted", 0)
            logger.info("syntheses_cleared", column=column, deleted=deleted)
            return deleted
        except Exception as e:
            logger.error("clear_syntheses_failed", error=str(e))
            return 0
