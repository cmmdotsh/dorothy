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
METADATA_INDEX = "dorothy-metadata"


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
            "article": {"type": "text"},
            "analysis": {"type": "text"},
            "sources_used": {"type": "keyword"},
            "bias_coverage": {"type": "object", "enabled": True},
            "article_count": {"type": "integer"},
            "source_count": {"type": "integer"},
            "generated_at": {"type": "date"},
            "hero_image_url": {"type": "keyword"},
            "article_urls": {"type": "keyword"},
            "edition": {"type": "integer"},
            "is_current": {"type": "boolean"},
            "superseded_by": {"type": "keyword"},
            "hotness_score": {"type": "float"},
            "median_pub_date": {"type": "date"},
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
        timeout: int = 120,
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
            timeout=timeout,
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
            "article": synthesis.get("article", ""),
            "analysis": synthesis.get("analysis", ""),
            "sources_used": synthesis.get("sources_used", []),
            "bias_coverage": synthesis.get("bias_coverage", {}),
            "article_count": synthesis.get("article_count", 0),
            "source_count": len(synthesis.get("sources_used", [])),
            "generated_at": synthesis.get("generated_at", utcnow().isoformat()),
            "hero_image_url": synthesis.get("hero_image_url"),
            "articles": synthesis.get("articles", []),
            "article_urls": synthesis.get("article_urls", []),
            "edition": synthesis.get("edition", 1),
            "is_current": synthesis.get("is_current", True),
            "superseded_by": synthesis.get("superseded_by"),
            "hotness_score": synthesis.get("hotness_score", 0.0),
            "median_pub_date": synthesis.get("median_pub_date"),
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
                "article": synthesis.get("article", ""),
                "analysis": synthesis.get("analysis", ""),
                "sources_used": synthesis.get("sources_used", []),
                "bias_coverage": synthesis.get("bias_coverage", {}),
                "article_count": synthesis.get("article_count", 0),
                "source_count": len(synthesis.get("sources_used", [])),
                "generated_at": synthesis.get("generated_at", utcnow().isoformat()),
                "hero_image_url": synthesis.get("hero_image_url"),
                "articles": synthesis.get("articles", []),
                "article_urls": synthesis.get("article_urls", []),
                "edition": synthesis.get("edition", 1),
                "is_current": synthesis.get("is_current", True),
                "superseded_by": synthesis.get("superseded_by"),
                "hotness_score": synthesis.get("hotness_score", 0.0),
                "median_pub_date": synthesis.get("median_pub_date"),
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
        include_historical: bool = False,
    ) -> list[dict]:
        """Get synthesized stories, optionally filtered by column.

        By default only returns current (non-superseded) syntheses.
        """
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return []
        except Exception:
            return []

        must_clauses = []
        if column:
            must_clauses.append({"term": {"column": column}})
        if not include_historical:
            # Match docs where is_current=true OR is_current doesn't exist (legacy docs)
            must_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"is_current": True}},
                        {"bool": {"must_not": [{"exists": {"field": "is_current"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            })

        if must_clauses:
            query = {"bool": {"must": must_clauses}}
        else:
            query = {"match_all": {}}

        body = {
            "query": query,
            "size": limit,
            "sort": [
                {"hotness_score": {"order": "desc", "missing": "_last"}},
                {"generated_at": {"order": "desc"}},
                {"article_count": {"order": "desc"}},
            ],
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
        """Get count of current syntheses, optionally by column."""
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return 0

            must_clauses = [{
                "bool": {
                    "should": [
                        {"term": {"is_current": True}},
                        {"bool": {"must_not": [{"exists": {"field": "is_current"}}]}},
                    ],
                    "minimum_should_match": 1,
                }
            }]
            if column:
                must_clauses.append({"term": {"column": column}})

            body = {"query": {"bool": {"must": must_clauses}}}
            response = self.client.count(index=SYNTHESIS_INDEX, body=body)
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

    # Synthesis deduplication methods

    def find_overlapping_synthesis(self, article_urls: list[str]) -> Optional[dict]:
        """Find an existing current synthesis whose articles overlap significantly.

        Uses a terms query on article_urls, then computes Jaccard similarity
        in Python. Returns the best match (highest overlap) or None.
        """
        try:
            if not self.client.indices.exists(index=SYNTHESIS_INDEX):
                return None

            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"article_urls": article_urls}},
                            {"term": {"is_current": True}},
                        ]
                    }
                },
                "size": 10,
                "_source": ["story_id", "article_urls", "article_count", "generated_headline"],
            }

            response = self.client.search(index=SYNTHESIS_INDEX, body=body)
            hits = response["hits"]["hits"]

            if not hits:
                return None

            new_set = set(article_urls)
            best_match = None
            best_jaccard = 0.0

            for hit in hits:
                existing_urls = set(hit["_source"].get("article_urls", []))
                if not existing_urls:
                    continue
                intersection = len(new_set & existing_urls)
                union = len(new_set | existing_urls)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_match = {**hit["_source"], "jaccard": jaccard}

            return best_match

        except Exception as e:
            logger.error("find_overlapping_synthesis_failed", error=str(e))
            return None

    def mark_synthesis_historical(self, story_id: str, superseded_by: str) -> bool:
        """Mark an existing synthesis as historical (superseded by a newer version)."""
        try:
            self.client.update(
                index=SYNTHESIS_INDEX,
                id=story_id,
                body={"doc": {"is_current": False, "superseded_by": superseded_by}},
            )
            logger.info("synthesis_marked_historical", story_id=story_id, superseded_by=superseded_by)
            return True
        except Exception as e:
            logger.error("mark_historical_failed", story_id=story_id, error=str(e))
            return False

    # Edition counter methods

    def _ensure_metadata_index(self) -> None:
        """Create metadata index if it doesn't exist."""
        if not self.client.indices.exists(index=METADATA_INDEX):
            self.client.indices.create(
                index=METADATA_INDEX,
                body={
                    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
                    "mappings": {
                        "properties": {
                            "edition": {"type": "integer"},
                            "updated_at": {"type": "date"},
                        }
                    },
                },
            )

    def get_edition(self) -> int:
        """Get the current edition (pipeline run counter)."""
        try:
            self._ensure_metadata_index()
            response = self.client.get(index=METADATA_INDEX, id="edition_counter")
            return response["_source"].get("edition", 0)
        except Exception:
            return 0

    def increment_edition(self) -> int:
        """Increment and return the new edition number."""
        try:
            self._ensure_metadata_index()
            current = self.get_edition()
            new_edition = current + 1
            self.client.index(
                index=METADATA_INDEX,
                id="edition_counter",
                body={"edition": new_edition, "updated_at": utcnow().isoformat()},
            )
            logger.info("edition_incremented", edition=new_edition)
            return new_edition
        except Exception as e:
            logger.error("increment_edition_failed", error=str(e))
            return 1
