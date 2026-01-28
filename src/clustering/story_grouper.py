"""Story grouping using HDBSCAN density clustering."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import structlog

from src.storage import OpenSearchClient

logger = structlog.get_logger(__name__)


@dataclass
class Story:
    """A story is a cluster of related articles from different sources."""

    id: str
    articles: list[dict] = field(default_factory=list)

    @property
    def headline(self) -> str:
        """Get representative headline (from first article)."""
        return self.articles[0]["headline"] if self.articles else ""

    @property
    def source_count(self) -> int:
        """Number of unique sources covering this story."""
        return len(set(a.get("source_slug", "") for a in self.articles))

    @property
    def bias_spread(self) -> dict[str, int]:
        """Count of articles by bias rating."""
        spread: dict[str, int] = defaultdict(int)
        for article in self.articles:
            bias = article.get("source_bias", "unknown")
            spread[bias] += 1
        return dict(spread)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "headline": self.headline,
            "article_count": len(self.articles),
            "source_count": self.source_count,
            "bias_spread": self.bias_spread,
            "articles": self.articles,
        }


class StoryGrouper:
    """
    Groups similar articles into stories using HDBSCAN density clustering.

    Unlike connected-components, HDBSCAN:
    - Does not transitively chain loosely related articles
    - Automatically determines cluster count
    - Handles noise (unclustered articles)
    - Produces tighter, more coherent story clusters
    """

    def __init__(
        self,
        os_client: OpenSearchClient,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        max_cluster_size: int = 30,
        cluster_selection_epsilon: float = 0.0,
        # Legacy params (ignored, kept for backward compatibility)
        similarity_threshold: Optional[float] = None,
        k_neighbors: Optional[int] = None,
    ):
        """
        Initialize the story grouper.

        Args:
            os_client: OpenSearch client for fetching articles
            min_cluster_size: Minimum articles to form a cluster (default: 3)
            min_samples: Core point density threshold (default: 2)
            max_cluster_size: Split clusters larger than this (default: 30)
            cluster_selection_epsilon: Merge nearby clusters threshold (default: 0.0)
            similarity_threshold: Deprecated, kept for backward compatibility
            k_neighbors: Deprecated, kept for backward compatibility
        """
        self.os_client = os_client
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon

        if similarity_threshold is not None or k_neighbors is not None:
            logger.warning(
                "deprecated_params_ignored",
                msg="similarity_threshold and k_neighbors are deprecated, using HDBSCAN params instead",
            )

    def _compute_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine distance matrix.

        Cosine distance = 1 - cosine_similarity
        """
        return cosine_distances(embeddings)

    def _run_hdbscan(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Run HDBSCAN clustering on precomputed distance matrix.

        Returns:
            Array of cluster labels (-1 = noise)
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
            cluster_selection_method="eom",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        return clusterer.fit_predict(distance_matrix)

    def _labels_to_stories(
        self,
        articles: list[dict],
        labels: np.ndarray,
    ) -> list[Story]:
        """
        Convert HDBSCAN labels to Story objects.

        Args:
            articles: List of article dicts
            labels: HDBSCAN cluster labels

        Returns:
            List of Story objects
        """
        clusters: dict[int, list[dict]] = defaultdict(list)
        noise_articles: list[dict] = []

        for article, label in zip(articles, labels):
            if label == -1:
                noise_articles.append(article)
            else:
                clusters[label].append(article)

        stories = []

        # Create stories from clusters
        for label, cluster_articles in clusters.items():
            story = Story(
                id=f"story-{cluster_articles[0]['id'][:8]}",
                articles=cluster_articles,
            )
            stories.append(story)

        # Create single-article stories from noise (won't be synthesized due to source_count < 2)
        for article in noise_articles:
            story = Story(
                id=f"story-{article['id'][:8]}",
                articles=[article],
            )
            stories.append(story)

        return stories

    def _split_large_clusters(self, stories: list[Story]) -> list[Story]:
        """
        Split clusters that exceed max_cluster_size.

        Uses recursive HDBSCAN with tighter parameters.
        """
        result = []

        for story in stories:
            if len(story.articles) <= self.max_cluster_size:
                result.append(story)
                continue

            logger.info(
                "splitting_large_cluster",
                story_id=story.id,
                size=len(story.articles),
            )

            # Extract embeddings for sub-clustering
            embeddings = np.array([a["embedding"] for a in story.articles])
            distances = self._compute_distance_matrix(embeddings)

            # Use tighter min_cluster_size for sub-clustering
            sub_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(3, self.min_cluster_size),
                min_samples=max(2, self.min_samples),
                metric="precomputed",
                cluster_selection_method="eom",
                cluster_selection_epsilon=0.05,  # Slightly merge sub-clusters
            )
            sub_labels = sub_clusterer.fit_predict(distances)

            sub_stories = self._labels_to_stories(story.articles, sub_labels)

            # Recursively split if still too large
            for sub_story in sub_stories:
                if len(sub_story.articles) > self.max_cluster_size:
                    # Give up on further splitting, keep as-is but log warning
                    logger.warning(
                        "cluster_still_large_after_split",
                        story_id=sub_story.id,
                        size=len(sub_story.articles),
                    )
                result.append(sub_story)

        return result

    def group_articles(
        self,
        articles: list[dict],
        index_name: Optional[str] = None,
        column: Optional[str] = None,
    ) -> list[Story]:
        """
        Group articles into stories using HDBSCAN clustering.

        Args:
            articles: List of article dicts (must have 'id' and 'embedding')
            index_name: OpenSearch index name (unused, kept for interface compat)
            column: Column name (unused, kept for interface compat)

        Returns:
            List of Story objects
        """
        if not articles:
            return []

        # Filter to articles with embeddings
        articles_with_embeddings = [a for a in articles if a.get("embedding")]

        if len(articles_with_embeddings) < self.min_cluster_size:
            # Not enough articles for clustering
            return [
                Story(id=f"story-{a['id'][:8]}", articles=[a])
                for a in articles_with_embeddings
            ]

        # Extract embeddings
        embeddings = np.array([a["embedding"] for a in articles_with_embeddings])

        # Compute distance matrix
        logger.debug("computing_distance_matrix", size=len(embeddings))
        distances = self._compute_distance_matrix(embeddings)

        # Run HDBSCAN
        logger.debug("running_hdbscan")
        labels = self._run_hdbscan(distances)

        # Convert to stories
        stories = self._labels_to_stories(articles_with_embeddings, labels)

        # Split oversized clusters
        stories = self._split_large_clusters(stories)

        # Sort by article count (biggest stories first)
        stories.sort(key=lambda s: len(s.articles), reverse=True)

        # Log statistics
        multi_source = sum(1 for s in stories if s.source_count > 1)
        noise_count = sum(1 for s in stories if len(s.articles) == 1)
        cluster_sizes = [len(s.articles) for s in stories if len(s.articles) > 1]

        logger.info(
            "articles_grouped",
            article_count=len(articles_with_embeddings),
            story_count=len(stories),
            multi_source_stories=multi_source,
            noise_articles=noise_count,
            largest_clusters=cluster_sizes[:5] if cluster_sizes else [],
        )

        return stories

    def get_stories_for_column(
        self,
        column: str,
        size: int = 100,
        index_name: Optional[str] = None,
    ) -> list[Story]:
        """
        Get clustered stories for a specific column.

        Args:
            column: Column name to filter by
            size: Maximum number of articles to process
            index_name: OpenSearch index name

        Returns:
            List of Story objects
        """
        articles = self.os_client.search_articles(
            column=column,
            size=size,
            index_name=index_name,
        )

        # Filter to only articles with embeddings
        articles_with_embeddings = [a for a in articles if a.get("embedding")]

        if not articles_with_embeddings:
            logger.warning("no_articles_with_embeddings", column=column)
            return []

        return self.group_articles(articles_with_embeddings, index_name, column)
