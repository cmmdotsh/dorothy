"""Build claim graphs from story clusters.

Takes a Story (cluster of articles), chunks the bodies, embeds the chunks,
builds a cosine similarity graph, and groups high-similarity cross-source
chunks into claim clusters.
"""

import difflib
import re

import numpy as np
import structlog
from sklearn.metrics.pairwise import cosine_distances

from src.claim_graph.chunker import chunk_story
from src.claim_graph.embedder import embed_chunks_sync
from src.claim_graph.models import Chunk, ClaimCluster, ClaimGraph
from src.clustering import Story

logger = structlog.get_logger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize for syndication comparison: casefold, unify quotes, squash
    punctuation/whitespace so wire copies differing only in glyphs compare equal."""
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text.casefold())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text)).strip()


def count_independent_sources(chunks: list[Chunk]) -> int:
    """Count sources that INDEPENDENTLY state a claim.

    Syndicated wire copy (the same AP text republished by several outlets)
    is one source, not corroboration: chunks whose normalized texts are
    near-verbatim (SequenceMatcher ratio >= 0.85) collapse into one wording
    group. Corroboration = number of distinct wordings, capped by the number
    of distinct articles. Measured 2026-08-16: 17% of corroboration pairs
    were verbatim wire copies at cosine ~1.0.
    """
    texts = [_normalize_text(c.text)[:600] for c in chunks]
    wording_groups: list[int] = []  # representative index per group
    for i, text in enumerate(texts):
        for rep in wording_groups:
            if difflib.SequenceMatcher(None, text, texts[rep]).ratio() >= 0.85:
                break
        else:
            wording_groups.append(i)
    distinct_articles = len(set(c.article_id for c in chunks))
    independent = min(len(wording_groups), distinct_articles)
    if independent < distinct_articles:
        logger.debug(
            "syndicated_copy_collapsed",
            chunks=len(chunks),
            articles=distinct_articles,
            independent=independent,
        )
    return independent


class ClaimGraphBuilder:
    """Builds claim graphs for story clusters."""

    def __init__(
        self,
        base_url: str,
        model: str,
        similarity_threshold: float = 0.75,
        min_sources_corroborated: int = 2,
        embedding_concurrency: int = 8,
        min_chunk_chars: int = 80,
        max_chunk_chars: int = 800,
    ):
        self.base_url = base_url
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.min_sources_corroborated = min_sources_corroborated
        self.embedding_concurrency = embedding_concurrency
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def build(self, story: Story) -> ClaimGraph:
        """Build a claim graph for a story cluster."""
        # 1. Chunk all articles
        chunks = chunk_story(
            story.articles,
            min_chars=self.min_chunk_chars,
            max_chars=self.max_chunk_chars,
        )

        if len(chunks) < 2:
            return ClaimGraph(
                story_id=story.id,
                corroborated=[],
                unique_details=chunks,
                chunk_count=len(chunks),
                edge_count=0,
            )

        # 2. Embed all chunks
        chunks = embed_chunks_sync(
            chunks,
            base_url=self.base_url,
            model=self.model,
            concurrency=self.embedding_concurrency,
        )

        # Filter to chunks that got embeddings
        embedded = [c for c in chunks if c.embedding is not None]
        unembedded = [c for c in chunks if c.embedding is None]

        if len(embedded) < 2:
            return ClaimGraph(
                story_id=story.id,
                corroborated=[],
                unique_details=chunks,
                chunk_count=len(chunks),
                edge_count=0,
            )

        # 3. Build similarity matrix
        emb_matrix = np.array([c.embedding for c in embedded])
        dist_matrix = cosine_distances(emb_matrix)
        sim_matrix = 1.0 - dist_matrix

        # 4. Find cross-source edges above threshold
        n = len(embedded)
        edges: list[tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if embedded[i].article_id == embedded[j].article_id:
                    continue
                sim = float(sim_matrix[i][j])
                if sim >= self.similarity_threshold:
                    edges.append((i, j, sim))

        # 5. Union-find to group connected chunks into claim clusters
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, j, _ in edges:
            union(i, j)

        # Group by root
        groups: dict[int, list[int]] = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        # 6. Build claim clusters and unique details
        corroborated: list[ClaimCluster] = []
        unique_details: list[Chunk] = list(unembedded)  # unembedded are always unique

        for indices in groups.values():
            group_chunks = [embedded[i] for i in indices]
            source_names = list(dict.fromkeys(c.source_name for c in group_chunks))
            unique_sources = count_independent_sources(group_chunks)

            if unique_sources < self.min_sources_corroborated:
                unique_details.extend(group_chunks)
                continue

            # Find representative chunk (closest to group centroid)
            group_embs = np.array([c.embedding for c in group_chunks])
            centroid = group_embs.mean(axis=0, keepdims=True)
            dists = cosine_distances(centroid, group_embs)[0]
            rep_idx = int(np.argmin(dists))

            # Average similarity among edges in this group
            group_set = set(indices)
            group_edges = [
                sim for i, j, sim in edges
                if i in group_set and j in group_set
            ]
            avg_sim = sum(group_edges) / len(group_edges) if group_edges else 0.0

            # Sort chunks by centroid distance so prompt dedup picks best per source
            sorted_indices = np.argsort(dists)
            group_chunks = [group_chunks[i] for i in sorted_indices]

            corroborated.append(
                ClaimCluster(
                    chunks=group_chunks,
                    representative_text=group_chunks[0].text,
                    source_count=len(source_names),
                    source_names=source_names,
                    avg_similarity=round(avg_sim, 4),
                )
            )

        graph = ClaimGraph(
            story_id=story.id,
            corroborated=corroborated,
            unique_details=unique_details,
            chunk_count=len(chunks),
            edge_count=len(edges),
        )

        logger.info(
            "claim_graph_built",
            story_id=story.id,
            chunks=graph.chunk_count,
            edges=graph.edge_count,
            corroborated_clusters=len(graph.corroborated),
            unique_details=len(graph.unique_details),
        )

        return graph
