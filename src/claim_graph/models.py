"""Data structures for the claim graph."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A paragraph-sized passage from an article."""

    article_id: str
    source_name: str
    source_slug: str
    source_bias: str
    source_region: str | None
    source_perspective: str | None
    column: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None


@dataclass
class ClaimCluster:
    """A group of semantically similar chunks from different sources."""

    chunks: list[Chunk]
    representative_text: str
    source_count: int
    source_names: list[str]
    avg_similarity: float


@dataclass
class ClaimGraph:
    """The full claim graph for a story cluster."""

    story_id: str
    corroborated: list[ClaimCluster]
    unique_details: list[Chunk]
    chunk_count: int
    edge_count: int

    def to_prompt_text(self, column: str) -> str:
        """Render the graph as structured text for the synthesis prompt.

        Groups corroborated facts first (sorted by source count descending),
        then unique details grouped by source bias/region/perspective.
        """
        sections: list[str] = []

        if self.corroborated:
            claims_text: list[str] = []
            sorted_claims = sorted(
                self.corroborated, key=lambda c: c.source_count, reverse=True
            )
            for cluster in sorted_claims:
                header = (
                    f"### {cluster.representative_text[:200]}\n"
                    f"Sources: {', '.join(cluster.source_names)} "
                    f"(similarity: {cluster.avg_similarity:.2f})"
                )
                # One chunk per source (first seen = closest to centroid
                # since graph_builder sorts by centroid distance)
                lines = [header]
                seen_sources: set[str] = set()
                for chunk in cluster.chunks:
                    key = chunk.source_slug or chunk.source_name
                    if key in seen_sources:
                        continue
                    seen_sources.add(key)
                    label = self._source_label(chunk, column)
                    lines.append(f"- {label}: \"{chunk.text}\"")
                claims_text.append("\n".join(lines))

            sections.append(
                "## Corroborated Facts\n\n" + "\n\n".join(claims_text)
            )

        if self.unique_details:
            unique_lines: list[str] = []
            for chunk in self.unique_details:
                label = self._source_label(chunk, column)
                unique_lines.append(f"- {label}: \"{chunk.text}\"")
            sections.append(
                "## Unique Details\n\n" + "\n".join(unique_lines)
            )

        return "\n\n".join(sections)

    def to_viz_dict(self) -> dict:
        """Serialize the claim graph for storage and frontend visualization.

        Strips embeddings — only keeps source metadata, text, and cluster structure.
        """
        clusters = []
        for cluster in self.corroborated:
            seen: set[str] = set()
            sources = []
            for chunk in cluster.chunks:
                key = chunk.source_slug or chunk.source_name
                if key in seen:
                    continue
                seen.add(key)
                sources.append({
                    "source_name": chunk.source_name,
                    "source_slug": chunk.source_slug,
                    "source_bias": chunk.source_bias,
                    "text": chunk.text,
                })
            clusters.append({
                "representative_text": cluster.representative_text,
                "source_count": cluster.source_count,
                "source_names": cluster.source_names,
                "avg_similarity": round(cluster.avg_similarity, 3),
                "sources": sources,
            })

        unique = []
        for chunk in self.unique_details:
            unique.append({
                "source_name": chunk.source_name,
                "source_slug": chunk.source_slug,
                "source_bias": chunk.source_bias,
                "text": chunk.text,
            })

        return {
            "corroborated": clusters,
            "unique_details": unique,
            "chunk_count": self.chunk_count,
            "edge_count": self.edge_count,
        }

    @staticmethod
    def _source_label(chunk: Chunk, column: str) -> str:
        """Build a source label with the relevant dimension annotation."""
        name = chunk.source_name
        if column == "sports" and chunk.source_region:
            return f"{name} ({chunk.source_region})"
        elif column == "tech" and chunk.source_perspective:
            return f"{name} ({chunk.source_perspective})"
        else:
            return f"{name} ({chunk.source_bias})"
