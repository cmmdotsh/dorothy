#!/usr/bin/env python3
"""
Visualize article clusters for a Dorothy column.

Reduces 1024-dim embeddings to 2D with UMAP, colors by HDBSCAN cluster,
and produces an interactive plotly HTML file.

Usage:
    python -m scripts.visualize_clusters                          # politics (default)
    python -m scripts.visualize_clusters --column tech            # different column
    python -m scripts.visualize_clusters --size 500               # limit articles
    python -m scripts.visualize_clusters --output clusters.html   # custom output
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import umap
import structlog
from rich.console import Console

from src.config import config
from src.storage import OpenSearchClient
from src.clustering import StoryGrouper

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()

# Bias → color mapping (matches Dorothy's site)
BIAS_COLORS = {
    "left": "#3b82f6",
    "lean-left": "#60a5fa",
    "center": "#a855f7",
    "lean-right": "#f97316",
    "right": "#ef4444",
}

# Cluster colors — enough for most runs
CLUSTER_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e6beff",
    "#1abc9c", "#d35400", "#2ecc71", "#8e44ad", "#2c3e50",
]
NOISE_COLOR = "#444444"


def visualize(
    column: str,
    size: int,
    output_path: Path,
    min_cluster_size: int = 3,
    color_by: str = "cluster",
) -> None:
    """Fetch articles, cluster, reduce dimensions, and plot."""

    # --- Init ---
    auth_kwargs = {}
    if config.opensearch.username and config.opensearch.password:
        auth_kwargs["username"] = config.opensearch.username
        auth_kwargs["password"] = config.opensearch.password

    os_client = OpenSearchClient(
        host=config.opensearch.host,
        port=config.opensearch.port,
        use_ssl=config.opensearch.use_ssl,
        **auth_kwargs,
    )

    if not os_client.health_check():
        console.print("[red]OpenSearch unavailable[/red]")
        return

    # --- Fetch ---
    console.print(f"[dim]Fetching up to {size} articles for {column}...[/dim]")
    articles = os_client.search_articles(column=column, size=size)
    articles_with_emb = [a for a in articles if a.get("embedding")]
    console.print(f"[green]Got {len(articles_with_emb)} articles with embeddings[/green]")

    if len(articles_with_emb) < min_cluster_size:
        console.print("[red]Not enough articles to cluster[/red]")
        return

    # --- Cluster ---
    console.print("[dim]Running HDBSCAN...[/dim]")
    grouper = StoryGrouper(os_client, min_cluster_size=min_cluster_size, min_samples=2)
    stories = grouper.group_articles(articles_with_emb)

    # Build article_id → (cluster_label, story_headline) mapping
    article_cluster = {}
    for i, story in enumerate(stories):
        for a in story.articles:
            article_cluster[a.get("id")] = {
                "cluster": i,
                "story_headline": story.headline[:80],
                "story_size": len(story.articles),
            }

    # --- UMAP ---
    console.print("[dim]Running UMAP dimensionality reduction...[/dim]")
    embeddings = np.array([a["embedding"] for a in articles_with_emb])

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    console.print("[green]UMAP complete[/green]")

    # --- Build traces ---
    # Separate noise from clustered articles
    cluster_ids = set()
    for a in articles_with_emb:
        info = article_cluster.get(a.get("id"), {})
        cid = info.get("cluster", -1)
        cluster_ids.add(cid)

    # Sort clusters by size (biggest first), separate noise
    cluster_sizes = {}
    for a in articles_with_emb:
        cid = article_cluster.get(a.get("id"), {}).get("cluster", -1)
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

    sorted_clusters = sorted(
        [c for c in cluster_sizes if cluster_sizes[c] > 1],
        key=lambda c: cluster_sizes[c],
        reverse=True,
    )

    fig = go.Figure()

    for idx, cluster_id in enumerate(sorted_clusters):
        mask = []
        for i, a in enumerate(articles_with_emb):
            info = article_cluster.get(a.get("id"), {})
            if info.get("cluster") == cluster_id:
                mask.append(i)

        if not mask:
            continue

        x = coords[mask, 0]
        y = coords[mask, 1]

        # Hover text
        hover = []
        for i in mask:
            a = articles_with_emb[i]
            info = article_cluster.get(a.get("id"), {})
            hover.append(
                f"<b>{a.get('headline', '')[:60]}</b><br>"
                f"Source: {a.get('source_name', '?')} ({a.get('source_bias', '?')})<br>"
                f"Cluster: {info.get('story_headline', '?')}<br>"
                f"Cluster size: {info.get('story_size', '?')}"
            )

        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
        story_label = article_cluster.get(
            articles_with_emb[mask[0]].get("id"), {}
        ).get("story_headline", f"Cluster {cluster_id}")

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=7, color=color, opacity=0.8, line=dict(width=0.5, color="#222")),
            hovertext=hover,
            hoverinfo="text",
            name=f"({len(mask)}) {story_label[:50]}",
        ))

    # Noise points
    noise_mask = []
    for i, a in enumerate(articles_with_emb):
        info = article_cluster.get(a.get("id"), {})
        if info.get("story_size", 1) == 1:
            noise_mask.append(i)

    if noise_mask:
        x = coords[noise_mask, 0]
        y = coords[noise_mask, 1]
        hover = []
        for i in noise_mask:
            a = articles_with_emb[i]
            hover.append(
                f"<b>{a.get('headline', '')[:60]}</b><br>"
                f"Source: {a.get('source_name', '?')} ({a.get('source_bias', '?')})<br>"
                f"<i>Noise (unclustered)</i>"
            )

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=4, color=NOISE_COLOR, opacity=0.3),
            hovertext=hover,
            hoverinfo="text",
            name=f"Noise ({len(noise_mask)})",
        ))

    # --- Layout ---
    n_clusters = len(sorted_clusters)
    n_noise = len(noise_mask)

    fig.update_layout(
        title=dict(
            text=(
                f"Dorothy: {column.title()} Article Clusters<br>"
                f"<sub>{len(articles_with_emb)} articles · {n_clusters} clusters · "
                f"{n_noise} noise · UMAP + HDBSCAN</sub>"
            ),
        ),
        template="plotly_dark",
        width=1400,
        height=900,
        showlegend=True,
        legend=dict(
            title="Stories (by cluster size)",
            font=dict(size=10),
            itemsizing="constant",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=12),
    )

    # --- Save ---
    fig.write_html(str(output_path), include_plotlyjs=True)
    console.print(f"\n[bold green]Saved to {output_path.absolute()}[/bold green]")
    console.print(f"[dim]Open in browser: file://{output_path.absolute()}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Dorothy article clusters")
    parser.add_argument(
        "--column", "-c", default="politics",
        help="Column to visualize (default: politics)",
    )
    parser.add_argument(
        "--size", "-s", type=int, default=2000,
        help="Max articles to fetch (default: 2000)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output HTML path (default: output/clusters-{column}.html)",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=3,
        help="Minimum articles to form a cluster (default: 3)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(f"output/clusters-{args.column}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Dorothy Cluster Visualizer[/bold blue]")
    console.print(f"Column: {args.column} | Max articles: {args.size}")
    console.print()

    visualize(
        column=args.column,
        size=args.size,
        output_path=output_path,
        min_cluster_size=args.min_cluster_size,
    )


if __name__ == "__main__":
    main()
