"""Tests for extractive article assembly."""

from src.synthesis.assembler import assemble_article


def _make_claim_graph(clusters, unique_details=None):
    """Build a minimal claim_graph viz dict for testing."""
    return {
        "corroborated": clusters,
        "unique_details": unique_details or [],
        "chunk_count": sum(len(c["sources"]) for c in clusters),
        "edge_count": 0,
    }


def test_assemble_basic_ordering():
    graph = _make_claim_graph([
        {
            "representative_text": "The president signed the bill into law.",
            "source_count": 3,
            "source_names": ["AP", "Reuters", "BBC"],
            "avg_similarity": 0.9,
            "sources": [
                {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "President signed the bill Tuesday."},
                {"source_name": "Reuters", "source_slug": "reuters", "source_bias": "center", "text": "The legislation was signed into law."},
                {"source_name": "BBC", "source_slug": "bbc", "source_bias": "center", "text": "Bill signed by president on Tuesday."},
            ],
        },
        {
            "representative_text": "Opposition lawmakers criticized the move.",
            "source_count": 2,
            "source_names": ["NYT", "Fox News"],
            "avg_similarity": 0.85,
            "sources": [
                {"source_name": "NYT", "source_slug": "nyt", "source_bias": "lean-left", "text": "Democrats in the Senate objected."},
                {"source_name": "Fox News", "source_slug": "foxnews", "source_bias": "lean-right", "text": "Republican leaders praised the decision."},
            ],
        },
    ])

    ordering = {
        "headline": "President Signs Bill Into Law",
        "ordering": [
            {"cluster": 0, "transition": ""},
            {"cluster": 1, "transition": "The decision drew mixed reactions."},
        ],
    }

    article = assemble_article(graph, ordering)
    assert article.startswith("President signed the bill Tuesday.")
    assert "— *AP*" in article
    assert "The decision drew mixed reactions." in article
    assert "Democrats in the Senate objected." in article
    assert "— *NYT*" in article


def test_assemble_with_unique_details():
    graph = _make_claim_graph(
        clusters=[{
            "representative_text": "Markets fell sharply.",
            "source_count": 2,
            "source_names": ["AP", "Reuters"],
            "avg_similarity": 0.88,
            "sources": [
                {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "The S&P 500 dropped 3%."},
                {"source_name": "Reuters", "source_slug": "reuters", "source_bias": "center", "text": "Markets plunged on the news."},
            ],
        }],
        unique_details=[
            {"source_name": "The Intercept", "source_slug": "intercept", "source_bias": "left", "text": "Internal documents show the policy was drafted months ago."},
        ],
    )

    ordering = {
        "headline": "Markets Drop",
        "ordering": [{"cluster": 0, "transition": ""}],
    }

    article = assemble_article(graph, ordering)
    assert "The S&P 500 dropped 3%." in article
    assert "Internal documents show" in article
    assert "The Intercept" in article


def test_assemble_skips_invalid_cluster_index():
    graph = _make_claim_graph([{
        "representative_text": "Something happened.",
        "source_count": 2,
        "source_names": ["AP", "BBC"],
        "avg_similarity": 0.9,
        "sources": [
            {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "An event occurred."},
        ],
    }])

    ordering = {
        "headline": "Test",
        "ordering": [
            {"cluster": 0, "transition": ""},
            {"cluster": 99, "transition": "This cluster does not exist."},
        ],
    }

    article = assemble_article(graph, ordering)
    assert "An event occurred." in article
    assert "This cluster does not exist." not in article


def test_assemble_picks_best_source_per_cluster():
    graph = _make_claim_graph([{
        "representative_text": "Lead text.",
        "source_count": 2,
        "source_names": ["AP", "Fox News"],
        "avg_similarity": 0.9,
        "sources": [
            {"source_name": "AP", "source_slug": "ap", "source_bias": "center", "text": "The definitive AP version of events."},
            {"source_name": "Fox News", "source_slug": "foxnews", "source_bias": "lean-right", "text": "Fox take on it."},
        ],
    }])

    ordering = {
        "headline": "Test",
        "ordering": [{"cluster": 0, "transition": ""}],
    }

    article = assemble_article(graph, ordering)
    assert "The definitive AP version" in article
