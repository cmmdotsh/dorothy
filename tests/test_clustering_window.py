"""Recency window + per-source cap for clustering input."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.clustering.story_grouper import StoryGrouper


def _article(source, pub_date, i):
    return {"id": f"a{i}", "source_slug": source, "pub_date": pub_date.isoformat(),
            "embedding": [0.1] * 8, "column": "lifestyle"}


def test_cap_per_source_keeps_newest():
    now = datetime.now(timezone.utc)
    arts = [_article("npr-lifekit", now - timedelta(hours=h), h) for h in range(10)]
    arts += [_article("fox-lifestyle", now - timedelta(hours=h), 100 + h) for h in range(3)]
    capped = StoryGrouper._cap_per_source(arts, max_per_source=5)
    npr = [a for a in capped if a["source_slug"] == "npr-lifekit"]
    assert len(npr) == 5
    # newest 5 kept (hours 0-4)
    assert {a["id"] for a in npr} == {"a0", "a1", "a2", "a3", "a4"}
    assert len([a for a in capped if a["source_slug"] == "fox-lifestyle"]) == 3


def test_get_stories_passes_since_window():
    client = MagicMock()
    client.search_articles.return_value = []
    client.get_current_index_name.return_value = "dorothy-articles-2026-08"
    g = StoryGrouper(client, window_hours=72)
    g.get_stories_for_column("politics", size=2000)
    kwargs = client.search_articles.call_args.kwargs
    assert kwargs["since"] is not None
    age = datetime.now(timezone.utc) - kwargs["since"]
    assert timedelta(hours=71) < age < timedelta(hours=73)


def test_month_boundary_queries_both_indices():
    client = MagicMock()
    client.search_articles.return_value = []
    client.get_current_index_name.return_value = "dorothy-articles-2026-08"
    g = StoryGrouper(client, window_hours=72)
    # 2026-08-02T00:00Z minus 72h crosses into July
    g.get_stories_for_column("politics", size=2000,
                             now=datetime(2026, 8, 2, tzinfo=timezone.utc))
    kwargs = client.search_articles.call_args.kwargs
    assert kwargs["index_name"] == ["dorothy-articles-2026-07", "dorothy-articles-2026-08"]


def test_mid_month_queries_single_index():
    client = MagicMock()
    client.search_articles.return_value = []
    client.get_current_index_name.return_value = "dorothy-articles-2026-08"
    g = StoryGrouper(client, window_hours=72)
    g.get_stories_for_column("politics", size=2000,
                             now=datetime(2026, 8, 16, tzinfo=timezone.utc))
    kwargs = client.search_articles.call_args.kwargs
    assert kwargs["index_name"] is None  # default = current index


def test_search_articles_returns_hits():
    """Regression: the multi-index refactor once dropped the success return."""
    from src.storage.opensearch import OpenSearchClient
    client = OpenSearchClient.__new__(OpenSearchClient)
    client.client = MagicMock()
    client.client.search.return_value = {"hits": {"hits": [{"_source": {"id": "a1"}}]}}
    got = client.search_articles(column="politics", index_name="dorothy-articles-2026-08")
    assert got == [{"id": "a1"}]
