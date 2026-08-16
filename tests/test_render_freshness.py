"""Render-time freshness filter for front/column story lists."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from scripts.render_static import StaticSiteGenerator

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _synth(story_id, age_hours):
    return {"story_id": story_id, "article_urls": [f"https://x/{story_id}"],
            "generated_headline": story_id,
            "generated_at": (NOW - timedelta(hours=age_hours)).isoformat()}


def test_stale_syntheses_filtered(monkeypatch):
    r = StaticSiteGenerator.__new__(StaticSiteGenerator)   # skip filesystem/Jinja setup
    r.os_client = MagicMock()
    r.os_client.get_syntheses.return_value = [_synth("fresh", 5), _synth("june", 1340)]
    monkeypatch.setattr("scripts.render_static._utcnow", lambda: NOW, raising=False)
    got = r.get_stories_for_column("politics", limit=20, max_age_hours=72)
    assert [s["story_id"] for s in got] == ["fresh"]


def test_filter_disabled_with_none():
    r = StaticSiteGenerator.__new__(StaticSiteGenerator)
    r.os_client = MagicMock()
    r.os_client.get_syntheses.return_value = [_synth("june", 1340)]
    got = r.get_stories_for_column("politics", limit=20, max_age_hours=None)
    assert len(got) == 1


def test_missing_generated_at_dropped():
    r = StaticSiteGenerator.__new__(StaticSiteGenerator)
    r.os_client = MagicMock()
    r.os_client.get_syntheses.return_value = [
        _synth("fresh", 5), {"story_id": "undated", "article_urls": ["https://x/u"]},
    ]
    got = r.get_stories_for_column("politics", limit=20, max_age_hours=72)
    assert [s["story_id"] for s in got] == ["fresh"]
