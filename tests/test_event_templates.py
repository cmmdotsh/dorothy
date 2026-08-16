"""Event thread templates render standalone (no OpenSearch needed)."""
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader


ENV = Environment(loader=FileSystemLoader("src/web/templates"))
CTX = {
    "columns": ["politics"], "bias_colors": {}, "region_colors": {},
    "region_labels": {}, "perspective_colors": {}, "perspective_labels": {},
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "dateline": "Sunday, August 16, 2026", "edition": 1,
    "assets": {},
}


def _chapter(story_id, headline, at):
    return {"story_id": story_id, "generated_headline": headline,
            "generated_at": at, "article_count": 4}


def test_event_page_renders():
    ev = {"event_id": "event-abc", "title": "T", "summary": "S", "status": "active",
          "last_seen": "2026-08-16T00:00:00+00:00", "first_seen": "2026-06-01T00:00:00+00:00",
          "chapters": [_chapter("story-1", "H", "2026-08-16T00:00:00+00:00")]}
    html = ENV.get_template("event.html").render(event=ev, **CTX)
    assert "T" in html and "story-1" in html


def test_event_page_newest_chapter_first_and_dormant_line():
    ev = {"event_id": "event-abc", "title": "T", "summary": "S", "status": "dormant",
          "last_seen": "2026-08-16T00:00:00+00:00", "first_seen": "2026-06-01T00:00:00+00:00",
          "chapters": [
              _chapter("story-old", "Old headline", "2026-06-01T00:00:00+00:00"),
              _chapter("story-new", "New headline", "2026-08-16T00:00:00+00:00"),
          ]}
    html = ENV.get_template("event.html").render(event=ev, **CTX)
    # chapters stored oldest→newest; the timeline renders newest first
    assert html.index("New headline") < html.index("Old headline")
    assert "Dormant since" in html
    # chapters link to the real story-page URL pattern
    assert 'href="/story/story-new"' in html


def test_events_index_renders():
    html = ENV.get_template("events_index.html").render(active=[], dormant=[], **CTX)
    assert "html" in html.lower()


def test_events_index_lists_active_then_dormant():
    active = {"event_id": "event-a", "title": "Active thread", "summary": "S" * 300,
              "status": "active", "chapters": [_chapter("s1", "H", "2026-08-16T00:00:00+00:00")],
              "last_seen": "2026-08-16T00:00:00+00:00", "first_seen": "2026-08-01T00:00:00+00:00"}
    dormant = {"event_id": "event-d", "title": "Dormant thread", "summary": "short",
               "status": "dormant", "chapters": [], "last_seen": "2026-05-01T00:00:00+00:00",
               "first_seen": "2026-04-01T00:00:00+00:00"}
    html = ENV.get_template("events_index.html").render(active=[active], dormant=[dormant], **CTX)
    assert "Active thread" in html and "Dormant thread" in html
    assert html.index("Active thread") < html.index("Dormant thread")
    # one-line summary truncated to 160 chars + ellipsis
    assert "S" * 160 in html and "S" * 161 not in html and "…" in html
    assert 'href="/event/event-a"' in html
