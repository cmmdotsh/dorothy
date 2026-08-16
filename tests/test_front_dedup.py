"""Front-page Developments: one entry per thread, cross-zone dedup."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from scripts.render_static import StaticSiteGenerator
from src.events.store import Event

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _synth(story_id, event_id, age_hours, headline=None):
    return {
        "story_id": story_id,
        "column": "politics",
        "event_id": event_id,
        "generated_headline": headline or f"{story_id} headline",
        "generated_at": (NOW - timedelta(hours=age_hours)).isoformat(),
        "article_urls": [f"https://x/{story_id}"],
        "article_count": 3,
    }


def _chapter(story_id, headline, at):
    return {"story_id": story_id, "generated_headline": headline,
            "generated_at": at, "article_count": 3}


def _event(event_id, title, chapters):
    return Event(event_id=event_id, title=title, summary="s",
                 summary_embedding=[], status="active", chapters=chapters,
                 columns=["politics"], first_seen=chapters[0]["generated_at"],
                 last_seen=chapters[-1]["generated_at"])


def _generator(monkeypatch, syntheses_by_column, events=None, events_error=None):
    r = StaticSiteGenerator.__new__(StaticSiteGenerator)   # skip filesystem/Jinja setup
    r.os_client = MagicMock()
    r.os_client.get_syntheses.side_effect = (
        lambda column=None, limit=50: syntheses_by_column.get(column, []))
    monkeypatch.setattr("scripts.render_static._utcnow", lambda: NOW, raising=False)

    store = MagicMock()
    if events_error is not None:
        store.get_all_events.side_effect = events_error
    else:
        store.get_all_events.return_value = events or []
    monkeypatch.setattr("scripts.render_static.EventStore", lambda os_client: store)
    return r


def test_two_chapters_collapse_to_one_entry(monkeypatch):
    old = _synth("s-old", "event-1", 70, headline="First report")
    new = _synth("s-new", "event-1", 5, headline="Second report")
    event = _event("event-1", "The Thread", [
        _chapter("s-old", "First report", old["generated_at"]),
        _chapter("s-new", "Second report", new["generated_at"]),
    ])
    r = _generator(monkeypatch, {"politics": [new, old]}, events=[event])

    devs = r.get_developments()

    assert len(devs) == 1
    dev = devs[0]
    assert dev["event_id"] == "event-1"
    assert dev["thread_title"] == "The Thread"
    assert dev["story"]["story_id"] == "s-new"
    assert dev["prev_headline"] == "First report"
    assert dev["chapter_count"] == 2
    assert dev["gap_days"] == 2  # 65h between chapters → 2 whole days


def test_department_story_excluded_from_developments(monkeypatch):
    # event-1's latest chapter is shown in a department slot → the whole
    # thread is dropped; the earlier in-window chapter is NOT resurrected.
    old = _synth("s-old", "event-1", 70, headline="First report")
    new = _synth("s-new", "event-1", 5, headline="Second report")
    other = _synth("s-other", "event-2", 10, headline="Other thread")
    event1 = _event("event-1", "Thread One", [
        _chapter("s-old", "First report", old["generated_at"]),
        _chapter("s-new", "Second report", new["generated_at"]),
    ])
    event2 = _event("event-2", "Thread Two",
                    [_chapter("s-other", "Other thread", other["generated_at"])])
    r = _generator(monkeypatch, {"sports": [new, old, other]},
                   events=[event1, event2])

    devs = r.get_developments(exclude_story_ids={"s-new"})

    assert [d["story"]["story_id"] for d in devs] == ["s-other"]


def test_event_store_failure_degrades_to_story_only(monkeypatch):
    old = _synth("s-old", "event-1", 70, headline="First report")
    new = _synth("s-new", "event-1", 5, headline="Second report")
    r = _generator(monkeypatch, {"politics": [new, old]},
                   events_error=RuntimeError("events index blown up"))

    devs = r.get_developments()  # must not raise

    assert len(devs) == 1
    dev = devs[0]
    assert dev["thread_title"] is None
    assert dev["prev_headline"] is None
    assert dev["gap_days"] is None
    assert dev["story"]["story_id"] == "s-new"
    assert dev["chapter_count"] == 2  # in-window chapters still counted


def test_gap_days_whole_days(monkeypatch):
    prev = _synth("s-prev", "event-1", 5 + 36, headline="Earlier")  # 36h older
    latest = _synth("s-late", "event-1", 5, headline="Later")
    event = _event("event-1", "T", [
        _chapter("s-prev", "Earlier", prev["generated_at"]),
        _chapter("s-late", "Later", latest["generated_at"]),
    ])
    r = _generator(monkeypatch, {"money": [latest, prev]}, events=[event])

    assert r.get_developments()[0]["gap_days"] == 1  # 36h truncates to 1 day


def test_sorted_by_latest_chapter_desc_and_capped(monkeypatch):
    a_new = _synth("s-a", "event-a", 3, headline="A latest")
    b_new = _synth("s-b", "event-b", 1, headline="B latest")
    ea = _event("event-a", "A", [_chapter("s-a", "A latest", a_new["generated_at"])])
    eb = _event("event-b", "B", [_chapter("s-b", "B latest", b_new["generated_at"])])
    r = _generator(monkeypatch, {"lifestyle": [a_new, b_new]}, events=[ea, eb])

    devs = r.get_developments()

    assert [d["story"]["story_id"] for d in devs] == ["s-b", "s-a"]
    assert [d["story"]["story_id"] for d in r.get_developments(limit=1)] == ["s-b"]


def test_render_front_page_dedups_across_zones(monkeypatch, tmp_path):
    # The live bug: a thread's latest chapter sits in a department slot AND
    # (twice) in Developments. After the fix it appears exactly once.
    lead = _synth("s-lead", "event-1", 2, headline="Lead story")
    filler1 = _synth("s-f1", None, 4, headline="Filler one")
    filler2 = _synth("s-f2", None, 6, headline="Filler two")
    old = _synth("s-old", "event-1", 70, headline="First report")
    solo = _synth("s-solo", "event-2", 30, headline="Solo thread")
    event1 = _event("event-1", "Thread One", [
        _chapter("s-old", "First report", old["generated_at"]),
        _chapter("s-lead", "Lead story", lead["generated_at"]),
    ])
    event2 = _event("event-2", "Thread Two",
                    [_chapter("s-solo", "Solo thread", solo["generated_at"])])
    r = _generator(monkeypatch, {"politics": [lead, filler1, filler2, solo, old]},
                   events=[event1, event2])
    r.output_dir = tmp_path
    captured = {}
    r.render_template = (
        lambda name, context: (captured.update(context), "<html></html>")[1])
    r.write_page = lambda path, content: None

    r.render_front_page()

    assert [s["story_id"] for s in captured["stories_by_column"]["politics"]] == [
        "s-lead", "s-f1", "s-f2"]
    # event-1's latest chapter is the politics lead → dropped from Developments
    # (earlier chapter s-old not resurrected); event-2 survives.
    assert [d["story"]["story_id"] for d in captured["developments"]] == ["s-solo"]
