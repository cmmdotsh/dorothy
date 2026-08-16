"""EventStore CRUD against a mocked OpenSearch client (no OpenSearch needed)."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.events.store import Event, EventStore

EXISTING_DOC = {
    "event_id": "event-abc",
    "title": "Iran strikes",
    "summary": "s1",
    "summary_embedding": [0.1] * 4,
    "status": "dormant",
    "chapters": [
        {
            "story_id": "story-1",
            "generated_headline": "h",
            "generated_at": "2026-06-01T00:00:00+00:00",
            "article_count": 5,
        }
    ],
    "columns": [],
    "first_seen": "2026-06-01T00:00:00+00:00",
    "last_seen": "2026-06-01T00:00:00+00:00",
}


def test_event_id_deterministic():
    client = MagicMock()
    store = EventStore(client)
    e1 = store.create_event("T", "S", [0.1], [{"story_id": "story-abc", "generated_headline": "h", "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 5}], ["politics"])
    e2 = store.create_event("T", "S", [0.1], [{"story_id": "story-abc", "generated_headline": "h", "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 5}], ["politics"])
    assert e1.event_id == e2.event_id
    assert e1.event_id.startswith("event-") and len(e1.event_id) == 18


def test_create_event_indexes_doc():
    client = MagicMock()
    store = EventStore(client)
    e = store.create_event("Iran strikes", "sum", [0.1] * 4,
                           [{"story_id": "story-1", "generated_headline": "h",
                             "generated_at": "2026-08-16T00:00:00+00:00", "article_count": 5}],
                           ["politics"])
    call = client.client.index.call_args.kwargs
    assert call["index"] == "dorothy-events"
    assert call["id"] == e.event_id
    assert call["body"]["status"] == "active"
    assert call["body"]["chapters"][0]["story_id"] == "story-1"


def test_attach_chapter_updates_and_reactivates():
    client = MagicMock()
    client.client.get.return_value = {"_source": dict(EXISTING_DOC)}
    store = EventStore(client)
    store.attach_chapter("event-abc", {"story_id": "story-2", "generated_headline": "h2",
                                       "generated_at": "2026-08-16T01:00:00+00:00", "article_count": 3},
                         new_summary="s2", new_embedding=[0.2] * 4, column="politics")
    # read-modify-write full re-index: get, mutate, index with refresh
    assert client.client.get.call_args.kwargs["id"] == "event-abc"
    call = client.client.index.call_args.kwargs
    assert call["index"] == "dorothy-events"
    assert call["id"] == "event-abc"
    assert call["refresh"] is True
    assert call["body"]["status"] == "active"
    assert [c["story_id"] for c in call["body"]["chapters"]] == ["story-1", "story-2"]
    assert call["body"]["summary"] == "s2"
    assert call["body"]["summary_embedding"] == [0.2] * 4
    assert call["body"]["columns"] == ["politics"]


def test_get_event_roundtrip():
    client = MagicMock()
    client.client.get.return_value = {"_source": dict(EXISTING_DOC)}
    store = EventStore(client)
    ev = store.get_event("event-abc")
    assert isinstance(ev, Event)
    assert ev.event_id == "event-abc"
    assert ev.status == "dormant"
    assert ev.chapters[0]["story_id"] == "story-1"
    assert ev.summary_embedding == [0.1] * 4


def test_get_event_missing_returns_none():
    client = MagicMock()
    client.client.get.side_effect = RuntimeError("missing")
    store = EventStore(client)
    assert store.get_event("event-nope") is None


def test_get_all_events_maps_hits():
    client = MagicMock()
    client.client.search.return_value = {"hits": {"hits": [
        {"_source": dict(EXISTING_DOC)},
        {"_source": {**EXISTING_DOC, "event_id": "event-def", "status": "active"}},
    ]}}
    store = EventStore(client)
    events = store.get_all_events()
    assert [e.event_id for e in events] == ["event-abc", "event-def"]
    assert {e.status for e in events} == {"dormant", "active"}
    body = client.client.search.call_args.kwargs["body"]
    assert body["query"] == {"match_all": {}}


def test_mark_dormant_older_than_body_and_count():
    client = MagicMock()
    client.client.update_by_query.return_value = {"updated": 2}
    store = EventStore(client)
    before = datetime.now(timezone.utc)
    count = store.mark_dormant_older_than(days=14)
    assert count == 2
    call = client.client.update_by_query.call_args.kwargs
    assert call["index"] == "dorothy-events"
    qb = call["body"]["query"]["bool"]
    assert {"term": {"status": "active"}} in qb["filter"]
    range_q = next(f for f in qb["filter"] if "range" in f)
    cutoff = datetime.fromisoformat(range_q["range"]["last_seen"]["lt"])
    expected = before - timedelta(days=14)
    assert expected - timedelta(minutes=1) < cutoff < datetime.now(timezone.utc) - timedelta(days=14) + timedelta(minutes=1)
    script = call["body"]["script"]
    assert "dormant" in script["source"]


def test_search_threadless_query_excludes_event_ids_and_or_branch():
    client = MagicMock()
    client.client.search.return_value = {"hits": {"hits": [
        {"_source": {"story_id": "s1", "summary_embedding": [0.1]}}
    ]}}
    store = EventStore(client)
    store.search_threadless(window_days=14)
    call = client.client.search.call_args.kwargs
    assert call["index"] == "dorothy-synthesis"
    body = call["body"]
    qb = body["query"]["bool"]
    # docs WITH an event_id are excluded
    assert qb["must_not"] == [{"exists": {"field": "event_id"}}]
    # recent OR flagged thread_candidate
    assert qb["minimum_should_match"] == 1
    range_q = next(s for s in qb["should"] if "range" in s)
    since = datetime.fromisoformat(range_q["range"]["generated_at"]["gte"])
    now = datetime.now(timezone.utc)
    assert now - timedelta(days=14, minutes=1) < since < now - timedelta(days=14) + timedelta(minutes=1)
    assert {"term": {"thread_candidate": True}} in qb["should"]
    assert "summary_embedding" in body["_source"]
    assert "article_urls" in body["_source"]


def test_search_threadless_drops_docs_without_embedding():
    client = MagicMock()
    client.client.search.return_value = {"hits": {"hits": [
        {"_source": {"story_id": "s-emb", "summary_embedding": [0.1]}},
        {"_source": {"story_id": "s-null", "summary_embedding": None}},
        {"_source": {"story_id": "s-absent"}},
    ]}}
    store = EventStore(client)
    got = store.search_threadless(window_days=14)
    assert [d["story_id"] for d in got] == ["s-emb"]
