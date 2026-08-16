import math
from unittest.mock import MagicMock

from src.events.matcher import EventMatcher


def _settings(**kw):
    s = MagicMock()
    s.shortlist_threshold = kw.get("threshold", 0.60)
    s.shortlist_k = 3
    s.threadless_window_days = 14
    return s


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _synth(sid, emb, urls=("https://a",)):
    return {"story_id": sid, "column": "politics", "generated_headline": sid,
            "summary": "s", "article_urls": list(urls),
            "generated_at": "2026-08-16T00:00:00+00:00",
            "summary_embedding": _unit(emb)}


def test_attach_on_yes():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.event_id = "event-1"; ev.summary = "old"; ev.status = "active"
    ev.summary_embedding = _unit([1.0, 0.0]); store.get_all_events.return_value = [ev]
    llm.generate.side_effect = ["yes", "updated summary"]
    emb_vec = _unit([0.9, 0.1])
    m = EventMatcher(store, llm, emb, _settings())
    got = m.match_story(_synth("story-9", emb_vec))
    assert got == "event-1"
    assert store.attach_chapter.called


def test_no_attach_on_no():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.event_id = "event-1"; ev.summary = "old"
    ev.summary_embedding = _unit([1.0, 0.0]); store.get_all_events.return_value = [ev]
    store.search_threadless.return_value = []
    llm.generate.return_value = "no"
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-9", _unit([0.9, 0.1]))) is None
    assert not store.attach_chapter.called


def test_below_threshold_skips_llm():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.summary_embedding = _unit([1.0, 0.0])
    store.get_all_events.return_value = [ev]
    store.search_threadless.return_value = []
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-9", _unit([0.0, 1.0]))) is None
    assert not llm.generate.called


def test_recurrence_birth():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    store.get_all_events.return_value = []
    old = _synth("story-old", [1.0, 0.05], urls=("https://old",))
    store.search_threadless.return_value = [old]
    llm.generate.side_effect = ["yes", "seed summary"]
    created = MagicMock(); created.event_id = "event-n"; store.create_event.return_value = created
    m = EventMatcher(store, llm, emb, _settings())
    got = m.match_story(_synth("story-new", _unit([0.95, 0.1]), urls=("https://new",)))
    assert got == "event-n"
    chapters = store.create_event.call_args.kwargs["chapters"]
    assert [c["story_id"] for c in chapters] == ["story-old", "story-new"]


def test_same_cluster_candidate_skipped():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    store.get_all_events.return_value = []
    dup = _synth("story-old", [1.0, 0.05], urls=("https://a", "https://b"))
    store.search_threadless.return_value = [dup]
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-new", _unit([1.0, 0.05]),
                                urls=("https://a", "https://b", "https://c"))) is None
    assert not llm.generate.called


def test_llm_error_returns_none():
    store, llm, emb = MagicMock(), MagicMock(), MagicMock()
    ev = MagicMock(); ev.event_id = "event-1"; ev.summary = "old"
    ev.summary_embedding = _unit([1.0, 0.0]); store.get_all_events.return_value = [ev]
    llm.generate.side_effect = RuntimeError("down")
    m = EventMatcher(store, llm, emb, _settings())
    assert m.match_story(_synth("story-9", _unit([0.9, 0.1]))) is None
