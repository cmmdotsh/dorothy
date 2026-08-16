"""Publisher-cycle event wiring: inherit fast-path + matcher hookup (stubs)."""

from unittest.mock import MagicMock

from scripts.run_pipeline import (
    _embed_synthesis_summary,
    _plan_event_attachment,
    _thread_synthesis_event,
)


def _doc(**over):
    doc = {
        "story_id": "story-9",
        "column": "politics",
        "generated_headline": "H",
        "summary": "S",
        "article_urls": ["https://a", "https://b"],
        "generated_at": "2026-08-16T00:00:00+00:00",
    }
    doc.update(over)
    return doc


# -- pure decision: _plan_event_attachment ---------------------------------


def test_plan_inherited_id_wins_and_builds_chapter():
    event_id, chapter = _plan_event_attachment(_doc(), "event-a", "event-b")
    assert event_id == "event-a"
    assert chapter == {
        "story_id": "story-9",
        "generated_headline": "H",
        "generated_at": "2026-08-16T00:00:00+00:00",
        "article_count": 2,
    }


def test_plan_matcher_result_has_no_chapter():
    # The matcher attaches its own chapters — pipeline must not double-attach.
    event_id, chapter = _plan_event_attachment(_doc(), None, "event-b")
    assert event_id == "event-b"
    assert chapter is None


def test_plan_nothing_matches():
    assert _plan_event_attachment(_doc(), None, None) == (None, None)
    assert _plan_event_attachment(_doc(), "", "") == (None, None)


# -- inherit path: _thread_synthesis_event ----------------------------------


def test_inherit_attaches_llm_free_and_tags_doc():
    store, matcher = MagicMock(), MagicMock()
    event = MagicMock()
    event.summary = "rolling summary"
    event.summary_embedding = [0.1, 0.2]
    store.get_event.return_value = event

    doc = _doc(summary_embedding=[0.1, 0.2])
    got = _thread_synthesis_event(doc, "politics", "event-a", store, matcher)

    assert got == "event-a"
    assert doc["event_id"] == "event-a"
    # inherit fast-path: matcher never consulted
    matcher.match_story.assert_not_called()
    # summary/embedding passed through unchanged (LLM-free fallback)
    store.attach_chapter.assert_called_once_with(
        "event-a",
        {
            "story_id": "story-9",
            "generated_headline": "H",
            "generated_at": "2026-08-16T00:00:00+00:00",
            "article_count": 2,
        },
        new_summary="rolling summary",
        new_embedding=[0.1, 0.2],
        column="politics",
    )


def test_matcher_path_used_when_no_inheritance():
    store, matcher = MagicMock(), MagicMock()
    matcher.match_story.return_value = "event-m"

    doc = _doc(summary_embedding=[0.5])
    got = _thread_synthesis_event(doc, "tech", None, store, matcher)

    assert got == "event-m"
    assert doc["event_id"] == "event-m"
    matcher.match_story.assert_called_once_with(doc)
    store.attach_chapter.assert_not_called()  # matcher attached already


def test_matcher_skipped_without_embedding():
    store, matcher = MagicMock(), MagicMock()
    doc = _doc()  # no summary_embedding
    assert _thread_synthesis_event(doc, "politics", None, store, matcher) is None
    matcher.match_story.assert_not_called()
    assert "event_id" not in doc


def test_stage_noop_without_store_or_matcher():
    doc = _doc()
    assert _thread_synthesis_event(doc, "politics", "event-a", None, None) is None
    assert "event_id" not in doc


def test_inherit_link_survives_attach_failure():
    # get_event blowing up must log events_stage_failed, keep the doc's
    # inherited event_id (chain continuity), and never raise.
    store, matcher = MagicMock(), MagicMock()
    store.get_event.side_effect = RuntimeError("os down")

    doc = _doc(summary_embedding=[0.5])
    got = _thread_synthesis_event(doc, "politics", "event-a", store, matcher)

    assert got is None
    assert doc["event_id"] == "event-a"
    store.attach_chapter.assert_not_called()


def test_unreadable_event_still_tags_doc():
    store, matcher = MagicMock(), MagicMock()
    store.get_event.return_value = None  # missing/unreadable, not an exception

    doc = _doc(summary_embedding=[0.5])
    got = _thread_synthesis_event(doc, "politics", "event-a", store, matcher)

    assert got == "event-a"
    assert doc["event_id"] == "event-a"
    store.attach_chapter.assert_not_called()


# -- summary embedding ------------------------------------------------------


def test_embed_synthesis_summary_sets_embedding():
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2]
    doc = _doc()
    _embed_synthesis_summary(doc, embedder)
    embedder.embed_single.assert_called_once_with("H\nS")
    assert doc["summary_embedding"] == [0.1, 0.2]


def test_embed_synthesis_truncates_body_to_500_chars():
    embedder = MagicMock()
    doc = _doc(summary="x" * 2000)
    _embed_synthesis_summary(doc, embedder)
    text = embedder.embed_single.call_args.args[0]
    assert text == "H\n" + "x" * 500


def test_embed_synthesis_failure_stores_without():
    embedder = MagicMock()
    embedder.embed_single.side_effect = RuntimeError("embedding down")
    doc = _doc()
    _embed_synthesis_summary(doc, embedder)  # must not raise
    assert "summary_embedding" not in doc


def test_embed_synthesis_noop_without_client():
    doc = _doc()
    _embed_synthesis_summary(doc, None)
    assert "summary_embedding" not in doc
