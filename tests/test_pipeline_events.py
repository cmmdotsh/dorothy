from src.events.matcher import resolve_event_id


def test_inherited_event_wins():
    assert resolve_event_id({"event_id": "event-a"}, "event-b") == "event-a"


def test_matcher_used_when_no_inheritance():
    assert resolve_event_id({"event_id": None}, "event-b") == "event-b"
    assert resolve_event_id(None, "event-b") == "event-b"


def test_none_when_nothing():
    assert resolve_event_id(None, None) is None
