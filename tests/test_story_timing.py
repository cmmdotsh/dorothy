from datetime import datetime, timedelta, timezone

from src.synthesis.summarizer import compute_story_timing

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _art(pub, bias="center"):
    return {"pub_date": pub.isoformat(), "column": "politics", "source_bias": bias}


def test_ancient_date_clamps_to_window_floor():
    arts = [_art(datetime(2022, 2, 8, tzinfo=timezone.utc)),
            _art(NOW - timedelta(hours=2)), _art(NOW - timedelta(hours=3))]
    t = compute_story_timing(arts, now=NOW, window_hours=72)
    floor = (NOW - timedelta(hours=72)).isoformat()
    assert t.first_pub_date == floor          # 2022 date floored, not shown
    assert t.median_pub_date >= floor


def test_future_date_clamps_to_now():
    arts = [_art(NOW + timedelta(days=2)), _art(NOW - timedelta(hours=1)),
            _art(NOW - timedelta(hours=2))]
    t = compute_story_timing(arts, now=NOW, window_hours=72)
    assert t.last_pub_date == NOW.isoformat()
    # future date can no longer pin hotness via a 1.0h clamp on a "future median"
    assert t.hotness_score <= len(arts) * 2.0


def test_in_window_dates_unchanged():
    d1, d2, d3 = (NOW - timedelta(hours=h) for h in (5, 10, 20))
    t = compute_story_timing([_art(d1), _art(d2), _art(d3)], now=NOW, window_hours=72)
    assert t.first_pub_date == d3.isoformat()
    assert t.last_pub_date == d1.isoformat()
