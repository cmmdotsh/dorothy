"""Event thread storage — recurrence-born threads in the dorothy-events index."""

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import structlog

from src.storage.opensearch import EVENTS_INDEX, SYNTHESIS_INDEX, utcnow

logger = structlog.get_logger(__name__)

THREADLESS_SOURCE_FIELDS = [
    "story_id",
    "column",
    "generated_headline",
    "summary",
    "article_urls",
    "generated_at",
    "summary_embedding",
]


@dataclass
class Event:
    """A long-lived event thread that daily stories attach to as chapters."""

    event_id: str
    title: str
    summary: str
    summary_embedding: list[float]
    status: str  # "active" | "dormant"
    chapters: list[dict]  # {story_id, generated_headline, generated_at, article_count}
    columns: list[str]
    first_seen: str  # isoformat
    last_seen: str

    @classmethod
    def from_doc(cls, doc: dict) -> "Event":
        return cls(
            event_id=doc.get("event_id", ""),
            title=doc.get("title", ""),
            summary=doc.get("summary", ""),
            summary_embedding=doc.get("summary_embedding") or [],
            status=doc.get("status", "active"),
            chapters=doc.get("chapters") or [],
            columns=doc.get("columns") or [],
            first_seen=doc.get("first_seen", ""),
            last_seen=doc.get("last_seen", ""),
        )

    def to_doc(self) -> dict:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "summary": self.summary,
            "summary_embedding": self.summary_embedding,
            "status": self.status,
            "chapters": self.chapters,
            "columns": self.columns,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class EventStore:
    """CRUD for event threads. Thread counts are tiny; every write uses refresh=True."""

    INDEX = EVENTS_INDEX

    def __init__(self, os_client):
        self.os = os_client

    def ensure_index(self) -> None:
        """Create the dorothy-events index if it doesn't exist."""
        self.os.ensure_events_index()

    def create_event(self, title, summary, summary_embedding, chapters, columns) -> Event:
        """Create a new thread. Idempotent by event_id (deterministic from chapter 1)."""
        chapters = list(chapters or [])
        event_id = "event-" + hashlib.sha256(
            chapters[0]["story_id"].encode()).hexdigest()[:12]
        now = utcnow().isoformat()
        first_seen = chapters[0].get("generated_at") or now
        event = Event(
            event_id=event_id,
            title=title,
            summary=summary,
            summary_embedding=list(summary_embedding or []),
            status="active",
            chapters=chapters,
            columns=list(columns or []),
            first_seen=first_seen,
            last_seen=now,
        )
        self.os.client.index(
            index=self.INDEX, id=event_id, body=event.to_doc(), refresh=True,
        )
        logger.info("event_created", event_id=event_id, title=title,
                    chapters=len(event.chapters))
        return event

    def attach_chapter(self, event_id, chapter, new_summary, new_embedding,
                       column) -> None:
        """Append chapter, update summary/embedding/last_seen/columns, set status=active.

        Read-modify-write full re-index (get, mutate, index) — simpler than painless.
        """
        event = self.get_event(event_id)
        if event is None:
            logger.error("event_attach_failed", event_id=event_id,
                         story_id=chapter.get("story_id"), error="event not found")
            return
        event.chapters.append(chapter)
        event.summary = new_summary
        event.summary_embedding = list(new_embedding or [])
        if column and column not in event.columns:
            event.columns.append(column)
        event.status = "active"
        event.last_seen = utcnow().isoformat()
        self.os.client.index(
            index=self.INDEX, id=event_id, body=event.to_doc(), refresh=True,
        )
        logger.info("event_chapter_attached", event_id=event_id,
                    story_id=chapter.get("story_id"), chapters=len(event.chapters))

    def get_all_events(self) -> list[Event]:
        """All events, active + dormant (thread counts are small)."""
        try:
            response = self.os.client.search(
                index=self.INDEX,
                body={"query": {"match_all": {}}, "size": 1000},
            )
            return [Event.from_doc(hit["_source"])
                    for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error("get_all_events_failed", error=str(e))
            return []

    def get_event(self, event_id) -> Optional[Event]:
        try:
            response = self.os.client.get(index=self.INDEX, id=event_id)
            return Event.from_doc(response["_source"])
        except Exception as e:
            logger.error("get_event_failed", event_id=event_id, error=str(e))
            return None

    def mark_dormant_older_than(self, days: int = 14) -> int:
        """status=dormant where status=active and last_seen < now-days. Returns count."""
        cutoff = (utcnow() - timedelta(days=days)).isoformat()
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"status": "active"}},
                        {"range": {"last_seen": {"lt": cutoff}}},
                    ]
                }
            },
            "script": {
                "source": "ctx._source.status = 'dormant'",
                "lang": "painless",
            },
        }
        try:
            response = self.os.client.update_by_query(
                index=self.INDEX, body=body, refresh=True, conflicts="proceed",
            )
            updated = response.get("updated", 0)
            if updated:
                logger.info("events_marked_dormant", count=updated, cutoff=cutoff)
            return updated
        except Exception as e:
            logger.error("mark_dormant_failed", error=str(e))
            return 0

    def search_threadless(self, window_days: int) -> list[dict]:
        """Threadless syntheses (no event_id) recent enough to match against.

        Recent = generated_at within window_days OR flagged thread_candidate
        (bootstrap). Only docs carrying a summary_embedding are returned —
        the matcher computes cosine python-side.
        """
        since = (utcnow() - timedelta(days=window_days)).isoformat()
        body = {
            "query": {
                "bool": {
                    "must_not": [{"exists": {"field": "event_id"}}],
                    "should": [
                        {"range": {"generated_at": {"gte": since}}},
                        {"term": {"thread_candidate": True}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": THREADLESS_SOURCE_FIELDS,
            "size": 1000,
        }
        try:
            response = self.os.client.search(index=SYNTHESIS_INDEX, body=body)
            docs = [hit["_source"] for hit in response["hits"]["hits"]]
            candidates = [d for d in docs if d.get("summary_embedding")]
            skipped = len(docs) - len(candidates)
            if skipped:
                logger.info("threadless_candidates_skipped",
                            missing_embedding=skipped)
            return candidates
        except Exception as e:
            logger.error("search_threadless_failed", error=str(e))
            return []
