"""Event thread matcher: cosine shortlist + LLM confirm + recurrence birth.

Implements spec §2.2-2.3: new syntheses are matched against existing event
threads (and recent threadless syntheses) via python-side cosine similarity
plus a strict small-LLM yes/no confirm. Matching failure never blocks
publishing — every failure path returns None.
"""

import math
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Confirm prompt (bias to "no", strict token).
CONFIRM_SYSTEM = (
    "You decide whether a news story is a development of an ongoing event. "
    "Answer with exactly one word: yes or no. When unsure, answer no. "
    "Same topic is NOT enough - it must be the same specific ongoing event, "
    "conflict, case, or storyline."
)
CONFIRM_TEMPLATE = (
    "ONGOING EVENT:\n{thread_summary}\n\n"
    "NEW STORY ({date}):\n{headline}\n{summary}\n\n"
    "Is the new story a development of the ongoing event? yes or no:"
)

# Summary-refresh prompt.
SUMMARY_SYSTEM = (
    "You maintain a neutral running summary of an ongoing news event. "
    "Rewrite the summary to incorporate the new development. "
    "Maximum 120 words. No commentary, no markdown."
)
SUMMARY_TEMPLATE = (
    "CURRENT SUMMARY:\n{summary}\n\nNEW DEVELOPMENT ({date}):\n{headline}\n{story_summary}\n\n"
    "UPDATED SUMMARY:"
)

# URL overlap above this means two syntheses are the same cluster
# (the same story re-run), not a recurring development.
SAME_CLUSTER_JACCARD = 0.3

SYNTHESIS_INDEX = "dorothy-synthesis"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _chapter(synthesis: dict) -> dict:
    """Chapter record stored on an event for a synthesis."""
    return {
        "story_id": synthesis.get("story_id"),
        "generated_headline": synthesis.get("generated_headline"),
        "generated_at": synthesis.get("generated_at"),
        "article_count": len(synthesis.get("article_urls") or []),
    }


def resolve_event_id(existing: Optional[dict], matcher_result: Optional[str]) -> Optional[str]:
    """Decide a synthesis' event_id: an inherited id (from the superseded
    synthesis, Jaccard fast-path) wins over a fresh matcher result."""
    if existing and existing.get("event_id"):
        return existing["event_id"]
    return matcher_result


class EventMatcher:
    """Matches new syntheses to event threads.

    Matching order (spec §2.2): existing threads first, then recurrence
    birth from recent threadless syntheses. Cosine shortlist is computed
    python-side; each shortlisted candidate gets one LLM yes/no confirm.
    """

    def __init__(self, store, llm_client, embedding_client, settings):
        self.store = store
        self.llm = llm_client
        self.embedding_client = embedding_client
        self.settings = settings

    def _embed(self, text: str) -> list[float]:
        return self.embedding_client.embed_single(text)

    def match_story(self, synthesis: dict) -> Optional[str]:
        """Returns event_id if the story attaches (or founds) a thread, else None.

        synthesis: a stored synthesis dict (story_id, column, generated_headline,
        summary, article_urls, generated_at, summary_embedding).
        """
        try:
            return self._match(synthesis)
        except Exception as e:
            logger.error("event_match_failed",
                         story_id=synthesis.get("story_id"), error=str(e))
            return None

    def _match(self, synthesis: dict) -> Optional[str]:
        embedding = synthesis.get("summary_embedding")
        if not embedding:
            logger.info("event_match_none", story_id=synthesis.get("story_id"),
                        reason="missing_summary_embedding")
            return None

        event_id = self._match_existing_thread(synthesis, embedding)
        if event_id:
            return event_id
        return self._match_recurrence_birth(synthesis, embedding)

    # -- stage 1: attach to an existing event thread ----------------------

    def _match_existing_thread(self, synthesis: dict,
                               embedding: list[float]) -> Optional[str]:
        shortlist = []
        for event in self.store.get_all_events():
            score = _cosine(embedding, event.summary_embedding or [])
            if score >= self.settings.shortlist_threshold:
                shortlist.append((score, event))
        shortlist.sort(key=lambda pair: pair[0], reverse=True)

        for score, event in shortlist[: self.settings.shortlist_k]:
            verdict = self._confirm(synthesis, event.summary)
            logger.info("event_match_candidate", story_id=synthesis.get("story_id"),
                        event_id=event.event_id, score=round(score, 4),
                        verdict=verdict)
            if verdict == "yes":
                new_summary = self._refresh_summary(synthesis, event.summary)
                self.store.attach_chapter(
                    event.event_id, _chapter(synthesis), new_summary,
                    self._embed(new_summary), synthesis.get("column"),
                )
                logger.info("event_attached", story_id=synthesis.get("story_id"),
                            event_id=event.event_id)
                return event.event_id
        return None

    # -- stage 2: recurrence birth from a threadless synthesis -------------

    def _match_recurrence_birth(self, synthesis: dict,
                                embedding: list[float]) -> Optional[str]:
        story_id = synthesis.get("story_id")
        urls = set(synthesis.get("article_urls") or [])
        candidates = []
        for cand in self.store.search_threadless(self.settings.threadless_window_days):
            cand_id = cand.get("story_id")
            if not cand_id or cand_id == story_id:
                continue  # self
            if _jaccard(urls, set(cand.get("article_urls") or [])) > SAME_CLUSTER_JACCARD:
                logger.debug("event_candidate_same_cluster", story_id=story_id,
                             candidate_story_id=cand_id)
                continue  # same cluster, not a development
            score = _cosine(embedding, cand.get("summary_embedding") or [])
            if score >= self.settings.shortlist_threshold:
                candidates.append((score, cand))
        candidates.sort(key=lambda pair: pair[0], reverse=True)

        for score, cand in candidates[: self.settings.shortlist_k]:
            verdict = self._confirm(synthesis, cand.get("summary") or "")
            logger.info("event_match_candidate", story_id=story_id, event_id=None,
                        score=round(score, 4), verdict=verdict,
                        candidate_story_id=cand.get("story_id"))
            if verdict == "yes":
                return self._birth_event(synthesis, cand)

        logger.info("event_match_none", story_id=story_id)
        return None

    def _birth_event(self, synthesis: dict, cand: dict) -> Optional[str]:
        story_id = synthesis.get("story_id")
        # Seed the rolling summary from both stories: the old threadless
        # synthesis is the "current summary", the new story the development.
        summary = self._refresh_summary(synthesis, cand.get("summary") or "")
        columns = [synthesis["column"]] if synthesis.get("column") else []
        event = self.store.create_event(
            title=synthesis.get("generated_headline") or story_id,
            summary=summary,
            summary_embedding=self._embed(summary),
            chapters=[_chapter(cand), _chapter(synthesis)],  # oldest -> newest
            columns=columns,
        )
        # Tag both synthesis docs with the event_id so neither stays threadless.
        self._tag_syntheses(event.event_id, [cand.get("story_id"), story_id])
        logger.info("event_born", story_id=story_id, event_id=event.event_id,
                    first_chapter=cand.get("story_id"))
        return event.event_id

    def _tag_syntheses(self, event_id: str, story_ids: list[Optional[str]]) -> None:
        for story_id in story_ids:
            if not story_id:
                continue
            try:
                self.store.os.client.update(
                    index=SYNTHESIS_INDEX, id=story_id,
                    body={"doc": {"event_id": event_id}}, refresh=True,
                )
            except Exception as e:
                # The event exists; a failed tag must not unwind it.
                logger.warning("event_tag_failed", story_id=story_id,
                               event_id=event_id, error=str(e))

    # -- LLM helpers --------------------------------------------------------

    def _confirm(self, synthesis: dict, thread_summary: str) -> str:
        prompt = CONFIRM_TEMPLATE.format(
            thread_summary=thread_summary,
            date=synthesis.get("generated_at") or "",
            headline=synthesis.get("generated_headline") or "",
            summary=synthesis.get("summary") or "",
        )
        response = self.llm.generate(prompt, system_prompt=CONFIRM_SYSTEM,
                                     max_tokens=16)
        words = (response or "").strip().lower().split()
        return words[0] if words else "no"

    def _refresh_summary(self, synthesis: dict, current_summary: str) -> str:
        prompt = SUMMARY_TEMPLATE.format(
            summary=current_summary,
            date=synthesis.get("generated_at") or "",
            headline=synthesis.get("generated_headline") or "",
            story_summary=synthesis.get("summary") or "",
        )
        response = self.llm.generate(prompt, system_prompt=SUMMARY_SYSTEM,
                                     max_tokens=300)
        updated = (response or "").strip()
        return updated or current_summary
