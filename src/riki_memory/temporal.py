"""Date arithmetic between remembered events: "how many days before X did I Y?".

The question is split into anchors, each anchor is searched on its own, and
the difference between the anchors' dates is computed in code — never by an
LLM reading two dates. An anchor's date is only trusted when it comes from a
memory that is actually about that anchor; otherwise the answer is "unknown"
and recall falls back to ordinary search instead of computing a wrong number.
"""

import logging
import re
from datetime import date, datetime
from typing import Awaitable, Callable

from pydantic import BaseModel, Field
from sqlalchemy import select

from .heuristic import STOPWORDS, extract_temporal_anchors
from .models import Instance, Memory

log = logging.getLogger(__name__)

# How much of an anchor's content words a memory must contain to date it.
ANCHOR_MATCH = 0.5

_UNIT_RE = re.compile(r"\bhow (?:many|much|long)\s+(day|week|month|year)s?\b", re.IGNORECASE)
_AGO_RE = re.compile(r"\bago\s+(?:did|was|were|have|had)\s+I\s+(.+?)\s*\??\s*$", re.IGNORECASE)

ANCHOR_PROMPT = """Split a question about the time between two events into those two events.
Return each event as a short phrase in the user's own words, without question words.
Example: "How many days after I adopted Max did I start the pottery class?" ->
anchor_a = "adopted Max", anchor_b = "start the pottery class".
If the question compares one event with today ("how long ago"), leave anchor_b empty."""


class AnchorPair(BaseModel):
    anchor_a: str = Field(description="The first event, as a short phrase.")
    anchor_b: str = Field(default="", description="The second event, or empty if it is 'now'.")


SearchFn = Callable[[str], Awaitable[list[Memory]]]


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    """Lower-cased, roughly stemmed words that carry meaning ("attended the AFI Fest" -> {attend, afi, fest})."""
    return {
        _stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if w not in STOPWORDS and len(w) >= 2
    }


def anchor_overlap(anchor: str, content: str) -> float:
    wanted = content_words(anchor)
    if not wanted:
        return 0.0
    return len(wanted & content_words(content)) / len(wanted)


def question_unit(text: str) -> str:
    m = _UNIT_RE.search(text or "")
    return f"{m.group(1).lower()}s" if m else "days"


async def anchors_for(llm, question: str) -> tuple[list[str], bool]:
    """``(anchors, compares_with_now)``. Regex first; the fast LLM only when regex
    finds fewer than two usable anchors and the question is not "how long ago"."""
    m = _AGO_RE.search(question)
    if m:
        return [m.group(1).strip()], True

    anchors = [a for a in extract_temporal_anchors(question) if content_words(a)]
    if len(anchors) >= 2:
        return anchors[:2], False
    try:
        pair = await llm.chat_json(system=ANCHOR_PROMPT, user=question, schema=AnchorPair, fast=True)
    except Exception:
        log.warning("anchor extraction LLM call failed; using regex anchors", exc_info=True)
        return anchors, False
    llm_anchors = [a.strip() for a in (pair.anchor_a, pair.anchor_b) if a and content_words(a)]
    if not pair.anchor_b.strip() and llm_anchors:
        return llm_anchors[:1], True
    return (llm_anchors if len(llm_anchors) >= len(anchors) else anchors), False


async def _instance_date(db, memory: Memory, anchor: str) -> datetime | None:
    """A matching instance's own date beats the memory's event_at (it was extracted for that thing)."""
    async with db.session_scope() as session:
        rows = (await session.scalars(
            select(Instance).where(Instance.source_memory_id == memory.id, Instance.date.is_not(None))
        )).all()
    for row in rows:
        if anchor_overlap(anchor, row.display_name) >= ANCHOR_MATCH:
            return row.date
    return None


async def date_anchor(db, anchor: str, found: list[Memory]) -> tuple[datetime | None, Memory | None]:
    """The date of the best-ranked memory that is about ``anchor`` and has a date."""
    for memory in found:
        if anchor_overlap(anchor, memory.content) < ANCHOR_MATCH:
            continue
        when = await _instance_date(db, memory, anchor) or memory.event_at
        if when is not None:
            return when, memory
    return None, None


def _as_date(value: datetime | date) -> date:
    return value.date() if isinstance(value, datetime) else value


def difference(a: datetime | date, b: datetime | date, unit: str) -> tuple[int, float | int]:
    """``(days, value in the asked unit)`` between two dates, order-independent.

    Months and years are calendar-based (Jan 31 -> Feb 28 is 0 months);
    weeks are days / 7, to one decimal.
    """
    first, second = sorted((_as_date(a), _as_date(b)))
    days = (second - first).days
    if unit == "weeks":
        weeks = round(days / 7, 1)
        return days, int(weeks) if weeks.is_integer() else weeks
    if unit in ("months", "years"):
        months = (second.year - first.year) * 12 + second.month - first.month
        if second.day < first.day:
            months -= 1
        return days, months if unit == "months" else months // 12
    return days, days


async def compute(
    db, llm, question: str, search: SearchFn, now: datetime
) -> tuple[list[Memory], dict]:
    """Run the temporal path. The report always says what happened; ``report["computed"]``
    is set only when every anchor was dated."""
    anchors, with_now = await anchors_for(llm, question)
    report: dict = {"temporal_arithmetic": True, "anchors": anchors}
    if not anchors or (len(anchors) < 2 and not with_now):
        report["temporal"] = "anchors_not_found"
        return [], report

    memories: list[Memory] = []
    dated: list[dict] = []
    for anchor in anchors:
        found = await search(anchor)
        for m in found:
            if all(m.id != x.id for x in memories):
                memories.append(m)
        when, source = await date_anchor(db, anchor, found)
        dated.append({
            "text": anchor,
            "date": _as_date(when).isoformat() if when else None,
            "memory_id": source.id if source else None,
        })
    if with_now:
        dated.append({"text": "now (when the question was asked)", "date": _as_date(now).isoformat(),
                      "memory_id": None})
    report["anchor_dates"] = [d["date"] for d in dated]

    if any(d["date"] is None for d in dated):
        report["temporal"] = "anchor_unresolved"
        return memories, report

    unit = question_unit(question)
    days, value = difference(date.fromisoformat(dated[0]["date"]), date.fromisoformat(dated[1]["date"]), unit)
    report["temporal"] = "computed"
    report["computed_days"] = days
    report["computed_answer"] = f"{value} {unit}"
    report["computed"] = {"type": "temporal", "unit": unit, "value": value, "days": days, "anchors": dated}
    # Anchor memories first: they are the evidence for the number.
    evidence_ids = [d["memory_id"] for d in dated if d["memory_id"]]
    memories.sort(key=lambda m: evidence_ids.index(m.id) if m.id in evidence_ids else len(evidence_ids))
    return memories, report
