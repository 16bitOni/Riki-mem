"""Deterministic SQL computation for aggregation queries.

When the question is "how many X", "total Y", or "average Z", this module
bypasses RRF top-k entirely and queries the Instance table directly.
The result is a computed value plus the evidence instance list for the
host LLM to cite.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select

from .db import Database
from .models import Instance, Memory
from .schemas import RetrievalPlan

log = logging.getLogger(__name__)

# Maps query nouns/tags to Instance category slugs.
TAG_TO_INSTANCE_CATEGORY: dict[str, str] = {
    "charity-event": "charity-event",
    "charity": "charity-event",
    "fundraiser": "charity-event",
    "wedding": "wedding-attended",
    "model-kit": "model-kit",
    "model-tanks": "model-kit",
    "model": "model-kit",
    "instrument": "instrument",
    "guitar": "instrument",
    "piano": "instrument",
    "ukulele": "instrument",
    "drum": "instrument",
    "doctor-visit": "doctor-visit",
    "doctor": "doctor-visit",
    "camping": "camping-trip",
    "camping-trip": "camping-trip",
    "bike-purchase": "bike-expense",
    "bike-maintenance": "bike-expense",
    "bike-expense": "bike-expense",
    "bike": "bike-expense",
    "aquarium": "aquarium-fish",
    "aquarium-fish": "aquarium-fish",
    "fish": "aquarium-fish",
    "fish-tank": "fish-tank",
    "tank": "fish-tank",
    "health-device": "health-device",
    "road-trip": "road-trip",
    "birth-event": "baby-born",
    "baby-born": "baby-born",
    "property-viewing": "property-viewed",
    "property-viewed": "property-viewed",
    "property": "property-viewed",
    "art-event": "art-event",
    "citrus": "citrus",
    "game-played": "game-played",
    "game": "game-played",
    "movie-festival": "movie-festival",
    "festival": "movie-festival",
    "pet": "pet",
    "vehicle": "vehicle",
    "car": "vehicle",
    "concert": "concert-attended",
    "concert-attended": "concert-attended",
    "subscription": "subscription",
    "class-taken": "class-taken",
    "class": "class-taken",
    "book-read": "book-read",
    "book": "book-read",
    "recipe-made": "recipe-made",
    "recipe": "recipe-made",
}

_OP_COUNT_RE = re.compile(
    r"\b(how many|count|number of)\b", re.IGNORECASE
)
_OP_SUM_RE = re.compile(
    r"\b(how much|total|sum|combined|altogether)\b", re.IGNORECASE
)
_OP_AVG_RE = re.compile(
    r"\b(average|mean|avg)\b", re.IGNORECASE
)
_OP_LIST_RE = re.compile(
    r"\b(list|what are|name all|which|all (?:the |my )?)\b", re.IGNORECASE
)

_STATUS_CURRENT_RE = re.compile(
    r"\b(currently|right now|do I (?:have|own)|still)\b", re.IGNORECASE
)
_STATUS_PAST_RE = re.compile(
    r"\b(did I|have I|attended|visited|bought|purchased|went to|raised|spent|played|made|took)\b",
    re.IGNORECASE,
)


@dataclass
class AggregationResult:
    value: str
    count: int = 0
    numeric_total: float | None = None
    evidence: list = field(default_factory=list)


@dataclass
class AggregationQuery:
    operation: str  # COUNT_DISTINCT, SUM, AVG, LIST
    category: str
    status_filter: list[str] = field(default_factory=list)
    not_after: datetime | None = None
    start: datetime | None = None


def parse_aggregation_operation(text: str) -> str:
    if _OP_AVG_RE.search(text):
        return "AVG"
    if _OP_SUM_RE.search(text):
        return "SUM"
    if _OP_COUNT_RE.search(text):
        return "COUNT_DISTINCT"
    if _OP_LIST_RE.search(text):
        return "LIST"
    return "COUNT_DISTINCT"


def _infer_status_filter(text: str) -> list[str]:
    if _STATUS_CURRENT_RE.search(text):
        return ["current", "occurred"]
    if _STATUS_PAST_RE.search(text):
        return ["occurred"]
    return ["occurred", "current"]


def _resolve_categories(tags: list[str]) -> list[str]:
    """Map plan tags to Instance category slugs."""
    cats = set()
    for tag in tags:
        key = tag.strip().lower().replace(" ", "-")
        cat = TAG_TO_INSTANCE_CATEGORY.get(key)
        if cat:
            cats.add(cat)
        else:
            cats.add(key)
    return list(cats)


def parse_aggregation_query(
    question: str, plan: RetrievalPlan, not_after: datetime | None = None
) -> AggregationQuery | None:
    """Parse an aggregation question into a structured query, or None if not parseable."""
    from .heuristic import is_aggregation_query
    if not is_aggregation_query(question):
        return None

    categories = _resolve_categories(plan.tags) if plan.tags else []
    if not categories:
        return None

    operation = parse_aggregation_operation(question)
    status_filter = _infer_status_filter(question)

    return AggregationQuery(
        operation=operation,
        category=categories[0],
        status_filter=status_filter,
        not_after=not_after,
        start=plan.start,
    )


async def execute_aggregation(
    db: Database,
    user_id: str,
    query: AggregationQuery,
) -> AggregationResult:
    """Run a deterministic SQL query against the Instance table."""
    async with db.session_scope() as session:
        base = select(Instance).where(
            Instance.user_id == user_id,
            Instance.category == query.category,
        )
        if query.status_filter:
            base = base.where(Instance.status.in_(query.status_filter))
        if query.not_after is not None:
            from sqlalchemy import or_
            base = base.where(
                or_(Instance.date.is_(None), Instance.date <= query.not_after)
            )
        if query.start is not None:
            base = base.where(
                or_(Instance.date.is_(None), Instance.date >= query.start)
            )

        rows = (await session.scalars(base.order_by(Instance.date.asc().nullslast()))).all()

    if not rows:
        return AggregationResult(value="0", evidence=[])

    # Dedup by instance_key for COUNT_DISTINCT
    seen_keys: dict[str, Instance] = {}
    for row in rows:
        if row.instance_key not in seen_keys:
            seen_keys[row.instance_key] = row

    unique_instances = list(seen_keys.values())

    if query.operation == "COUNT_DISTINCT":
        count = len(unique_instances)
        return AggregationResult(
            value=str(count),
            count=count,
            evidence=unique_instances,
        )
    elif query.operation == "SUM":
        total = sum(r.numeric_value for r in unique_instances if r.numeric_value is not None)
        return AggregationResult(
            value=str(total),
            count=len(unique_instances),
            numeric_total=total,
            evidence=unique_instances,
        )
    elif query.operation == "AVG":
        values = [r.numeric_value for r in unique_instances if r.numeric_value is not None]
        if values:
            avg = sum(values) / len(values)
            return AggregationResult(
                value=str(round(avg, 2)),
                count=len(values),
                numeric_total=avg,
                evidence=unique_instances,
            )
        return AggregationResult(value="0", evidence=unique_instances)
    else:  # LIST
        names = [r.display_name for r in unique_instances]
        return AggregationResult(
            value=", ".join(names),
            count=len(unique_instances),
            evidence=unique_instances,
        )


async def aggregation_to_memories(
    db: Database, evidence: list
) -> list[Memory]:
    """JOIN instance evidence back to their source Memory rows."""
    memory_ids = list({inst.source_memory_id for inst in evidence})
    if not memory_ids:
        return []
    async with db.session_scope() as session:
        rows = (await session.scalars(
            select(Memory).where(Memory.id.in_(memory_ids), Memory.is_active.is_(True))
        )).all()
    return list(rows)
