"""Deterministic SQL computation for aggregation queries.

When the question is "how many X", "total Y", or "average Z", this module
bypasses RRF top-k entirely and queries the Instance table directly.
The result is a computed value plus the evidence instance list for the
host LLM to cite — and the instances that were left out, with the reason,
so the answer can be checked.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import or_, select

from .db import Database
from .models import Instance, Memory
from .schemas import UNIT_ALIASES, RetrievalPlan, singular, subject_slug
from .similarity import cosine_similarity
from .tags import get_tag_embedding

log = logging.getLogger(__name__)

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

# Phrase families -> (statuses to count, whether the newest mention decides).
# Checked in order; the first family that matches wins.
_STATUS_FAMILIES: list[tuple[re.Pattern, list[str], bool]] = [
    (re.compile(r"\b(used to|no longer|sold|sell|got rid of|gave away|formerly|anymore)\b", re.I),
     ["former"], False),
    (re.compile(r"\b(plan(?:ning|ned)? to|going to|intend(?:ing)? to|will I|upcoming)\b", re.I),
     ["planned"], False),
    (re.compile(r"\b(want(?:ed)? to|wish(?:ed)?|hop(?:e|ing) to|would like|dream(?:ing)? of)\b", re.I),
     ["desired"], False),
    (re.compile(r"\b(currently|right now|still|do I (?:have|own)|at the moment|these days)\b", re.I),
     ["current", "occurred"], True),
]
# Past tense still includes "current": a guitar bought and still owned was bought.
_DEFAULT_STATUS = (["occurred", "current"], False)

# A unit named in the question turns "how many hours" into a SUM of hours.
_QUESTION_UNIT_RE = re.compile(
    r"\bhow (?:many|much) (hours?|minutes?|days?|weeks?|months?|years?|miles?|km|kilometers?|dollars?)\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(\$|\b(money|spen[dt]|spending|cost|paid|pay|raised?|donated?|earned?|dollars?|budget)\b)",
    re.IGNORECASE,
)


@dataclass
class AggregationResult:
    value: str
    count: int = 0
    numeric_total: float | None = None
    evidence: list = field(default_factory=list)  # Instance rows that were counted
    excluded: list[tuple] = field(default_factory=list)  # (Instance, reason) left out
    subjects: list[str] = field(default_factory=list)  # what the terms resolved to
    operation: str = ""  # what actually ran (COUNT can become SUM, see execute_aggregation)
    unit: str | None = None

    @property
    def resolved(self) -> bool:
        """False when no subject matched — nothing was counted, so "0" means nothing."""
        return bool(self.subjects)


@dataclass
class AggregationQuery:
    operation: str  # COUNT_DISTINCT, SUM, AVG, LIST
    # Candidate nouns from the question, in priority order. Resolved against
    # the user's real subjects at execution time (see resolve_subjects).
    terms: list[str]
    status_filter: list[str] = field(default_factory=list)
    # True for "what do I have now" questions: an instance's newest mention
    # decides its status (bought, then sold = former). False for history
    # questions ("did I ever buy"): any matching mention counts.
    current_state: bool = False
    unit: str | None = None  # the unit a SUM/AVG adds up; None = most common one found
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


def infer_status(text: str) -> tuple[list[str], bool]:
    """``(statuses to count, newest-mention-decides)`` for a question."""
    for pattern, statuses, current_state in _STATUS_FAMILIES:
        if pattern.search(text):
            return list(statuses), current_state
    return list(_DEFAULT_STATUS[0]), _DEFAULT_STATUS[1]


def question_unit(text: str) -> str | None:
    """The unit a question asks to add up, if it names one."""
    m = _QUESTION_UNIT_RE.search(text)
    if m:
        word = m.group(1).lower()
        return UNIT_ALIASES.get(word, UNIT_ALIASES.get(singular(word), word))
    if _MONEY_RE.search(text):
        return "USD"
    return None


def resolve_subjects(terms: list[str], existing: set[str]) -> list[str]:
    """Pick the subject a question counts, from the subjects this user actually has.

    Tried in order, first hit wins (one subject — a stray tag like "sister"
    must not be unioned into a wedding count):

    1. a term equals a subject ("instruments" -> "instrument");
    2. a word inside a term is a subject ("doctor-visit" -> "doctor");
    3. a subject ends with a term ("festival" -> "film-festival").

    No match returns [] — the caller tries meaning-level matching next
    (:func:`resolve_subjects_by_meaning`), then falls back to ranked retrieval
    rather than answer "0" for something it has simply never seen.
    """
    wanted = [subject_slug(t) for t in terms if t and t.strip()]
    for subject in wanted:
        if subject in existing:
            return [subject]
    for subject in wanted:
        for word in subject.split("-"):
            if singular(word) in existing:
                return [singular(word)]
    for subject in wanted:
        tails = sorted(e for e in existing if e.rsplit("-", 1)[-1] == subject)
        if tails:
            return tails
    return []


MAX_MEANING_CANDIDATES = 15


class SubjectMembers(BaseModel):
    members: list[str] = Field(
        default_factory=list,
        description="The candidates that are the asked-about thing or a kind of it, copied exactly.",
    )


_MEMBERS_PROMPT = """A person asks how many "{term}" they have, did or own. These are \
labels from their stored records. Return the labels that ARE a "{term}" or a kind of \
"{term}" (asked "instrument": guitar and piano count). Leave out labels that are only \
related, used with it, or a different thing of the same sort (asked "instrument": \
guitar-lesson and concert do not count; asked "dog": cat does not count). \
If none fit, return an empty list."""


async def candidate_subjects(
    llm, term: str, existing: set[str], threshold: float
) -> list[str]:
    """Stored subjects whose embedding is close to ``term``, closest first."""
    term_vec = await get_tag_embedding(llm, subject_slug(term))
    if term_vec is None:
        return []
    scored = []
    for subject in existing:
        vec = await get_tag_embedding(llm, subject)
        if vec is not None:
            sim = cosine_similarity(term_vec, vec)
            if sim >= threshold:
                scored.append((sim, subject))
    scored.sort(reverse=True)
    return [subject for _, subject in scored[:MAX_MEANING_CANDIDATES]]


async def confirm_members(llm, term: str, candidates: list[str]) -> list[str]:
    """Ask the fast model which candidates really are a kind of ``term``.

    Embeddings only say "related": dog and cat sit as close as pet and dog.
    A wrong subject silently corrupts a count, so if the model can't be asked
    this returns [] and the caller falls back to ranked retrieval.
    """
    try:
        reply = await llm.chat_json(
            _MEMBERS_PROMPT.format(term=subject_slug(term)),
            "Labels:\n" + "\n".join(candidates),
            SubjectMembers,
            fast=True,
        )
    except Exception as exc:
        log.warning("could not confirm subjects for %r: %s", term, exc)
        return []
    allowed = set(candidates)
    return sorted({m.strip() for m in reply.members} & allowed)


async def resolve_subjects_by_meaning(
    llm, terms: list[str], existing: set[str], threshold: float
) -> list[str]:
    """Subjects that are a kind of the first term that has any ("instrument" -> "guitar", "piano").

    Two steps: embeddings pick candidates above ``threshold`` (set low, for
    recall), then one fast LLM call keeps only the real members. Every member
    is included: a user may have stored guitars and pianos separately and
    still ask about "instruments".
    """
    for term in terms:
        if not term or not term.strip():
            continue
        candidates = await candidate_subjects(llm, term, existing, threshold)
        if not candidates:
            continue
        members = await confirm_members(llm, term, candidates)
        if members:
            return members
    return []


def parse_aggregation_query(
    question: str, plan: RetrievalPlan, not_after: datetime | None = None
) -> AggregationQuery | None:
    """Parse an aggregation question into a structured query, or None if not parseable."""
    from .heuristic import is_aggregation_query
    if not is_aggregation_query(question):
        return None

    if not plan.tags:
        return None

    operation = parse_aggregation_operation(question)
    unit = question_unit(question)
    if unit and operation in ("COUNT_DISTINCT", "LIST"):
        operation = "SUM"  # "how many hours did I drive" adds hours up
    status_filter, current_state = infer_status(question)

    return AggregationQuery(
        operation=operation,
        terms=list(plan.tags),
        status_filter=status_filter,
        current_state=current_state,
        unit=unit,
        not_after=not_after,
        start=plan.start,
    )


def _when(row: Instance) -> tuple:
    """Chronological sort key for one mention: its own date, else when it was saved."""
    return (row.date or row.created_at, row.created_at)


def _passes(row: Instance, query: AggregationQuery) -> bool:
    return not query.status_filter or row.status in query.status_filter


def _pick_one_per_key(
    rows: list[Instance], query: AggregationQuery
) -> tuple[list[Instance], list[tuple[Instance, str]]]:
    """One row per instance_key that passes the status filter, oldest first,
    plus the keys that did not pass (their newest mention, with the reason)."""
    by_key: dict[str, list[Instance]] = {}
    for row in rows:
        by_key.setdefault(row.instance_key, []).append(row)

    picked, excluded = [], []
    for mentions in by_key.values():
        mentions.sort(key=_when)
        if query.current_state:
            candidates = mentions[-1:]  # only the newest mention says what is true now
        else:
            candidates = mentions
        matching = [m for m in candidates if _passes(m, query)]
        if matching:
            picked.append(matching[-1])
        else:
            excluded.append((mentions[-1], mentions[-1].status))
    picked.sort(key=_when)
    excluded.sort(key=lambda pair: _when(pair[0]))
    return picked, excluded


def _same_mention(a: Instance, b: Instance) -> bool:
    """Two rows describing one amount: same thing, same value, and not provably on different days."""
    if a.instance_key != b.instance_key or a.numeric_value != b.numeric_value:
        return False
    if a.date is None or b.date is None:
        return True
    return a.date.date() == b.date.date()


def _rows_to_add_up(
    rows: list[Instance], query: AggregationQuery
) -> tuple[list[Instance], list[tuple[Instance, str]], str | None]:
    """The rows a SUM/AVG adds, the rows it leaves out (with why), and the unit used.

    Every row is its own amount — two $40 repairs under one key are $80 — except
    a repeat mention of the same amount (same key, value and day), which counts once.
    Only one unit is ever added: dollars and hours never mix.
    """
    excluded: list[tuple[Instance, str]] = []
    valued = []
    for row in sorted(rows, key=_when):
        if not _passes(row, query):
            excluded.append((row, row.status))
        elif row.numeric_value is None:
            excluded.append((row, "no amount stated"))
        else:
            valued.append(row)

    unit = query.unit
    if unit is None and valued:
        unit = Counter(r.unit for r in valued).most_common(1)[0][0]

    kept: list[Instance] = []
    for row in valued:
        if row.unit != unit:
            excluded.append((row, f"in {row.unit}, not {unit}"))
        elif any(_same_mention(row, k) for k in kept):
            excluded.append((row, "repeat mention, counted once"))
        else:
            kept.append(row)
    return kept, excluded, unit


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(round(value, 2))


async def execute_aggregation(
    db: Database,
    user_id: str,
    query: AggregationQuery,
    llm=None,
    subject_threshold: float = 0.4,
) -> AggregationResult:
    """Run a deterministic SQL query against the Instance table.

    Pass ``llm`` to let terms match subjects by meaning when no word matches.
    """
    async with db.session_scope() as session:
        existing = set(
            (await session.scalars(
                select(Instance.subject).where(Instance.user_id == user_id).distinct()
            )).all()
        )
    subjects = resolve_subjects(query.terms, existing)
    if not subjects and llm is not None and existing:
        subjects = await resolve_subjects_by_meaning(llm, query.terms, existing, subject_threshold)
    if not subjects:
        return AggregationResult(value="0", operation=query.operation)

    async with db.session_scope() as session:
        base = select(Instance).where(
            Instance.user_id == user_id,
            Instance.subject.in_(subjects),
        )
        # The status filter is applied after grouping by key (below), not here:
        # filtering first would hide a later "sold it" behind an earlier "bought it".
        if query.not_after is not None:
            base = base.where(
                or_(Instance.date.is_(None), Instance.date <= query.not_after)
            )
        if query.start is not None:
            base = base.where(
                or_(Instance.date.is_(None), Instance.date >= query.start)
            )
        rows = list((await session.scalars(base)).all())

    operation = query.operation
    picked, excluded = _pick_one_per_key(rows, query)

    # "How many fish do I have" over rows that each state a count is a SUM of
    # those counts (6 tetras + 4 guppies = 10), not a count of the rows.
    if operation == "COUNT_DISTINCT" and picked and all(
        r.numeric_value is not None and r.unit == "count" for r in picked
    ):
        operation = "SUM"
        query = AggregationQuery(**{**query.__dict__, "unit": "count"})

    if operation == "COUNT_DISTINCT":
        return AggregationResult(
            value=str(len(picked)), count=len(picked), evidence=picked, excluded=excluded,
            subjects=subjects, operation=operation,
        )
    if operation == "LIST":
        return AggregationResult(
            value=", ".join(r.display_name for r in picked), count=len(picked), evidence=picked,
            excluded=excluded, subjects=subjects, operation=operation,
        )

    kept, excluded, unit = _rows_to_add_up(rows, query)
    values = [r.numeric_value for r in kept]
    if operation == "AVG":
        total = sum(values) / len(values) if values else 0.0
    else:
        total = sum(values)
    return AggregationResult(
        value=_number(total), count=len(kept), numeric_total=total, evidence=kept,
        excluded=excluded, subjects=subjects, operation=operation, unit=unit,
    )


def _instance_view(row: Instance) -> dict:
    return {
        "name": row.display_name,
        "key": row.instance_key,
        "status": row.status,
        "date": row.date.date().isoformat() if row.date else None,
        "value": row.numeric_value,
        "unit": row.unit,
        "memory_id": row.source_memory_id,
    }


def computed_view(result: AggregationResult, query: AggregationQuery) -> dict:
    """A JSON-safe description of what was computed, from what, and what was left out."""
    return {
        "type": "aggregate",
        "operation": result.operation,
        "subjects": result.subjects,
        "status_filter": query.status_filter,
        "current_state": query.current_state,
        "unit": result.unit,
        "value": result.value,
        "included": [_instance_view(r) for r in result.evidence],
        "excluded": [{**_instance_view(r), "reason": why} for r, why in result.excluded],
    }


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
