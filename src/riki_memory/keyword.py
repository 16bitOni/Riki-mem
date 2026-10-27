"""Keyword search on Postgres full-text search.

A GIN index on ``to_tsvector('english', content)`` (created by the baseline
migration), queried with ``websearch_to_tsquery`` and ranked with
``ts_rank_cd``. Nothing to maintain: Postgres keeps the index in sync on every write.
"""

from datetime import datetime

from sqlalchemy import and_, func, literal_column, or_, select

from .models import Memory

CHANNEL_DEPTH = 20  # candidates from each search channel before fusion
MAX_TERMS = 10


def live_clauses(
    user_id: str, start: datetime | None, end: datetime | None, history: bool = False
) -> list:
    """WHERE clauses shared by the search channels: this user, not forgotten, inside the window.

    ``history``: also facts that are no longer true (retired with a ``valid_to``);
    never forgotten ones, never reworded duplicates.
    """
    current = Memory.is_active.is_(True)
    if history:
        current = or_(current, and_(Memory.valid_to.is_not(None), Memory.forgotten_at.is_(None)))
    clauses = [Memory.user_id == user_id, current]
    if start:
        clauses.append(func.coalesce(Memory.event_at, Memory.created_at) >= start)
    if end:
        clauses.append(func.coalesce(Memory.event_at, Memory.created_at) <= end)
    return clauses


def build_fts_query(keywords: list[str]) -> str | None:
    """Turn plan keywords into a ``websearch_to_tsquery`` query: quoted phrases OR-ed together.

    Each keyword becomes a quoted phrase so "New York" matches the phrase, and
    any single keyword matching is enough.
    """
    terms = [k.replace('"', "").strip() for k in keywords[:MAX_TERMS]]
    terms = [t for t in terms if t]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def keyword_search_stmt(user_id: str, keywords: list[str], start, end, history: bool = False):
    """Factored out so the generated SQL can be compile-checked without a live DB."""
    query = build_fts_query(keywords)
    english = literal_column("'english'")  # a literal, so Postgres can use the full-text index
    document = func.to_tsvector(english, Memory.content)
    wanted = func.websearch_to_tsquery(english, query)
    return (
        select(Memory.id)
        .where(*live_clauses(user_id, start, end, history), document.op("@@")(wanted))
        .order_by(func.ts_rank_cd(document, wanted).desc())
        .limit(CHANNEL_DEPTH)
    )


async def keyword_search(
    db, user_id: str, keywords: list[str], start: datetime | None, end: datetime | None,
    history: bool = False,
) -> list[str]:
    """Memory ids, best first, honouring the live-memory filters."""
    if not build_fts_query(keywords):
        return []
    async with db.session_scope() as session:
        return list((await session.scalars(keyword_search_stmt(user_id, keywords, start, end, history))).all())
