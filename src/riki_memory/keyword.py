"""Keyword search backends. One interface, two engines.

The library talks to :class:`KeywordIndex`; :func:`select_keyword_index`
picks the implementation from the database URL:

- ``sqlite…`` → :class:`SQLiteFTSIndex` — an FTS5 side table with BM25 ranking.
  The table is maintained explicitly (every write goes through ``store.py``).
- ``postgres…`` → :class:`PostgresTSIndex` — a GIN index on
  ``to_tsvector('english', content)``, queried with ``websearch_to_tsquery``
  and ranked with ``ts_rank_cd``. Nothing to maintain: the index is live.

Everything else (dedup, vector search, graph walk, maintenance) is plain
SQLAlchemy + Python and already runs on both.
"""

import logging
from datetime import datetime
from typing import Protocol

from sqlalchemy import func, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Memory

log = logging.getLogger(__name__)

CHANNEL_DEPTH = 20  # candidates from the keyword channel before fusion
MAX_TERMS = 10


def live_clauses(user_id: str, start: datetime | None, end: datetime | None) -> list:
    """WHERE clauses shared by the search channels: this user, not forgotten, inside the window."""
    clauses = [Memory.user_id == user_id, Memory.is_active.is_(True)]
    if start:
        clauses.append(func.coalesce(Memory.event_at, Memory.created_at) >= start)
    if end:
        clauses.append(func.coalesce(Memory.event_at, Memory.created_at) <= end)
    return clauses


def build_fts_query(keywords: list[str]) -> str | None:
    """Turn plan keywords into a keyword query: quoted phrases OR-ed together.

    Each keyword becomes a quoted phrase so "New York" matches the phrase, and
    any single keyword matching is enough. Both FTS5 ``MATCH`` and Postgres
    ``websearch_to_tsquery`` understand this shape.
    """
    terms = [k.replace('"', "").strip() for k in keywords[:MAX_TERMS]]
    terms = [t for t in terms if t]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


class KeywordIndex(Protocol):
    """What the library needs from a keyword engine."""

    name: str

    async def init(self, db) -> None:
        """Create whatever index/table the engine needs. Idempotent."""
        ...

    async def add(self, session: AsyncSession, memory_id: str, content: str) -> None:
        """Index one memory (call inside the write transaction)."""
        ...

    async def remove(self, session: AsyncSession, memory_id: str) -> None:
        """Drop one memory from the index (forget / supersede)."""
        ...

    async def search(
        self, db, user_id: str, keywords: list[str], start: datetime | None, end: datetime | None
    ) -> list[str]:
        """Memory ids, best first, honouring the live-memory filters."""
        ...


FTS_TABLE_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'porter unicode61'
)
""".strip()

# GIN trigram/full-text index over the live-computed tsvector. Nothing to
# maintain: Postgres keeps it in sync on every write.
TSVECTOR_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS memory_fts_idx
ON memories USING gin (to_tsvector('english', content))
""".strip()


def _as_aware(value) -> datetime | None:
    """Parse a stored datetime string back to an aware datetime (naive → UTC).

    SQLite stores datetimes as text; comparing them in Python keeps the
    comparison exact regardless of storage format.
    """
    from datetime import timezone

    if value is None:
        return None
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SQLiteFTSIndex:
    """Keyword search on SQLite via an FTS5 side table (BM25 ranked)."""

    name = "sqlite-fts5"

    async def init(self, db) -> None:
        async with db.engine.begin() as conn:
            await conn.execute(text(FTS_TABLE_DDL))

    async def add(self, session: AsyncSession, memory_id: str, content: str) -> None:
        await session.execute(
            text("INSERT INTO memory_fts (memory_id, content) VALUES (:id, :content)"),
            {"id": memory_id, "content": content},
        )

    async def remove(self, session: AsyncSession, memory_id: str) -> None:
        await session.execute(text("DELETE FROM memory_fts WHERE memory_id = :id"), {"id": memory_id})

    async def search(
        self, db, user_id: str, keywords: list[str], start: datetime | None, end: datetime | None
    ) -> list[str]:
        query = build_fts_query(keywords)
        if not query:
            return []
        stmt = text(
            """
            SELECT f.memory_id, m.event_at, m.created_at FROM memory_fts f
            JOIN memories m ON m.id = f.memory_id
            WHERE f.content MATCH :q
              AND m.user_id = :user_id
              AND m.is_active = 1
            ORDER BY bm25(memory_fts)
            LIMIT :depth
            """
        )
        async with db.session_scope() as session:
            rows = (
                await session.execute(stmt, {"q": query, "user_id": user_id, "depth": CHANNEL_DEPTH})
            ).all()
        ids = []
        for memory_id, event_at, created_at in rows:
            when = _as_aware(event_at) or _as_aware(created_at)
            if when is None:
                continue
            if start and when < _as_aware(start):
                continue
            if end and when > _as_aware(end):
                continue
            ids.append(memory_id)
        return ids


class PostgresTSIndex:
    """Keyword search on Postgres via tsvector (ranked with ts_rank_cd).

    This is the engine the original Riki ran on; the query below is its
    keyword channel, ported verbatim.
    """

    name = "postgres-tsvector"

    async def init(self, db) -> None:
        async with db.engine.begin() as conn:
            await conn.execute(text(TSVECTOR_INDEX_DDL))

    async def add(self, session: AsyncSession, memory_id: str, content: str) -> None:
        pass  # the GIN index tracks the table live; nothing to maintain

    async def remove(self, session: AsyncSession, memory_id: str) -> None:
        pass

    def _search_stmt(self, user_id: str, keywords: list[str], start, end):
        """Factored out so the generated SQL can be compile-checked without a live DB."""
        query = build_fts_query(keywords)
        english = literal_column("'english'")  # a literal, so Postgres can use the full-text index
        document = func.to_tsvector(english, Memory.content)
        wanted = func.websearch_to_tsquery(english, query)
        return (
            select(Memory.id)
            .where(*live_clauses(user_id, start, end), document.op("@@")(wanted))
            .order_by(func.ts_rank_cd(document, wanted).desc())
            .limit(CHANNEL_DEPTH)
        )

    async def search(
        self, db, user_id: str, keywords: list[str], start: datetime | None, end: datetime | None
    ) -> list[str]:
        if not build_fts_query(keywords):
            return []
        async with db.session_scope() as session:
            return list(
                (await session.scalars(self._search_stmt(user_id, keywords, start, end))).all()
            )


def select_keyword_index(database_url: str) -> KeywordIndex:
    """Pick the keyword engine from the SQLAlchemy URL.

    ``postgresql://…`` / ``postgresql+asyncpg://…`` (Neon, RDS, self-hosted) →
    tsvector. Anything else → SQLite FTS5.
    """
    if database_url.startswith("postgres"):
        return PostgresTSIndex()
    return SQLiteFTSIndex()
