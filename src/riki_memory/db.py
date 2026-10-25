"""Database plumbing. One engine per :class:`RikiMemory` instance, one transaction
per ``session_scope()`` block: committed if the block finishes, rolled back if
it raises.

The data lives in Postgres (self-hosted, Neon, RDS, Supabase, ...) with the
pgvector extension. The schema is managed by migrations (``migrate.py``). Keyword search uses Postgres full-text search
(``keyword.py``), vector search uses pgvector HNSW indexes (``vectors.py``).
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event, inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from . import migrate
from .vectors import PgVectorIndex, register_codec


def async_url(database_url: str) -> str:
    """Accept a plain ``postgresql://`` URL (what most hosts hand out) and use asyncpg."""
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+asyncpg://" + database_url[len(prefix):]
    raise ValueError(
        "riki-memory needs a Postgres database with pgvector, e.g. "
        "postgresql://user:password@localhost:5432/riki (see docker-compose.yml)."
    )


def _refuse_outdated_instances(sync_conn) -> None:
    """Fail loudly on an ``instances`` table from before the kind/subject schema.

    ``create_all`` never alters an existing table, so an old one would only
    surface later as a failed insert on every save.
    """
    inspector = inspect(sync_conn)
    if not inspector.has_table("instances"):
        return
    columns = {c["name"] for c in inspector.get_columns("instances")}
    if "subject" not in columns:
        raise RuntimeError(
            "This database's 'instances' table predates the kind/subject schema. "
            "Instances are derived from extraction, so drop the table (or the whole "
            "database) and re-ingest."
        )


class Database:
    def __init__(self, database_url: str, pool_size: int = 5, max_overflow: int = 10) -> None:
        self.engine = create_async_engine(
            async_url(database_url),
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # survive the server or a proxy dropping idle connections
        )

        @event.listens_for(self.engine.sync_engine, "connect")
        def _binary_vectors(dbapi_conn, _):
            dbapi_conn.run_async(register_codec)

        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.vectors = PgVectorIndex()

    async def init(self) -> None:
        """Bring the schema up to date (migrations) and build indexes. Safe on every startup."""
        async with self.engine.begin() as conn:
            await self.vectors.prepare(conn)  # the vector type must exist before the tables
        # Connections opened before the extension existed have no vector codec.
        await self.engine.dispose()
        async with self.engine.begin() as conn:
            await conn.run_sync(_refuse_outdated_instances)
            await conn.run_sync(migrate.upgrade)
        await self.vectors.init(self)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            async with session.begin():
                yield session
