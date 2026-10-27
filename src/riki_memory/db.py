"""Database plumbing. One engine per :class:`RikiMemory` instance, one transaction
per ``session_scope()`` block: committed if the block finishes, rolled back if
it raises.

The data lives in Postgres (self-hosted, Neon, RDS, Supabase, ...) with the
pgvector extension. The schema is managed by migrations (``migrate.py``), run
automatically before the first query. Keyword search uses Postgres full-text
search (``keyword.py``), vector search uses pgvector HNSW indexes (``vectors.py``).
"""

import asyncio
import re
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
    schema = sync_conn.exec_driver_sql("SELECT current_schema()").scalar()
    inspector = inspect(sync_conn)
    if not inspector.has_table("instances", schema=schema):
        return
    columns = {c["name"] for c in inspector.get_columns("instances", schema=schema)}
    if "subject" not in columns:
        raise RuntimeError(
            "This database's 'instances' table predates the kind/subject schema. "
            "Instances are derived from extraction, so drop the table (or the whole "
            "database) and re-ingest."
        )


DEFAULT_SCHEMA = "riki"
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class Database:
    """``schema``: the Postgres schema that holds every riki table (default ``riki``),
    so names like ``users`` or ``memories`` never clash with the host app's tables."""

    def __init__(
        self, database_url: str, pool_size: int = 5, max_overflow: int = 10,
        schema: str = DEFAULT_SCHEMA,
    ) -> None:
        if not _SCHEMA_RE.match(schema):
            raise ValueError(f"schema must be a lowercase SQL identifier, got {schema!r}")
        self.schema = schema
        self.engine = create_async_engine(
            async_url(database_url),
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # survive the server or a proxy dropping idle connections
        )

        @event.listens_for(self.engine.sync_engine, "connect")
        def _on_connect(dbapi_conn, _):
            dbapi_conn.run_async(register_codec)
            # Prepended, not replaced: extensions installed in another schema
            # (Supabase's `extensions`) must still resolve.
            dbapi_conn.run_async(lambda c: c.execute(
                f"SELECT set_config('search_path', '\"{schema}\",' || current_setting('search_path'), false)"
            ))

        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.vectors = PgVectorIndex()
        self._ready = False
        self._init_lock = asyncio.Lock()

    async def init(self) -> None:
        """Bring the schema up to date (migrations) and build indexes. Safe on every startup.

        Optional: the first database call runs it automatically.
        """
        async with self._init_lock:
            if not self._ready:
                await self._init()
                self._ready = True

    async def _init(self) -> None:
        async with self.engine.begin() as conn:
            await self.vectors.prepare(conn)  # the vector type must exist before the tables
            await conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        # Connections opened before the extension existed have no vector codec.
        await self.engine.dispose()
        async with self.engine.begin() as conn:
            await conn.run_sync(migrate.adopt_public_tables)
            await conn.run_sync(_refuse_outdated_instances)
            await conn.run_sync(migrate.upgrade)
        await self.vectors.init(self)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        if not self._ready:
            await self.init()
        async with self.sessions() as session:
            async with session.begin():
                yield session
