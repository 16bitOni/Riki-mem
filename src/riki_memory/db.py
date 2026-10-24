"""Database plumbing. One engine per :class:`RikiMemory` instance, one transaction
per ``session_scope()`` block: committed if the block finishes, rolled back if
it raises.

The data lives wherever ``database_url`` points — a local SQLite file by
default, or Postgres (Neon, RDS, self-hosted) with the ``postgres`` extra.
The keyword engine is picked from the URL automatically (see ``keyword.py``);
everything else is plain SQLAlchemy and runs on both.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .keyword import KeywordIndex, select_keyword_index
from .models import Base


class Database:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_async_engine(database_url, connect_args=connect_args)
        if database_url.startswith("sqlite"):
            # Wait instead of failing on a locked database (multi-process use).
            @event.listens_for(self.engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):
                dbapi_conn.execute("PRAGMA busy_timeout = 5000")
                dbapi_conn.execute("PRAGMA journal_mode = WAL")

        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.keywords: KeywordIndex = select_keyword_index(database_url)

    async def init(self) -> None:
        """Create tables and the keyword index. Safe to call on every startup."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self.keywords.init(self)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            async with session.begin():
                yield session
