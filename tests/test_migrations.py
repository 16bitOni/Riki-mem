"""Schema migrations: fresh databases, databases from before migrations, concurrent
startups, and a guard that every model change ships with a migration."""

import asyncio

import asyncpg
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import select

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory, migrate
from riki_memory.migrate import include_object
from riki_memory.models import Base, Memory

from conftest import _admin, _url_for


def _head() -> str:
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(migrate.MIGRATIONS))
    return ScriptDirectory.from_config(config).get_current_head()


async def _fresh_db() -> str:
    import uuid

    name = f"riki_m_{uuid.uuid4().hex[:12]}"
    await asyncio.to_thread(_admin, f"CREATE DATABASE {name}")
    return name


async def _revision(mem) -> str | None:
    async with mem.db.engine.connect() as conn:
        return await conn.run_sync(migrate.current)


async def test_a_fresh_database_is_migrated_to_head(provider):
    name = await _fresh_db()
    mem = RikiMemory(database_url=_url_for(name), llm=provider, config=MemoryConfig())
    try:
        await mem.init()
        assert await _revision(mem) == _head()
        await mem.init()  # a second startup is a no-op
        assert await _revision(mem) == _head()
    finally:
        await mem.close()
        await asyncio.to_thread(_admin, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


async def test_the_models_match_the_migrations(memory):
    """Fails when a model changes without a migration: write one with
    ``python -m riki_memory.migrate revision "<message>" <scratch database url>``."""
    def diff(sync_conn):
        context = MigrationContext.configure(sync_conn, opts={"include_object": include_object, "compare_type": True})
        return compare_metadata(context, Base.metadata)

    async with memory.db.engine.connect() as conn:
        changes = await conn.run_sync(diff)
    assert changes == []


async def test_a_database_from_before_migrations_keeps_its_data(provider):
    name = await _fresh_db()
    url = _url_for(name)
    # How databases were made before migrations: the extension and create_all, no version table.
    conn = await asyncpg.connect(url)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.close()
    from sqlalchemy.ext.asyncio import create_async_engine

    from riki_memory.db import async_url

    engine = create_async_engine(async_url(url))
    async with engine.begin() as sa_conn:
        await sa_conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

    mem = RikiMemory(database_url=url, llm=provider, config=MemoryConfig())
    try:
        await mem.init()
        await mem.save(MemoryCreate(user_id="u1", content="Bruno is my dog.", kind="fact"))
        await mem.close()

        mem = RikiMemory(database_url=url, llm=provider, config=MemoryConfig())
        await mem.init()
        assert await _revision(mem) == _head()
        async with mem.db.session_scope() as session:
            assert (await session.scalars(select(Memory.content))).all() == ["Bruno is my dog."]
    finally:
        await mem.close()
        await asyncio.to_thread(_admin, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


async def test_two_processes_starting_at_once_both_succeed(provider):
    name = await _fresh_db()
    a = RikiMemory(database_url=_url_for(name), llm=provider, config=MemoryConfig())
    b = RikiMemory(database_url=_url_for(name), llm=provider, config=MemoryConfig())
    try:
        await asyncio.gather(a.init(), b.init())
        assert await _revision(a) == _head()
    finally:
        await a.close()
        await b.close()
        await asyncio.to_thread(_admin, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
