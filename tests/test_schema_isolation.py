"""riki's tables live in their own Postgres schema, so a host app's tables in the
same database (its own ``users``, ``memories``...) are never touched or confused."""

import asyncio
import uuid

import asyncpg
import pytest

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory

from conftest import _admin, _url_for


@pytest.fixture
async def empty_db():
    name = f"riki_s_{uuid.uuid4().hex[:12]}"
    await asyncio.to_thread(_admin, f"CREATE DATABASE {name}")
    yield _url_for(name)
    await asyncio.to_thread(_admin, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


async def _tables(url: str, schema: str) -> set[str]:
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = $1", schema)
        return {r["tablename"] for r in rows}
    finally:
        await conn.close()


async def test_host_tables_with_the_same_names_are_left_alone(empty_db, provider):
    conn = await asyncpg.connect(empty_db)
    await conn.execute("CREATE TABLE public.users (email text PRIMARY KEY, plan text)")
    await conn.execute("CREATE TABLE public.memories (id serial PRIMARY KEY, note text)")
    await conn.execute("INSERT INTO public.users VALUES ('a@b.c', 'pro')")
    await conn.close()

    async with RikiMemory(database_url=empty_db, llm=provider, config=MemoryConfig()) as mem:
        await mem.save(MemoryCreate(user_id="u1", content="Bruno is my dog.", kind="fact"))
        assert [m.content for m in await mem.list_memories("u1")] == ["Bruno is my dog."]

    assert {"users", "memories", "riki_schema_version"} <= await _tables(empty_db, "riki")
    assert await _tables(empty_db, "public") == {"users", "memories"}
    conn = await asyncpg.connect(empty_db)
    assert await conn.fetchval("SELECT plan FROM public.users") == "pro"
    await conn.close()


async def test_an_old_database_in_public_is_moved_with_its_rows(empty_db, provider):
    # Versions before 0.2 created everything in public.
    async with RikiMemory(database_url=empty_db, llm=provider, schema="public") as old:
        await old.save(MemoryCreate(user_id="u1", content="Bruno is my dog.", kind="fact"))

    async with RikiMemory(database_url=empty_db, llm=provider) as mem:
        assert [m.content for m in await mem.list_memories("u1")] == ["Bruno is my dog."]
        await mem.save(MemoryCreate(user_id="u1", content="Riya is my sister.", kind="fact"))

    assert "memories" not in await _tables(empty_db, "public")
    assert "memories" in await _tables(empty_db, "riki")


async def test_two_apps_can_keep_separate_memory_in_one_database(empty_db, provider):
    async with RikiMemory(database_url=empty_db, llm=provider, schema="app_a") as a, \
            RikiMemory(database_url=empty_db, llm=provider, schema="app_b") as b:
        await a.save(MemoryCreate(user_id="u1", content="Only app A knows this.", kind="fact"))
        assert await b.list_memories("u1") == []


def test_schema_name_is_validated():
    with pytest.raises(ValueError):
        RikiMemory(database_url="postgresql://x/y", schema="riki; drop table x")
