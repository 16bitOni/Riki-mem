"""pgvector search: per-model HNSW indexes, models of different dimension side by
side, user scoping, and converting databases that stored vectors as JSON text."""

import asyncpg
from sqlalchemy import select, text
from sqlalchemy.dialects import postgresql

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.keyword import live_clauses
from riki_memory.models import Memory
from riki_memory.retrieval import semantic_search
from riki_memory.vectors import index_name

from conftest import FakeProvider


class WideProvider(FakeProvider):
    """Same keyword axes, padded to a different dimension: a second embedding model."""

    embedding_model_id = "fake-wide-model"

    async def embed(self, text):
        return (await super().embed(text)) + [0.0] * 10


async def _index_names(mem) -> set[str]:
    async with mem.db.engine.connect() as conn:
        rows = await conn.execute(text(
            "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indisvalid AND c.relname LIKE 'memories_hnsw_%'"
        ))
        return {name for (name,) in rows}


async def _save(mem, user, content):
    return await mem.save(MemoryCreate(user_id=user, content=content, kind="fact", semantic=True))


async def test_search_finds_by_meaning_and_builds_the_model_index(memory, provider):
    await _save(memory, "u1", "Bruno is my dog.")
    await _save(memory, "u1", "I drink black coffee.")
    found = await semantic_search(memory.db, provider, memory.config, "u1", "my dog", None, None)
    async with memory.db.session_scope() as session:
        dog = await session.scalar(select(Memory.id).where(Memory.content == "Bruno is my dog."))
    assert found == [dog]
    assert index_name("memories", provider.embedding_model_id, 6) in await _index_names(memory)


async def test_search_never_crosses_users(memory, provider):
    await _save(memory, "u1", "Bruno is my dog.")
    await _save(memory, "u2", "Rex is my dog.")
    found = await semantic_search(memory.db, provider, memory.config, "u2", "dog", None, None)
    async with memory.db.session_scope() as session:
        owners = set((await session.scalars(select(Memory.user_id).where(Memory.id.in_(found)))).all())
    assert owners == {"u2"}


async def test_two_models_of_different_dimension_live_side_by_side(db_url, provider):
    old = RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig())
    await old.init()
    await _save(old, "u1", "Bruno is my dog.")
    await old.close()

    wide = WideProvider()
    new = RikiMemory(database_url=db_url, llm=wide, config=MemoryConfig())
    await new.init()
    try:
        # before re-embedding, the new model sees nothing: vectors are never compared across models
        assert await semantic_search(new.db, wide, new.config, "u1", "dog", None, None) == []
        await _save(new, "u1", "Rex is a dog too.")
        found = await semantic_search(new.db, wide, new.config, "u1", "dog", None, None)
        assert len(found) == 1
        names = await _index_names(new)
        assert {index_name("memories", provider.embedding_model_id, 6),
                index_name("memories", wide.embedding_model_id, 16)} <= names
    finally:
        await new.close()


async def test_search_sql_inlines_the_model_for_the_partial_index(memory):
    stmt = memory.db.vectors._search_stmt(
        Memory, live_clauses("u1", None, None), [0.1, 0.2, 0.3], "it's-a-model", 20
    )
    sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"render_postcompile": True}))
    assert "embedding_model = 'it''s-a-model'" in sql
    assert "CAST(memories.embedding AS VECTOR(3)) <=>" in sql


async def test_text_vectors_from_an_old_database_are_converted(db_url, provider):
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("DROP INDEX IF EXISTS memory_fts_idx")
        await conn.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE text USING embedding::text")
        await conn.execute(
            "INSERT INTO users (id, name, timezone, profile, created_at) VALUES ('u1', 'u1', 'UTC', '', now())"
        )
        await conn.execute(
            "INSERT INTO memories (id, user_id, kind, content, content_hash, importance, embedding, "
            "wants_embedding, embedding_model, tags, is_active, created_at, updated_at) VALUES "
            "('m1', 'u1', 'fact', 'Bruno is my dog.', 'h1', 3, '[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]', "
            "true, $1, '[]', true, now(), now())",
            provider.embedding_model_id,
        )
    finally:
        await conn.close()

    mem = RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig())
    await mem.init()
    try:
        assert await semantic_search(mem.db, provider, mem.config, "u1", "dog", None, None) == ["m1"]
    finally:
        await mem.close()
