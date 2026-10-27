"""Write-path behaviour: dedup, no-resurrection, forget, and the retrieval channels."""

from datetime import datetime, timezone

import pytest

from riki_memory import EntityMention, MemoryConfig, RelationMention, RikiMemory
from riki_memory.schemas import MemoryCreate


def make(content: str, **kwargs) -> MemoryCreate:
    return MemoryCreate(user_id="u1", content=content, **kwargs)


async def test_exact_duplicate_returns_first_without_reembedding(memory, provider):
    m1 = await memory.save(make("Sub likes black coffee."))
    calls_after_first = provider.embed_calls
    m2 = await memory.save(make("Sub likes black coffee."))
    assert m1.id == m2.id
    # the duplicate short-circuits before paying for an embedding
    assert provider.embed_calls == calls_after_first


async def test_near_duplicate_replaces_old(memory):
    m1 = await memory.save(make("Sub likes black coffee"))
    m2 = await memory.save(make("Sub likes black coffee."))
    assert m1.id != m2.id
    old = await memory.get("u1", m1.id)
    assert old.is_active is False
    assert old.superseded_by_id == m2.id
    # higher importance of the two is kept
    assert m2.importance == max(m1.importance, m2.importance)


async def test_same_slot_gate_keeps_different_dates(memory):
    march = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc)
    april = datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc)
    # Same thing, different wording (near-duplicate by spelling) — but different
    # date slots, so they must NOT merge.
    m1 = await memory.save(make("Sub has a dentist appointment booked", event_at=march))
    m2 = await memory.save(make("Sub has a dentist appointment scheduled", event_at=april))
    assert m1.id != m2.id
    assert (await memory.get("u1", m1.id)).is_active is True
    assert (await memory.get("u1", m2.id)).is_active is True


async def test_forget_is_final_no_resurrection(memory):
    m1 = await memory.save(make("Sub likes black coffee.", source_key="chat/1"))
    assert await memory.forget("u1", m1.id) is True
    # a retry or replay of the same turn never brings it back
    m2 = await memory.save(make("Sub likes black coffee.", source_key="chat/1"))
    assert m2.id == m1.id
    assert (await memory.get("u1", m2.id)).is_active is False


async def test_a_forgotten_fact_said_again_in_a_new_turn_is_saved(memory):
    m1 = await memory.save(make("Sub likes black coffee.", source_key="chat/1"))
    await memory.forget("u1", m1.id)
    m2 = await memory.save(make("Sub likes black coffee.", source_key="chat/9"))
    assert m2.id != m1.id and m2.is_active
    assert (await memory.get("u1", m1.id)).is_active is False  # the audit row stays
    assert [m.id for m in await memory.list_memories("u1")] == [m2.id]


async def test_the_same_sentence_on_two_dates_is_two_events(memory):
    march = datetime(2026, 3, 10, tzinfo=timezone.utc)
    april = datetime(2026, 4, 12, tzinfo=timezone.utc)
    m1 = await memory.save(make("Sub went to the dentist.", kind="event", event_at=march))
    m2 = await memory.save(make("Sub went to the dentist.", kind="event", event_at=april))
    m3 = await memory.save(make("Sub went to the dentist.", kind="event", event_at=april))
    assert m1.id != m2.id and m3.id == m2.id


async def test_forget_hides_from_keyword_search(memory):
    from riki_memory.retrieval import keyword_search

    await memory.save(make("Sub bought a blue bicycle yesterday"))
    ids = await keyword_search(memory.db, "u1", ["bicycle"], None, None)
    assert len(ids) == 1
    await memory.forget("u1", ids[0])
    ids = await keyword_search(memory.db, "u1", ["bicycle"], None, None)
    assert ids == []


async def test_semantic_search_finds_meaning(memory):
    from riki_memory.retrieval import semantic_search

    await memory.save(make("Sub adopted a golden retriever dog"))
    await memory.save(make("Sub visited Paris in the spring"))
    ids = await semantic_search(
        memory.db, memory.llm, memory.config, "u1", "tell me about the dog", None, None
    )
    assert len(ids) == 1
    found = await memory.get("u1", ids[0])
    assert "retriever" in found.content


async def test_graph_search_walks_relations(memory):
    from riki_memory.retrieval import graph_search

    await memory.save(
        make(
            "Riya works at Zerodha in Bangalore",
            entities=[EntityMention(name="Riya", type="person")],
            relations=[RelationMention(subject="Riya", predicate="sister of", target="Sub")],
        )
    )
    ids = await graph_search(memory.db, memory.config, "u1", ["sub"], ["sister"])
    assert len(ids) == 1


async def test_entities_can_be_disabled(db_url, provider):
    mem = RikiMemory(
        database_url=db_url,
        llm=provider,
        config=MemoryConfig(enable_entities=False),
    )
    await mem.init()
    try:
        m = await mem.save(
            make(
                "Riya works at Zerodha",
                entities=[EntityMention(name="Riya", type="person")],
            )
        )
        assert m.id
        from riki_memory.models import Entity

        async with mem.db.session_scope() as session:
            from sqlalchemy import select

            rows = (await session.scalars(select(Entity))).all()
        assert rows == []
    finally:
        await mem.close()


async def test_concurrent_identical_saves_return_same_row(memory):
    import asyncio

    async def save_once(i: int):
        return await memory.save(make("Sub likes black coffee."))

    first = await save_once(0)
    results = await asyncio.gather(*[save_once(i) for i in range(1, 6)])
    assert {r.id for r in results} == {first.id}


async def test_on_important_callback_fires(memory):
    seen = []

    async def cb(mem):
        seen.append(mem.id)

    memory.on_important = cb
    m = await memory.save(make("Sub got married today", importance=5))
    assert seen == [m.id]
    m2 = await memory.save(make("Sub drank water", importance=1))
    assert seen == [m.id]
    assert m2.id not in seen
