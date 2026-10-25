"""Kind + subject instances: singular slugs, write-time convergence, read-time resolution."""


import asyncpg
import pytest
from sqlalchemy import select

from riki_memory import InstanceItem, MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.aggregation import (
    execute_aggregation,
    parse_aggregation_query,
    resolve_subjects,
)
from riki_memory.heuristic import make_heuristic_plan
from riki_memory.models import Instance
from riki_memory.schemas import singular, subject_slug
from riki_memory.store import save_memory


@pytest.mark.parametrize("word,expected", [
    ("instruments", "instrument"), ("weddings", "wedding"), ("parties", "party"),
    ("watches", "watch"), ("boxes", "box"), ("classes", "class"), ("glass", "glass"),
    ("bus", "bus"), ("people", "person"), ("series", "series"), ("fish", "fish"), ("car", "car"),
])
def test_singular(word, expected):
    assert singular(word) == expected


def test_subject_slug_singularises_only_the_last_word():
    assert subject_slug("Musical Instruments") == "musical-instrument"
    assert subject_slug("glasses cases") == "glasses-case"


@pytest.mark.parametrize("terms,existing,expected", [
    (["instruments", "own"], {"instrument", "wedding"}, ["instrument"]),
    (["doctor-visit"], {"doctor"}, ["doctor"]),
    (["festival"], {"film-festival", "music-festival", "wedding"}, ["film-festival", "music-festival"]),
    (["boats"], {"instrument"}, []),
    # first matching term wins: a stray tag must not be unioned into the count
    (["weddings", "sister"], {"wedding", "sister"}, ["wedding"]),
])
def test_resolve_subjects(terms, existing, expected):
    assert resolve_subjects(terms, existing) == expected


async def _save(memory, provider, content, subject, key, kind="item", status="current"):
    data = MemoryCreate(
        user_id="u1", content=content, kind="fact", importance=3,
        instances=[InstanceItem(kind=kind, subject=subject, instance_key=key,
                                display_name=key, status=status)],
    )
    return await save_memory(memory.db, provider, memory.config, data)


async def _subjects(memory) -> set[str]:
    async with memory.db.session_scope() as session:
        return set((await session.scalars(select(Instance.subject))).all())


async def test_plural_subject_converges_on_write(memory, provider):
    await _save(memory, provider, "I own a Fender guitar.", "instrument", "fender")
    await _save(memory, provider, "I own a Korg piano.", "Instruments", "korg")
    assert await _subjects(memory) == {"instrument"}


async def test_synonym_subject_converges_by_embedding(memory, provider):
    # The fake provider embeds anything mentioning "dentist" onto the same axis.
    await _save(memory, provider, "Saw Dr. Rao.", "dentist", "dr-rao", kind="visit", status="occurred")
    await _save(memory, provider, "Saw Dr. Lee.", "dentist-appointment", "dr-lee", kind="visit",
                status="occurred")
    await _save(memory, provider, "Went to Priya's wedding.", "wedding", "priya-wedding", kind="event",
                status="occurred")
    assert await _subjects(memory) == {"dentist", "wedding"}


async def test_question_counts_by_the_users_subject(memory, provider):
    await _save(memory, provider, "I own a Fender guitar.", "instrument", "fender")
    await _save(memory, provider, "I own a Korg piano.", "instrument", "korg")
    await _save(memory, provider, "Went to Priya's wedding.", "wedding", "priya-wedding", kind="event",
                status="occurred")

    question = "How many instruments do I currently own?"
    query = parse_aggregation_query(question, make_heuristic_plan(question, "UTC"))
    result = await execute_aggregation(memory.db, "u1", query)
    assert result.count == 2
    assert result.subjects == ["instrument"]


async def test_unknown_subject_returns_no_evidence(memory, provider):
    """Nothing to count is not the answer "0": recall falls back to ranked search on empty evidence."""
    await _save(memory, provider, "I own a Fender guitar.", "instrument", "fender")
    question = "How many boats do I own?"
    query = parse_aggregation_query(question, make_heuristic_plan(question, "UTC"))
    result = await execute_aggregation(memory.db, "u1", query)
    assert result.evidence == []
    assert result.subjects == []


async def test_outdated_instances_table_fails_loudly(db_url, provider):
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute("DROP TABLE instances")
        await conn.execute("CREATE TABLE instances (id TEXT PRIMARY KEY, category TEXT)")
    finally:
        await conn.close()
    mem = RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig())
    with pytest.raises(RuntimeError, match="kind/subject"):
        await mem.init()
    await mem.close()
