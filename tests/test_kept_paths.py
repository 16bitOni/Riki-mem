"""Paths kept in the Phase 4 cleanup because they do real work: the agentic
reflector loop and re-extraction of past turns."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.agentic import ReflectionResult
from riki_memory.maintenance import reextract_turns
from riki_memory.models import Instance, Memory
from riki_memory.schemas import Extraction


class Scripted:
    """Answers chat_json from a queue of canned replies per schema; embeddings from the fake provider."""

    def __init__(self, provider, replies: dict):
        self._provider = provider
        self._replies = {schema: list(items) for schema, items in replies.items()}
        self.embedding_model_id = provider.embedding_model_id
        self.calls: list[type] = []

    async def embed(self, text):
        return await self._provider.embed(text)

    async def chat_json(self, system, user, schema, *, fast=True):
        self.calls.append(schema)
        queue = self._replies.get(schema)
        if not queue:
            raise AssertionError(f"unexpected chat_json for {schema.__name__}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        return schema.model_validate(reply)

    async def chat_text(self, system, user, *, fast=True):
        raise AssertionError("unexpected chat_text call")


async def _memory(db_url, llm, **config) -> RikiMemory:
    mem = RikiMemory(
        database_url=db_url, llm=llm,
        config=MemoryConfig(planner_mode="heuristic", rerank="off", **config),
    )
    await mem.init()
    return mem


async def test_agentic_follow_up_finds_what_the_first_search_missed(db_url, provider):
    llm = Scripted(provider, {ReflectionResult: [
        {"sufficient": False, "reasoning": "dog name missing", "follow_up_query": "dog name",
         "follow_up_keywords": ["Bruno"]},
        {"sufficient": True, "reasoning": "all there"},
    ]})
    mem = await _memory(db_url, llm, retrieval_mode="agentic", max_retrieval_steps=3)
    try:
        await mem.save(MemoryCreate(user_id="u1", content="My sister lives in Paris.", kind="fact"))
        await mem.save(MemoryCreate(user_id="u1", content="Our family pet is Bruno.", kind="fact"))

        found, report = await mem.recall("u1", "Where does my sister live?")
        assert report["agentic_total_steps"] == 2
        assert {m.content for m in found} >= {"My sister lives in Paris.", "Our family pet is Bruno."}
        assert llm.calls == [ReflectionResult, ReflectionResult]
    finally:
        await mem.close()


TURN = {
    "user": "I bought a Yamaha guitar today.",
    "assistant": "Nice!",
    "at": datetime(2025, 3, 2, tzinfo=timezone.utc),
    "key": "session-1/turn-1",
}
EXTRACTION = {"memories": [{
    "content": "Bought a Yamaha FG800 guitar on 2025-03-02.", "kind": "event", "importance": 3,
    "semantic": False, "event_at": "2025-03-02T00:00:00+00:00",
    "instances": [{"kind": "item", "subject": "instrument", "instance_key": "yamaha-fg800",
                   "display_name": "Yamaha FG800", "status": "occurred"}],
}]}


async def _counts(mem) -> tuple[int, int]:
    async with mem.db.session_scope() as session:
        memories = (await session.scalars(select(Memory).where(Memory.is_active.is_(True)))).all()
        instances = (await session.scalars(select(Instance))).all()
    return len(memories), len(instances)


async def test_reextract_twice_does_not_pile_up_copies(db_url, provider):
    mem = await _memory(db_url, Scripted(provider, {Extraction: [EXTRACTION]}))
    try:
        first = await reextract_turns(mem.db, mem.llm, mem.config, "u1", [TURN])
        second = await reextract_turns(mem.db, mem.llm, mem.config, "u1", [TURN])
        assert (first["saved"], second["saved"], second["failed"]) == (1, 1, 0)
        assert await _counts(mem) == (1, 1)
    finally:
        await mem.close()


async def test_reextract_dry_run_saves_nothing(db_url, provider):
    mem = await _memory(db_url, Scripted(provider, {Extraction: [EXTRACTION]}))
    try:
        result = await reextract_turns(mem.db, mem.llm, mem.config, "u1", [TURN], dry_run=True)
        assert result["preview"] == ["Bought a Yamaha FG800 guitar on 2025-03-02."]
        assert await _counts(mem) == (0, 0)
    finally:
        await mem.close()
