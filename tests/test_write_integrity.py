"""Write-path integrity: instances stay correct through extraction, supersede and forget."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from riki_memory import InstanceItem, MemoryCreate
from riki_memory.aggregation import (
    AggregationQuery,
    aggregation_to_memories,
    execute_aggregation,
    parse_aggregation_query,
)
from riki_memory.extraction import extract_and_save_turn
from riki_memory.models import Instance, Memory
from riki_memory.schemas import RetrievalPlan
from riki_memory.store import save_memory


def _at(month: int, day: int = 1) -> datetime:
    return datetime(2025, month, day, tzinfo=timezone.utc)


def _instrument(key: str, status: str, date: datetime | None = None, **extra) -> InstanceItem:
    return InstanceItem(
        kind="item", subject="instrument", instance_key=key, display_name=key.replace("-", " ").title(),
        status=status, date=date, **extra,
    )


async def _save(memory, provider, content: str, instances=(), event_at=None, kind="event"):
    data = MemoryCreate(
        user_id="u1", content=content, kind=kind, importance=3,
        event_at=event_at, instances=list(instances),
    )
    return await save_memory(memory.db, provider, memory.config, data)


async def _instances(memory) -> list[Instance]:
    async with memory.db.session_scope() as session:
        return list((await session.scalars(select(Instance).where(Instance.user_id == "u1"))).all())


async def _count(memory, **query) -> int:
    query.setdefault("operation", "COUNT_DISTINCT")
    query.setdefault("terms", ["instrument"])
    return (await execute_aggregation(memory.db, "u1", AggregationQuery(**query))).count


# ---------- W1: instances belong to the memory that mentions them ----------


class ScriptedExtractor:
    """Returns one canned extraction; embeddings come from the shared fake provider."""

    def __init__(self, provider, extraction: dict):
        self._provider = provider
        self._extraction = extraction
        self.embedding_model_id = provider.embedding_model_id

    async def embed(self, text):
        return await self._provider.embed(text)

    async def chat_json(self, system, user, schema, *, fast=True):
        return schema.model_validate(self._extraction)

    async def chat_text(self, system, user, *, fast=True):
        raise AssertionError("unexpected chat_text call")


async def test_each_memory_writes_only_its_own_instances(memory, provider):
    llm = ScriptedExtractor(provider, {"memories": [
        {"content": "Bought a B-29 model kit.", "kind": "event", "importance": 3, "semantic": False,
         "instances": [{"kind": "item", "subject": "model-kit", "instance_key": "b-29", "display_name": "B-29",
                        "status": "occurred"}]},
        {"content": "Bought a Camaro model kit.", "kind": "event", "importance": 3, "semantic": False,
         "instances": [{"kind": "item", "subject": "model-kit", "instance_key": "camaro", "display_name": "Camaro",
                        "status": "occurred"}]},
    ]})
    saved = await extract_and_save_turn(memory.db, llm, memory.config, "u1", "I bought kits", "Nice!")
    assert len(saved) == 2

    rows = await _instances(memory)
    assert len(rows) == 2
    by_key = {r.instance_key: r.source_memory_id for r in rows}
    assert by_key == {"b-29": saved[0].id, "camaro": saved[1].id}


async def test_one_invalid_instance_does_not_drop_the_memory(memory, provider):
    llm = ScriptedExtractor(provider, {"memories": [
        {"content": "Bought a Yamaha guitar.", "kind": "event", "importance": 3, "semantic": False,
         "instances": [{"kind": "item", "subject": "instrument", "instance_key": "yamaha", "display_name": "Yamaha",
                        "status": "bogus"}]},
    ]})
    saved = await extract_and_save_turn(memory.db, llm, memory.config, "u1", "I bought a guitar", "Cool")
    assert len(saved) == 1
    assert await _instances(memory) == []


# ---------- W2: an instance-key match never retires prose ----------


async def test_same_generic_key_keeps_both_memories(memory, provider):
    m1 = await _save(memory, provider, "Repaired the bike chain for $40.", event_at=_at(3),
                     instances=[InstanceItem(kind="money", subject="bike-expense", instance_key="bike-repair",
                                             display_name="Bike repair", status="occurred",
                                             numeric_value=40, unit="USD")])
    m2 = await _save(memory, provider, "Replaced a bike tyre for $35.", event_at=_at(4, 10),
                     instances=[InstanceItem(kind="money", subject="bike-expense", instance_key="bike-repair",
                                             display_name="Bike repair", status="occurred",
                                             numeric_value=35, unit="USD")])
    async with memory.db.session_scope() as session:
        active = (await session.scalars(
            select(Memory.id).where(Memory.user_id == "u1", Memory.is_active.is_(True))
        )).all()
    assert set(active) == {m1.id, m2.id}
    assert len(await _instances(memory)) == 2


# ---------- W3: forget() stops a thing being counted ----------


async def test_forget_removes_instances(memory, provider):
    m = await _save(memory, provider, "I own a Fender Stratocaster.", kind="fact",
                    instances=[_instrument("fender-stratocaster", "current")])
    assert await _count(memory, status_filter=["current"]) == 1

    assert await memory.forget("u1", m.id) is True
    assert await _count(memory, status_filter=["current"]) == 0
    assert await _instances(memory) == []


async def test_forget_one_mention_keeps_the_other(memory, provider):
    m1 = await _save(memory, provider, "I own a Fender Stratocaster.", kind="fact",
                     instances=[_instrument("fender-stratocaster", "current")])
    await _save(memory, provider, "My Fender Stratocaster is my favourite.", kind="preference",
                instances=[_instrument("fender-stratocaster", "current")])
    await memory.forget("u1", m1.id)
    assert await _count(memory, status_filter=["current"]) == 1


# ---------- W4: instances survive a near-duplicate supersede ----------


async def test_supersede_moves_instances_to_new_memory(memory, provider):
    old = await _save(memory, provider, "I bought a Fender Stratocaster guitar on March 2, 2025.",
                      event_at=_at(3, 2), instances=[_instrument("fender-stratocaster", "occurred")])
    new = await _save(memory, provider, "I bought a Fender Stratocaster guitar on March 2nd, 2025.",
                      event_at=_at(3, 2))

    async with memory.db.session_scope() as session:
        assert (await session.get(Memory, old.id)).is_active is False

    rows = await _instances(memory)
    assert [r.source_memory_id for r in rows] == [new.id]
    evidence = await aggregation_to_memories(memory.db, rows)
    assert [m.id for m in evidence] == [new.id]


async def test_supersede_with_same_instance_keeps_one_row(memory, provider):
    inst = _instrument("fender-stratocaster", "occurred")
    await _save(memory, provider, "I bought a Fender Stratocaster guitar on March 2, 2025.",
                event_at=_at(3, 2), instances=[inst])
    new = await _save(memory, provider, "I bought a Fender Stratocaster guitar on March 2nd, 2025.",
                      event_at=_at(3, 2), instances=[inst])

    rows = await _instances(memory)
    assert [r.source_memory_id for r in rows] == [new.id]


# ---------- W5: upsert on Postgres ----------


async def test_instances_roundtrip(memory):
    await memory.save(MemoryCreate(user_id="u1", content="I own a Korg piano.", kind="fact",
                                   instances=[_instrument("korg-piano", "current")]))
    result = await execute_aggregation(memory.db, "u1", AggregationQuery(
        operation="COUNT_DISTINCT", terms=["instrument"], status_filter=["current"]))
    assert result.count == 1


# ---------- W6: units ----------


@pytest.mark.parametrize("raw,expected", [
    ("$", "USD"), ("usd", "USD"), ("Dollars", "USD"), ("hrs", "hours"), ("Hour", "hours"),
    ("km", "km"), ("EUR", "EUR"), ("  ", None),
])
def test_unit_aliases(raw, expected):
    item = InstanceItem(kind="other", subject="x", instance_key="k", display_name="d", status="occurred", unit=raw)
    assert item.unit == expected


def test_bare_number_is_a_count():
    item = InstanceItem(kind="item", subject="fish", instance_key="neon-tetra", display_name="Neon tetra",
                        status="current", numeric_value=6)
    assert item.unit == "count"


# ---------- latest status wins for "now" questions ----------


async def test_bought_then_sold_is_not_currently_owned(memory, provider):
    await _save(memory, provider, "Bought a Pearl drum set.", event_at=_at(1),
                instances=[_instrument("pearl-drums", "occurred")])
    await _save(memory, provider, "Sold the Pearl drum set.", event_at=_at(6),
                instances=[_instrument("pearl-drums", "former")])

    now = dict(status_filter=["current", "occurred"], current_state=True)
    ever = dict(status_filter=["occurred", "current"], current_state=False)
    assert await _count(memory, **now) == 0
    assert await _count(memory, **ever) == 1


async def test_planned_then_attended_counts_once(memory, provider):
    for content, status, month in [("Plans to attend Sundance.", "planned", 1),
                                   ("Attended Sundance.", "occurred", 2)]:
        await _save(memory, provider, content, event_at=_at(month), instances=[InstanceItem(
            kind="event", subject="film-festival", instance_key="sundance", display_name="Sundance", status=status)])
    assert await _count(memory, terms=["film-festival"], status_filter=["occurred", "current"]) == 1


def test_current_state_comes_from_the_question():
    plan = RetrievalPlan(needs_memory=True, tags=["instrument"])
    now = parse_aggregation_query("How many instruments do I currently own?", plan)
    ever = parse_aggregation_query("How many instruments have I bought?", plan)
    assert now.current_state is True
    assert ever.current_state is False
    assert ever.status_filter == ["occurred", "current"]


async def test_start_window_without_not_after(memory, provider):
    await _save(memory, provider, "Bought a Korg piano.", event_at=_at(5),
                instances=[_instrument("korg-piano", "occurred", date=_at(5))])
    assert await _count(memory, status_filter=["occurred"], start=_at(4)) == 1
    assert await _count(memory, status_filter=["occurred"], start=_at(6)) == 0


# ---------- P0-A: one user's turn never leaks into another's extraction prompt ----------


async def test_saved_this_turn_is_scoped_to_the_user(memory, provider):
    await memory.save(MemoryCreate(user_id="other", content="Other user's secret plan.", source_key="chat-1/turn-1"))
    seen = []

    class Recorder(ScriptedExtractor):
        async def chat_json(self, system, user, schema, *, fast=True):
            seen.append(user)
            return await super().chat_json(system, user, schema, fast=fast)

    llm = Recorder(provider, {"memories": []})
    await extract_and_save_turn(memory.db, llm, memory.config, "u1", "I like tea.", "Noted.",
                                known=[], turn_key="chat-1/turn-1")
    assert seen and "secret plan" not in seen[0]


# ---------- 1.6 / 1.7: updates without `known`, lean and cacheable prompts, batched embeddings ----------


class Recording(ScriptedExtractor):
    """Records every extraction prompt and batched-embedding call; the reply may depend on the prompt."""

    def __init__(self, provider, extraction):
        super().__init__(provider, extraction)
        self.systems, self.prompts, self.batches = [], [], []

    async def embed_many(self, texts):
        self.batches.append(list(texts))
        return [await self._provider.embed(t) for t in texts]

    async def chat_json(self, system, user, schema, *, fast=True):
        self.systems.append(system)
        self.prompts.append(user)
        reply = self._extraction(user) if callable(self._extraction) else self._extraction
        return schema.model_validate(reply)


async def test_a_changed_fact_is_retired_without_the_caller_passing_known(memory, provider):
    old = await _save(memory, provider, "Sub lives in Pune.", kind="fact")

    def reply(prompt):
        assert old.id in prompt, "the old fact must be shown to the extractor"
        return {"memories": [{"content": "Sub lives in Bangalore.", "kind": "fact", "replaces": [old.id]}]}

    llm = Recording(provider, reply)
    await extract_and_save_turn(memory.db, llm, memory.config, "u1",
                                "I moved out of Pune, I live in Bangalore now.", "Big change!")
    retired = await memory.get("u1", old.id)
    assert retired.is_active is False and retired.valid_to is not None  # kept as history
    assert [m.content for m in await memory.list_memories("u1")] == ["Sub lives in Bangalore."]


async def test_the_system_prompt_is_identical_across_users_and_days(memory, provider):
    llm = Recording(provider, {"memories": []})
    await extract_and_save_turn(memory.db, llm, memory.config, "u1", "I like tea.", "Noted.",
                                at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    await extract_and_save_turn(memory.db, llm, memory.config, "u2", "I like jazz.", "Nice.",
                                at=datetime(2026, 6, 9, tzinfo=timezone.utc))
    assert len(llm.systems) == 2 and llm.systems[0] == llm.systems[1]
    assert "2026-06-09" in llm.prompts[1]  # the date travels in the message instead


async def test_only_instances_related_to_the_turn_are_sent(memory, provider):
    await _save(memory, provider, "Sub owns a Fender guitar.", [_instrument("fender-guitar", "current")])
    await _save(memory, provider, "Sub bought a snake plant.", [InstanceItem(
        kind="item", subject="plant", instance_key="snake-plant", display_name="Snake plant", status="current")])
    llm = Recording(provider, {"memories": []})
    await extract_and_save_turn(memory.db, llm, memory.config, "u1", "My plants need repotting.", "Spring is good for it.")
    prompt = llm.prompts[0]
    assert "snake-plant" in prompt and "fender-guitar" not in prompt
    assert "KNOWN SUBJECTS: instrument, plant" in prompt  # subjects are still all listed


async def test_one_turn_is_embedded_in_one_call(memory, provider):
    llm = Recording(provider, {"memories": [
        {"content": "Sub adopted a dog named Bruno.", "kind": "fact", "semantic": True},
        {"content": "Sub's sister Riya lives in Paris.", "kind": "fact", "semantic": True},
        {"content": "Sub drinks coffee every morning.", "kind": "fact", "semantic": True},
    ]})
    saved = await extract_and_save_turn(memory.db, llm, memory.config, "u1",
                                        "Lots of news: a dog, Riya moved, coffee daily.", "Wow!", known=[])
    assert len(saved) == 3 and len(llm.batches) == 1 and len(llm.batches[0]) == 3


# ---------- 2.3: where a memory came from ----------


async def test_a_guess_is_labelled_and_never_overrules_what_the_user_said(memory, provider):
    said = await _save(memory, provider, "Sub loves his job at the startup.", kind="fact")

    def reply(prompt):
        return {"memories": [{"content": "Sub seems unhappy at work.", "kind": "fact",
                              "source": "inferred", "replaces": [said.id]}]}

    await extract_and_save_turn(memory.db, Recording(provider, reply), memory.config, "u1",
                                "Ugh, long day at the startup job.", "Hang in there.")
    assert (await memory.get("u1", said.id)).is_active is True
    [guess] = await memory.list_memories("u1", source="inferred")
    assert guess.content == "Sub seems unhappy at work."
    text, _ = await memory.context("u1", "How do I feel about work at the startup job?")
    assert "[fact, inferred," in text
