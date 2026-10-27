"""The turn log: exchanges are written down before extraction, so nothing is lost,
retries cost nothing, and history can be replayed."""

import asyncio
import io
from datetime import timedelta

from sqlalchemy import select, update

from riki_memory import MemoryConfig, RikiMemory
from riki_memory import turns as turnlog
from riki_memory.models import Turn, utcnow


class Extractor:
    """Saves one memory per turn, echoing the user's message; can be told to fail."""

    def __init__(self, provider):
        self._provider = provider
        self.embedding_model_id = provider.embedding_model_id
        self.calls = 0
        self.fail = False
        self.prompts = []

    async def embed(self, text):
        return await self._provider.embed(text)

    async def chat_json(self, system, user, schema, *, fast=True):
        self.calls += 1
        self.prompts.append(user)
        if self.fail:
            raise RuntimeError("model down")
        said = user.rsplit("\n", 2)[-2].split(": ", 1)[1]
        return schema.model_validate({"memories": [{"content": f"Sub said: {said}", "kind": "fact"}]})

    async def chat_text(self, system, user, *, fast=True):
        return "A profile."


async def _mem(db_url, provider, **config):
    llm = Extractor(provider)
    mem = RikiMemory(database_url=db_url, llm=llm, config=MemoryConfig(planner_mode="heuristic", **config))
    await mem.init()
    return mem, llm


async def _turns(mem) -> list[Turn]:
    async with mem.db.session_scope() as session:
        return list((await session.scalars(select(Turn).order_by(Turn.created_at))).all())


async def test_every_turn_is_logged_and_a_retry_costs_nothing(db_url, provider):
    mem, llm = await _mem(db_url, provider)
    try:
        first = await mem.remember("u1", "I adopted a dog named Bruno.", "Congrats!", turn_key="chat/1")
        again = await mem.remember("u1", "I adopted a dog named Bruno.", "Congrats!", turn_key="chat/1")
        assert [m.id for m in again] == [m.id for m in first] and llm.calls == 1
        [turn] = await _turns(mem)
        assert (turn.status, turn.key, turn.attempts) == ("done", "chat/1", 1)
        assert first[0].source_key == "chat/1"
    finally:
        await mem.close()


async def test_a_turn_without_a_key_still_gets_provenance(db_url, provider):
    mem, _ = await _mem(db_url, provider)
    try:
        [saved] = await mem.remember("u1", "I adopted a dog named Bruno.", "Congrats!")
        [turn] = await _turns(mem)
        assert saved.source_key == turn.key == f"turn:{turn.id}"
    finally:
        await mem.close()


async def test_queued_turns_are_extracted_by_the_worker_in_order(db_url, provider):
    mem, llm = await _mem(db_url, provider)
    try:
        assert await mem.remember("u1", "I live in Pune.", "Nice.", wait=False) == []
        assert await mem.remember("u1", "I moved to Bangalore.", "Big change.", wait=False) == []
        assert llm.calls == 0 and {t.status for t in await _turns(mem)} == {"pending"}
        assert await mem.process_pending() == {"processed": 2, "failed": 0}
        assert [t.status for t in await _turns(mem)] == ["done", "done"]
        assert "I live in Pune." in llm.prompts[0] and "Bangalore" in llm.prompts[1]
        assert (await mem.health("u1"))["turns"] == {"done": 2}
    finally:
        await mem.close()


async def test_the_worker_loop_drains_the_queue_and_stops(db_url, provider):
    mem, llm = await _mem(db_url, provider)
    try:
        await mem.remember("u1", "I play the violin.", "Lovely.", wait=False)
        stop = asyncio.Event()
        worker = asyncio.create_task(mem.run_worker(stop, poll_seconds=0.05))
        for _ in range(100):
            if (await _turns(mem))[0].status == "done":
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(worker, 2)
        assert (await _turns(mem))[0].status == "done" and llm.calls == 1
    finally:
        await mem.close()


async def test_a_failed_turn_is_kept_retried_and_then_left_for_a_person(db_url, provider):
    mem, llm = await _mem(db_url, provider)
    try:
        llm.fail = True
        assert await mem.remember("u1", "I play the violin.", "Lovely.", turn_key="t") == []
        [turn] = await _turns(mem)
        assert turn.status == "failed" and "model down" in turn.error
        assert await mem.process_pending() == {"processed": 1, "failed": 1}
        assert await mem.process_pending() == {"processed": 1, "failed": 1}
        assert await mem.process_pending() == {"processed": 0, "failed": 0}  # 3 attempts used
        llm.fail = False
        async with mem.db.session_scope() as session:  # a person resets it
            await session.execute(update(Turn).values(attempts=0))
        assert await mem.process_pending() == {"processed": 1, "failed": 0}
        assert [m.content for m in await mem.list_memories("u1")] == ["Sub said: I play the violin."]
    finally:
        await mem.close()


async def test_a_turn_abandoned_by_a_crashed_worker_is_picked_up_again(db_url, provider):
    mem, _ = await _mem(db_url, provider)
    try:
        await mem.remember("u1", "I play the violin.", "Lovely.", wait=False)
        [taken] = await turnlog.claim_turns(mem.db, 5)  # this worker then "crashes"
        assert await turnlog.claim_turns(mem.db, 5) == []
        async with mem.db.session_scope() as session:
            await session.execute(update(Turn).values(claimed_at=utcnow() - turnlog.STALE - timedelta(seconds=1)))
        assert await mem.process_pending() == {"processed": 1, "failed": 0}
    finally:
        await mem.close()


async def test_two_workers_never_take_the_same_turn(db_url, provider):
    mem, _ = await _mem(db_url, provider)
    try:
        for i in range(10):
            await mem.remember("u1", f"Fact number {i}.", "ok.", wait=False)
        a, b = await asyncio.gather(turnlog.claim_turns(mem.db, 6), turnlog.claim_turns(mem.db, 6))
        ids_a, ids_b = {t.id for t in a}, {t.id for t in b}
        assert not ids_a & ids_b and len(ids_a | ids_b) == 10
    finally:
        await mem.close()


async def test_the_worker_shows_the_extractor_what_the_caller_showed(db_url, provider):
    mem, llm = await _mem(db_url, provider)
    try:
        [old] = await mem.remember("u1", "I live in Pune.", "Nice.")
        await mem.remember("u1", "Anything new?", "Tell me.", known=[old], wait=False)
        await mem.process_pending()
        assert old.id in llm.prompts[-1]
    finally:
        await mem.close()


async def test_a_hard_delete_also_erases_the_raw_turn(db_url, provider):
    mem, _ = await _mem(db_url, provider)
    try:
        [saved] = await mem.remember("u1", "My PIN hint is my cat's name.", "Noted.", turn_key="t")
        await mem.delete("u1", saved.id)
        [turn] = await _turns(mem)
        assert (turn.status, turn.user_text, turn.reply_text) == ("redacted", "", "")
    finally:
        await mem.close()


async def test_turns_survive_export_and_import(db_url, provider):
    mem, _ = await _mem(db_url, provider)
    try:
        await mem.remember("u1", "I play the violin.", "Lovely.", turn_key="t1")
        await mem.remember("u1", "I like jazz.", "Nice.", wait=False)
        out = io.StringIO()
        assert (await mem.export_user("u1", out))["turn"] == 2
        out.seek(0)
        assert (await mem.import_user(out, as_user="u2"))["turn"] == 2
        copies = [t for t in await _turns(mem) if t.user_id == "u2"]
        assert sorted(t.status for t in copies) == ["done", "pending"]
    finally:
        await mem.close()


async def test_reextract_replays_the_log(db_url, provider):
    mem, llm = await _mem(db_url, provider)
    try:
        await mem.remember("u1", "I play the violin.", "Lovely.")
        await mem.remember("u1", "I like jazz.", "Nice.")
        result = await mem.reextract("u1")
        assert result["turns"] == 2 and result["failed"] == 0
        assert len(await mem.list_memories("u1")) == 2  # already saved: nothing piles up
    finally:
        await mem.close()


async def test_without_the_turn_log_nothing_raw_is_kept(db_url, provider):
    mem, _ = await _mem(db_url, provider, store_turns=False)
    try:
        assert len(await mem.remember("u1", "I play the violin.", "Lovely.")) == 1
        assert await _turns(mem) == []
        try:
            await mem.remember("u1", "I like jazz.", "Nice.", wait=False)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        await mem.close()
