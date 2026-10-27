"""Core memory blocks: small labelled lists, edited a line at a time from new memories."""

import io

import pytest

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.profile import MAX_LINES, refresh_profile


class BlockEditor:
    """Answers the block-update call with edits built from the prompt."""

    def __init__(self, provider, make_edits):
        self._provider = provider
        self.embedding_model_id = provider.embedding_model_id
        self._make = make_edits
        self.prompts = []

    async def embed(self, text):
        return await self._provider.embed(text)

    async def chat_json(self, system, user, schema, *, fast=True):
        self.prompts.append(user)
        return schema.model_validate({"edits": self._make(user)})

    async def chat_text(self, system, user, *, fast=True):
        raise AssertionError("unexpected chat_text call")


@pytest.fixture
async def mem(db_url, provider):
    m = RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig(planner_mode="heuristic"))
    await m.init()
    yield m
    await m.close()


async def _save(mem, content, **kw):
    return await mem.save(MemoryCreate(user_id="u1", content=content, kind="fact", **kw))


async def test_new_memories_become_lines_that_cite_them(mem, provider):
    veg = await _save(mem, "Sub is vegetarian and allergic to peanuts.")
    await _save(mem, "Sub had toast today.")
    llm = BlockEditor(provider, lambda prompt: [
        {"block": "constraints", "op": "add", "text": "Vegetarian; peanut allergy.", "memory_ids": [veg.id, "made-up"]},
    ])
    text = await refresh_profile(mem.db, llm, mem.config, "u1")
    assert text == "constraints:\n- Vegetarian; peanut allergy."
    assert await mem.get_blocks("u1") == {"constraints": ["Vegetarian; peanut allergy."]}
    assert "Sub had toast today." in llm.prompts[0]  # it saw the new memories ...
    await refresh_profile(mem.db, llm, mem.config, "u1")
    assert len(llm.prompts) == 1  # ... and nothing new means no call at all


async def test_a_line_the_new_memories_contradict_is_removed(mem, provider):
    await mem.set_block("u1", "identity", ["Lives in Pune.", "Works as a business analyst."])
    await _save(mem, "Sub moved to Bangalore.")
    llm = BlockEditor(provider, lambda prompt: [
        {"block": "identity", "op": "remove", "text": "Lives in Pune."},
        {"block": "identity", "op": "add", "text": "Lives in Bangalore."},
    ])
    await refresh_profile(mem.db, llm, mem.config, "u1")
    assert (await mem.get_blocks("u1"))["identity"] == ["Works as a business analyst.", "Lives in Bangalore."]


async def test_a_full_block_drops_its_oldest_line(mem):
    await mem.set_block("u1", "projects", [f"Project {i}" for i in range(MAX_LINES + 3)])
    lines = (await mem.get_blocks("u1"))["projects"]
    assert len(lines) == MAX_LINES and lines[-1] == f"Project {MAX_LINES + 2}"


async def test_unknown_block_is_refused(mem):
    with pytest.raises(ValueError):
        await mem.set_block("u1", "secrets", ["x"])


async def test_deleting_a_memory_removes_the_lines_written_from_it(mem, provider):
    veg = await _save(mem, "Sub is vegetarian.")
    job = await _save(mem, "Sub works as a business analyst.")
    llm = BlockEditor(provider, lambda prompt: [
        {"block": "constraints", "op": "add", "text": "Vegetarian.", "memory_ids": [veg.id]},
        {"block": "identity", "op": "add", "text": "Business analyst.", "memory_ids": [job.id]},
    ])
    await refresh_profile(mem.db, llm, mem.config, "u1")
    await mem.delete("u1", veg.id)
    assert await mem.get_blocks("u1") == {"identity": ["Business analyst."]}


async def test_constraints_are_always_in_the_context(mem):
    await _save(mem, "Sub's sister Riya lives in Paris.")
    await mem.set_block("u1", "constraints", ["Allergic to peanuts."])
    await mem.set_block("u1", "identity", ["Business analyst in Kolkata."])
    text, found = await mem.context("u1", "Where does my sister Riya live?")
    assert found and "Allergic to peanuts." in text
    assert "Business analyst" not in text  # the rest of the profile only when nothing matched


async def test_blocks_follow_an_imported_copy(mem, provider):
    veg = await _save(mem, "Sub is vegetarian.")
    await refresh_profile(mem.db, BlockEditor(provider, lambda p: [
        {"block": "constraints", "op": "add", "text": "Vegetarian.", "memory_ids": [veg.id]}]), mem.config, "u1")
    out = io.StringIO()
    await mem.export_user("u1", out)
    out.seek(0)
    await mem.import_user(out, as_user="u2")
    [copy] = await mem.list_memories("u2")
    await mem.delete("u2", copy.id)
    assert await mem.get_blocks("u2") == {} and await mem.get_blocks("u1") == {"constraints": ["Vegetarian."]}
