"""The planner pre-gate: greetings never reach the LLM."""

import pytest

from riki_memory import MemoryConfig, RikiMemory
from riki_memory.llm import NullProvider


@pytest.fixture
async def memory_no_llm(tmp_path):
    mem = RikiMemory(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/nollm.db",
        llm=NullProvider(),  # raises on any model call
        config=MemoryConfig(),
    )
    await mem.init()
    yield mem
    await mem.close()


@pytest.mark.parametrize("message", ["hi", "hey!", "thanks", "thank you", "bye", "good morning"])
async def test_greetings_skip_the_planner(memory_no_llm, message):
    # NullProvider would raise if the planner LLM were called.
    found, report = await memory_no_llm.recall("u1", message)
    assert found == []
    assert report["needs_memory"] is False


async def test_context_without_memories_still_renders(memory_no_llm):
    text, memories = await memory_no_llm.context("u1", "hello")
    assert memories == []
    assert "Memories that may be relevant" in text
