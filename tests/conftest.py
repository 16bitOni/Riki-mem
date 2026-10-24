"""Shared fixtures: an isolated SQLite database and a fake LLM provider."""

import math

import pytest

from riki_memory import MemoryConfig, RikiMemory
from riki_memory.llm import LLMProvider


class FakeProvider:
    """Deterministic, keyword-routed embeddings; LLM calls are forbidden.

    Any test that triggers a real model call fails loudly, which proves the
    cheap paths (dedup, pregate, keyword search) never touch the network.
    """

    embedding_model_id = "fake-test-model"

    AXES = ["dog", "cat", "paris", "coffee", "dentist", "sister"]

    def __init__(self):
        self.embed_calls = 0

    async def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        lowered = text.lower()
        vec = [1.0 if axis in lowered else 0.0 for axis in self.AXES]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def chat_json(self, system: str, user: str, schema, *, fast: bool = True):
        raise AssertionError("test triggered an unexpected LLM chat_json call")

    async def chat_text(self, system: str, user: str, *, fast: bool = True):
        raise AssertionError("test triggered an unexpected LLM chat_text call")


@pytest.fixture
async def provider():
    return FakeProvider()


@pytest.fixture
async def memory(tmp_path, provider):
    mem = RikiMemory(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        llm=provider,
        config=MemoryConfig(),
    )
    await mem.init()
    yield mem
    await mem.close()
