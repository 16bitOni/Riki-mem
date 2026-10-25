"""Shared fixtures: an isolated Postgres database per test and a fake LLM provider.

Needs Postgres with pgvector: ``docker compose up -d`` starts one on port 5433.
Point ``RIKI_TEST_PG_URL`` elsewhere to use another server (the user needs
CREATEDB). Each test gets a fresh database cloned from a template that is
built once per run, so tests never see each other's rows.
"""

import asyncio
import math
import os
import uuid

import asyncpg
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


PG_URL = os.environ.get("RIKI_TEST_PG_URL", "postgresql://riki:riki@localhost:5433/riki")
TEMPLATE = "riki_test_template"


def _url_for(database: str) -> str:
    return PG_URL.rsplit("/", 1)[0] + "/" + database


def _admin(statement: str) -> None:
    async def run():
        conn = await asyncpg.connect(PG_URL.replace("postgresql+asyncpg://", "postgresql://"))
        try:
            await conn.execute(statement)
        finally:
            await conn.close()

    asyncio.run(run())


async def _build_template() -> None:
    mem = RikiMemory(database_url=_url_for(TEMPLATE), llm=FakeProvider())
    await mem.init()
    await mem.close()


@pytest.fixture(scope="session")
def _template_db():
    try:
        _admin(f"DROP DATABASE IF EXISTS {TEMPLATE} WITH (FORCE)")
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.exit(f"Postgres not reachable at {PG_URL} ({exc}). Run: docker compose up -d", 2)
    _admin(f"CREATE DATABASE {TEMPLATE}")
    asyncio.run(_build_template())
    yield TEMPLATE
    _admin(f"DROP DATABASE IF EXISTS {TEMPLATE} WITH (FORCE)")


@pytest.fixture
def db_url(_template_db):
    """URL of a fresh, empty (schema-only) database, dropped after the test."""
    name = f"riki_t_{uuid.uuid4().hex[:12]}"
    _admin(f"CREATE DATABASE {name} TEMPLATE {_template_db}")
    yield _url_for(name)
    _admin(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture
async def memory(db_url, provider):
    mem = RikiMemory(
        database_url=db_url,
        llm=provider,
        config=MemoryConfig(),
    )
    await mem.init()
    yield mem
    await mem.close()
