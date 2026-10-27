"""Getting started and day-to-day handling: one-line setup, no forgotten init(),
discoverable options, the app's own structured data, and the `riki` command."""

import io
import os

import pytest

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory, cli
from riki_memory.schemas import Extraction

from conftest import FakeProvider


# ---------- setup ----------


async def test_from_env_builds_everything_from_variables(db_url):
    memory = RikiMemory.from_env(env={
        "RIKI_DATABASE_URL": db_url, "RIKI_API_KEY": "k", "RIKI_BASE_URL": "http://x/v1",
        "RIKI_CHAT_MODEL": "big", "RIKI_FAST_MODEL": "small", "RIKI_EMBEDDING_MODEL": "emb",
    })
    try:
        assert memory.llm.embedding_model_id == "emb"
        assert (memory.llm._chat_model, memory.llm._fast_model) == ("big", "small")
    finally:
        await memory.close()


def test_from_env_explains_a_missing_database_url():
    with pytest.raises(ValueError, match="RIKI_DATABASE_URL"):
        RikiMemory.from_env(env={})


async def test_no_init_needed_before_the_first_call(db_url, provider):
    memory = RikiMemory(database_url=db_url, llm=provider)  # no init()
    try:
        saved = await memory.save(MemoryCreate(user_id="u1", content="Biscuit is my dog."))
        assert (await memory.get("u1", saved.id)).content == "Biscuit is my dog."
    finally:
        await memory.close()


async def test_async_with_opens_and_closes(db_url, provider):
    async with RikiMemory(database_url=db_url, llm=provider) as memory:
        await memory.save(MemoryCreate(user_id="u1", content="Biscuit is my dog."))
    assert memory.db._ready


# ---------- options ----------


def test_every_option_is_described_with_its_default():
    rows = {row["name"]: row for row in MemoryConfig.describe()}
    assert set(rows) == set(MemoryConfig.model_fields)
    assert all(row["description"] for row in rows.values())
    assert rows["rerank"]["limits"]["choices"] == ["off", "cross_encoder", "llm"]
    assert rows["recall_limit"]["default"] == 15 and rows["recall_limit"]["section"] == "retrieval"


def test_changed_lists_only_overrides():
    assert MemoryConfig().changed() == {}
    assert MemoryConfig(rerank="llm", recall_limit=5).changed() == {"rerank": "llm", "recall_limit": 5}


# ---------- the app's own data ----------


async def test_attributes_are_saved_filtered_and_kept_through_edits(memory):
    pill = await memory.save(MemoryCreate(
        user_id="u1", content="Sub takes 5 mg of amlodipine every morning.",
        attributes={"type": "medication", "dose": "5 mg", "times": ["08:00"]},
    ))
    await memory.save(MemoryCreate(user_id="u1", content="Sub likes black coffee."))
    found = await memory.list_memories("u1", attributes={"type": "medication"})
    assert [m.id for m in found] == [pill.id]

    edited = await memory.edit("u1", pill.id, content="Sub takes 10 mg of amlodipine every morning.")
    assert edited.attributes["type"] == "medication"
    edited = await memory.edit("u1", edited.id, attributes={"type": "medication", "dose": "10 mg"})
    assert edited.attributes == {"type": "medication", "dose": "10 mg"}


def test_bad_attributes_from_the_model_are_dropped_not_fatal():
    reply = Extraction.model_validate({"memories": [{
        "content": "Sub takes amlodipine.", "kind": "fact",
        "attributes": {"type": "medication", "nested": {"no": 1}, "": "x", "dose": "5 mg"},
    }]})
    assert reply.memories[0].attributes == {"type": "medication", "dose": "5 mg"}


class ScriptedExtractor(FakeProvider):
    def __init__(self, reply):
        super().__init__()
        self.reply = reply
        self.systems = []

    async def chat_json(self, system, user, schema, *, fast=True):
        self.systems.append(system)
        return schema.model_validate(self.reply)


async def test_the_app_can_teach_the_extractor_new_kinds_of_data(db_url):
    llm = ScriptedExtractor({"memories": [{
        "content": "Sub takes 5 mg of amlodipine every morning.", "kind": "fact",
        "attributes": {"type": "medication", "dose": "5 mg"},
    }]})
    config = MemoryConfig(
        planner_mode="heuristic", rerank="off",
        extraction_instructions="For medications, put type='medication' and the dose in attributes.",
    )
    async with RikiMemory(database_url=db_url, llm=llm, config=config) as memory:
        await memory.remember("u1", "I take 5 mg of amlodipine every morning.", "Noted.")
        assert "APP-SPECIFIC INSTRUCTIONS" in llm.systems[0] and "medications" in llm.systems[0]
        (saved,) = await memory.list_memories("u1", attributes={"type": "medication"})
        assert saved.attributes["dose"] == "5 mg"


# ---------- the riki command ----------


@pytest.fixture
def riki_env(db_url, monkeypatch):
    monkeypatch.setenv("RIKI_DATABASE_URL", db_url)
    monkeypatch.delenv("RIKI_API_KEY", raising=False)
    monkeypatch.delenv("RIKI_DB_URL", raising=False)
    return db_url


def test_riki_config_prints_every_option(capsys):
    cli.main(["config"])
    out = capsys.readouterr().out
    assert all(name in out for name in MemoryConfig.model_fields)


async def test_riki_doctor_checks_the_database(riki_env, capsys):
    failures = await cli.doctor()
    out = capsys.readouterr().out
    assert failures == 0
    assert "Postgres reachable" in out and "pgvector" in out and "schema" in out
    assert "RIKI_API_KEY not set" in out


async def test_riki_doctor_fails_clearly_without_a_database(monkeypatch, capsys):
    monkeypatch.delenv("RIKI_DATABASE_URL", raising=False)
    monkeypatch.delenv("RIKI_DB_URL", raising=False)
    assert await cli.doctor() == 1
    assert "not set" in capsys.readouterr().out


async def test_riki_users_export_and_delete(riki_env, tmp_path, capsys, monkeypatch):
    async with RikiMemory(database_url=riki_env, llm=FakeProvider()) as memory:
        await memory.create_namespace("acme/alice", tenant_id="acme")
        await memory.save(MemoryCreate(user_id="acme/alice", content="Alice likes tea."))

    await cli.users("acme", 10)
    assert "acme/alice\ttenant=acme" in capsys.readouterr().out
    target = tmp_path / "alice.jsonl"
    await cli.export("acme/alice", str(target), embeddings=False, since=None)
    assert "1 memories" in capsys.readouterr().out and target.exists()

    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # nobody there to confirm
    with pytest.raises(SystemExit, match="pass --yes"):
        await cli.delete_user("acme/alice", yes=False)
    await cli.delete_user("acme/alice", yes=True)
    assert "deleted acme/alice" in capsys.readouterr().out
    assert os.environ["RIKI_DATABASE_URL"] == riki_env


async def test_from_env_reads_a_dotenv_file_and_real_variables_win(db_url, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        f"# local settings\nRIKI_DATABASE_URL={db_url}\nRIKI_API_KEY='k'\nRIKI_FAST_MODEL=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for name in ("RIKI_DATABASE_URL", "RIKI_DB_URL", "RIKI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RIKI_FAST_MODEL", "from-environment")
    memory = RikiMemory.from_env()
    try:
        assert memory.llm._fast_model == "from-environment"
        assert memory.llm._api_key == "k"
    finally:
        await memory.close()
