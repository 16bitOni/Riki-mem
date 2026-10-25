"""A user's control over their memory: edit, hard delete, delete everything, export, import."""

import io
import json

from sqlalchemy import func, select, update

from riki_memory import InstanceItem, MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.models import Entity, Episode, EpisodeMemory, Instance, Memory, User
from riki_memory.schemas import EntityMention, RelationMention

from conftest import FakeProvider


def _guitar(key="fender", status="current"):
    return InstanceItem(kind="item", subject="instrument", instance_key=key, display_name=key.title(), status=status)


async def _count(mem, model, *where) -> int:
    async with mem.db.session_scope() as session:
        return await session.scalar(select(func.count()).select_from(model).where(*where))


async def _instances(mem, memory_id):
    async with mem.db.session_scope() as session:
        return (await session.scalars(select(Instance).where(Instance.source_memory_id == memory_id))).all()


# ---------- edit ----------


async def test_editing_only_the_importance_changes_it_in_place(memory):
    m = await memory.save(MemoryCreate(user_id="u1", content="Sub likes black coffee.", kind="preference"))
    edited = await memory.edit("u1", m.id, importance=5, tags=["coffee"])
    assert (edited.id, edited.importance, edited.tags) == (m.id, 5, ["coffee"])


async def test_editing_the_text_saves_a_new_version_and_drops_stale_counts(memory):
    m = await memory.save(MemoryCreate(user_id="u1", content="Sub owns a Fender guitar.", instances=[_guitar()]))
    edited = await memory.edit("u1", m.id, content="Sub owns a Gibson guitar.")
    assert edited.id != m.id and edited.content == "Sub owns a Gibson guitar."
    old = await memory.get("u1", m.id)
    assert (old.is_active, old.superseded_by_id) == (False, edited.id)
    # the Fender instance came from the old text, so it is not carried over
    assert await _count(memory, Instance, Instance.user_id == "u1") == 0


async def test_editing_the_text_can_keep_or_replace_counts(memory):
    m = await memory.save(MemoryCreate(user_id="u1", content="Sub owns a Fender guitar.", instances=[_guitar()]))
    kept = await memory.edit("u1", m.id, content="Sub owns a Fender Stratocaster.", keep_instances=True)
    assert [i.instance_key for i in await _instances(memory, kept.id)] == ["fender"]
    replaced = await memory.edit("u1", kept.id, content="Sub owns a Gibson Les Paul.", instances=[_guitar("gibson")])
    assert [i.instance_key for i in await _instances(memory, replaced.id)] == ["gibson"]


async def test_editing_a_missing_memory_returns_none(memory):
    assert await memory.edit("u1", "nope", content="x") is None


# ---------- hard delete ----------


async def test_delete_removes_the_memory_its_history_and_what_was_written_from_it(memory):
    first = await memory.save(MemoryCreate(
        user_id="u1", content="Sub's sister Riya lives in Paris.",
        entities=[EntityMention(name="Riya", type="person")], instances=[_guitar()],
    ))
    second = await memory.edit("u1", first.id, content="Sub's sister Riya lives in Lyon.")
    async with memory.db.session_scope() as session:
        session.add(Episode(id="ep1", user_id="u1", session_key="s1", summary="Talked about Riya in Lyon."))
        await session.flush()
        session.add(EpisodeMemory(episode_id="ep1", memory_id=second.id))
        await session.execute(update(User).where(User.id == "u1").values(profile="Sub's sister lives in Lyon."))

    assert await memory.delete("u1", second.id) == 2  # current version + the older one
    assert await _count(memory, Memory, Memory.user_id == "u1") == 0
    assert await _count(memory, Episode, Episode.user_id == "u1") == 0
    assert await _count(memory, Entity, Entity.user_id == "u1") == 0  # Riya is no longer mentioned
    async with memory.db.session_scope() as session:
        assert await session.scalar(select(User.profile).where(User.id == "u1")) == ""


async def test_deleting_an_old_version_keeps_the_newer_one(memory):
    first = await memory.save(MemoryCreate(user_id="u1", content="Sub lives in Pune."))
    second = await memory.edit("u1", first.id, content="Sub lives in Bangalore.")
    assert await memory.delete("u1", first.id) == 1
    assert (await memory.get("u1", second.id)).content == "Sub lives in Bangalore."


async def test_delete_never_touches_another_users_memory(memory):
    m = await memory.save(MemoryCreate(user_id="u1", content="Sub lives in Pune."))
    assert await memory.delete("u2", m.id) == 0
    assert await memory.get("u1", m.id) is not None


async def test_delete_user_removes_everything(memory):
    await memory.save(MemoryCreate(user_id="u1", content="Sub lives in Pune.", instances=[_guitar()]))
    await memory.save(MemoryCreate(user_id="u2", content="Ana lives in Rome."))
    assert await memory.delete_user("u1") is True
    assert await _count(memory, Memory, Memory.user_id == "u1") == 0
    assert await _count(memory, Instance, Instance.user_id == "u1") == 0
    assert await _count(memory, Memory, Memory.user_id == "u2") == 1
    assert await memory.delete_user("u1") is False


async def test_list_pages_through_everything(memory):
    for i in range(5):
        await memory.save(MemoryCreate(user_id="u1", content=f"Fact number {i} about Sub."))
    pages = [await memory.list_memories("u1", limit=2, offset=o) for o in (0, 2, 4)]
    ids = [m.id for page in pages for m in page]
    assert [len(p) for p in pages] == [2, 2, 1] and len(set(ids)) == 5


# ---------- export / import ----------


async def _rich_user(mem):
    first = await mem.save(MemoryCreate(
        user_id="u1", content="Sub's sister Riya lives in Paris.", kind="fact", importance=4,
        entities=[EntityMention(name="Riya", type="person"), EntityMention(name="Sub", type="person")],
        relations=[RelationMention(subject="Riya", predicate="sister of", target="Sub")],
    ))
    await mem.edit("u1", first.id, content="Sub's sister Riya lives in Lyon.")
    await mem.save(MemoryCreate(user_id="u1", content="Sub owns a Fender guitar.", instances=[_guitar()]))
    forgotten = await mem.save(MemoryCreate(user_id="u1", content="Sub's old PIN hint."))
    await mem.forget("u1", forgotten.id)


async def _snapshot(mem, user_id):
    async with mem.db.session_scope() as session:
        memories = (await session.scalars(select(Memory).where(Memory.user_id == user_id))).all()
        instances = (await session.scalars(select(Instance).where(Instance.user_id == user_id))).all()
        entities = (await session.scalars(select(Entity).where(Entity.user_id == user_id))).all()
    by_id = {m.id: m.content for m in memories}
    return (
        sorted((m.content, m.is_active, by_id.get(m.superseded_by_id)) for m in memories),
        sorted((i.subject, i.instance_key, i.status) for i in instances),
        sorted(e.name for e in entities),
    )


async def test_export_then_import_into_an_empty_database_restores_everything(memory, provider, db_url):
    await _rich_user(memory)
    buffer = io.StringIO()
    counts = await memory.export_user("u1", buffer)
    assert counts["memory"] == 4 and counts["entity"] == 2 and counts["relation"] == 1
    before = await _snapshot(memory, "u1")

    await memory.delete_user("u1")
    buffer.seek(0)
    await memory.import_user(buffer)
    assert await _snapshot(memory, "u1") == before
    # vectors were left out, so the restored memories wait for reembed_stale
    assert await _count(memory, Memory, Memory.user_id == "u1", Memory.embedding.is_not(None)) == 0


async def test_importing_the_same_file_twice_changes_nothing(memory):
    await _rich_user(memory)
    buffer = io.StringIO()
    await memory.export_user("u1", buffer)
    before = await _snapshot(memory, "u1")
    for _ in range(2):
        buffer.seek(0)
        await memory.import_user(buffer)
    assert await _snapshot(memory, "u1") == before


async def test_import_as_another_user_makes_an_independent_copy(memory):
    await _rich_user(memory)
    buffer = io.StringIO()
    await memory.export_user("u1", buffer)
    buffer.seek(0)
    await memory.import_user(buffer, as_user="u1-copy")
    assert await _snapshot(memory, "u1-copy") == await _snapshot(memory, "u1")
    await memory.delete_user("u1-copy")
    assert (await _snapshot(memory, "u1"))[0]  # the original is untouched


async def test_import_next_to_existing_memories_skips_duplicates_without_failing(memory):
    await _rich_user(memory)
    buffer = io.StringIO()
    await memory.export_user("u1", buffer)
    buffer.seek(0)
    # same user, but the guitar memory now exists under a different id
    async with memory.db.session_scope() as session:
        guitar = await session.scalar(select(Memory).where(Memory.content == "Sub owns a Fender guitar."))
    await memory.delete("u1", guitar.id)
    await memory.save(MemoryCreate(user_id="u1", content="Sub owns a Fender guitar."))
    counts = await memory.import_user(buffer)
    assert counts["memory"] == 4
    assert await _count(memory, Memory, Memory.content == "Sub owns a Fender guitar.") == 1


async def test_vectors_are_kept_only_for_the_current_embedding_model(memory, db_url):
    await memory.save(MemoryCreate(user_id="u1", content="Bruno is my dog.", semantic=True))
    buffer = io.StringIO()
    await memory.export_user("u1", buffer, include_embeddings=True)
    lines = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert lines[0]["_type"] == "header" and any(r.get("embedding") for r in lines[1:])

    await memory.delete_user("u1")
    buffer.seek(0)
    await memory.import_user(buffer)
    assert await _count(memory, Memory, Memory.embedding.is_not(None)) == 1

    class OtherModel(FakeProvider):
        embedding_model_id = "another-model"

    other = RikiMemory(database_url=db_url, llm=OtherModel(), config=MemoryConfig())
    await other.init()
    try:
        await other.delete_user("u1")
        buffer.seek(0)
        await other.import_user(buffer)
        assert await _count(other, Memory, Memory.embedding.is_not(None)) == 0
    finally:
        await other.close()


async def test_import_rejects_a_file_that_is_not_an_export(memory):
    import pytest

    with pytest.raises(ValueError, match="not a riki-memory export"):
        await memory.import_user(io.StringIO('{"hello": 1}\n'))


# ---------- namespaces ----------


async def test_namespaces_have_a_tenant_and_a_kind(memory):
    ns = await memory.create_namespace("acme/eng-team", tenant_id="acme", kind="team", name="Engineering")
    assert (ns.tenant_id, ns.kind, ns.name) == ("acme", "team", "Engineering")
    again = await memory.create_namespace("acme/eng-team", tenant_id="acme", kind="team")
    assert again.id == ns.id  # idempotent


async def test_implicit_namespaces_belong_to_the_default_tenant(memory):
    await memory.save(MemoryCreate(user_id="solo", content="Solo user fact."))
    (ns,) = await memory.list_namespaces("default")
    assert (ns.id, ns.tenant_id, ns.kind) == ("solo", "default", "user")


async def test_a_namespace_is_never_moved_to_another_tenant_silently(memory):
    import pytest

    await memory.create_namespace("acme/alice", tenant_id="acme")
    with pytest.raises(ValueError, match="already exists"):
        await memory.create_namespace("acme/alice", tenant_id="globex")
    with pytest.raises(ValueError, match="kind must be one of"):
        await memory.create_namespace("x", kind="planet")


async def test_delete_tenant_removes_only_that_tenant(memory):
    for user in ("acme/alice", "acme/bob"):
        await memory.create_namespace(user, tenant_id="acme")
        await memory.save(MemoryCreate(user_id=user, content=f"{user} likes tea."))
    await memory.create_namespace("globex/carol", tenant_id="globex")
    await memory.save(MemoryCreate(user_id="globex/carol", content="Carol likes coffee."))

    assert await memory.delete_tenant("acme") == 2
    assert [n.id for n in await memory.list_namespaces()] == ["globex/carol"]
    assert await _count(memory, Memory, Memory.user_id.like("acme/%")) == 0


async def test_export_keeps_the_namespace_owner(memory):
    await memory.create_namespace("acme/alice", tenant_id="acme", kind="user")
    await memory.save(MemoryCreate(user_id="acme/alice", content="Alice likes tea."))
    buffer = io.StringIO()
    await memory.export_user("acme/alice", buffer)
    await memory.delete_user("acme/alice")
    buffer.seek(0)
    await memory.import_user(buffer)
    (ns,) = await memory.list_namespaces("acme")
    assert (ns.id, ns.kind) == ("acme/alice", "user")
