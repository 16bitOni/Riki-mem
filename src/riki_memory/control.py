"""What a user can do with their own memory: see it, fix it, delete it for real,
and take it with them.

- :func:`edit_memory` — change a memory. A text change saves a new version that
  supersedes the old one (the history stays auditable); a change to only the
  importance, tags, date or kind is made in place.
- :func:`delete_memory` — hard delete: the row, every older version of it, the
  episode summaries written from it, and the profile text (rebuilt later).
  ``forget`` hides a memory; this removes it.
- :func:`delete_user` — everything stored for one user.
- :func:`export_user` / :func:`import_user` — streaming JSONL, one record per
  line, so years of memories never have to fit in RAM. Import is idempotent:
  run it again after an interruption and finished rows are skipped.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

from sqlalchemy import DateTime, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import undefer

from .config import MemoryConfig
from .db import Database
from .entities import prune_orphaned_entities
from .llm import LLMProvider
from .models import NAMESPACE_KINDS, Entity, EntityRelation, Episode, EpisodeMemory, Instance, Memory, MemoryEntity, User
from .schemas import InstanceItem, MemoryCreate
from .store import save_memory

log = logging.getLogger(__name__)

EXPORT_FORMAT = "riki-memory-export"
EXPORT_VERSION = 1
BATCH = 500
UNCHANGED = object()


# -- edit ---------------------------------------------------------------------


async def edit_memory(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    memory_id: str,
    *,
    content: str | None = None,
    importance: int | None = None,
    kind: str | None = None,
    tags: list[str] | None = None,
    event_at=UNCHANGED,
    instances: list[InstanceItem] | None = None,
    keep_instances: bool = False,
) -> Memory | None:
    """Change one memory; returns the current version, or ``None`` if it doesn't exist.

    When the text changes, the countable instances that were extracted from the
    old text are removed (they may no longer be true) unless you pass
    ``instances`` or ``keep_instances=True``.
    """
    async with db.session_scope() as session:
        old = await session.scalar(select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id))
        if old is None:
            return None
        old_instances = (
            await session.scalars(select(Instance).where(Instance.source_memory_id == memory_id))
        ).all()

    if content is None or content.strip() == old.content:
        values = {}
        if importance is not None:
            values["importance"] = importance
        if kind is not None:
            values["kind"] = kind
        if tags is not None:
            values["tags"] = tags
        if event_at is not UNCHANGED:
            values["event_at"] = event_at
        async with db.session_scope() as session:
            if values:
                await session.execute(update(Memory).where(Memory.id == memory_id).values(**values))
            if instances is not None:
                await session.execute(delete(Instance).where(Instance.source_memory_id == memory_id))
                from .store import _upsert_instances

                await _upsert_instances(session, user_id, old, instances)
            return await session.get(Memory, memory_id, populate_existing=True)

    if instances is None and keep_instances:
        instances = [
            InstanceItem(
                kind=i.kind, subject=i.subject, instance_key=i.instance_key, display_name=i.display_name,
                status=i.status, date=i.date, numeric_value=i.numeric_value, unit=i.unit,
            )
            for i in old_instances
        ]
    # The old memory's instances go now; the new version gets only the ones meant for it.
    async with db.session_scope() as session:
        await session.execute(delete(Instance).where(Instance.source_memory_id == memory_id))

    new = await save_memory(db, llm, config, MemoryCreate(
        user_id=user_id,
        content=content,
        kind=kind or old.kind,
        importance=importance or old.importance,
        event_at=old.event_at if event_at is UNCHANGED else event_at,
        tags=tags if tags is not None else (old.tags or []),
        source_key=old.source_key,
        semantic=old.wants_embedding,
        replaces=[memory_id],
        instances=instances or [],
    ))
    if new.id != memory_id:
        # The new text already existed as another memory: that one is now current.
        async with db.session_scope() as session:
            await session.execute(
                update(Memory)
                .where(Memory.id == memory_id, Memory.is_active.is_(True))
                .values(is_active=False, superseded_by_id=new.id)
            )
    return new


# -- delete -------------------------------------------------------------------


async def delete_memory(db: Database, user_id: str, memory_id: str) -> int:
    """Hard delete one memory and every older version of it. Returns rows deleted.

    Also deleted: the episode summaries that included any of them (they quote
    the text; ``close_session`` can rebuild them), and the user's profile text
    (rebuilt from what remains on the next refresh). Instances, entity links and
    vectors go with the rows. A newer version of the memory is not touched.
    """
    async with db.session_scope() as session:
        doomed: set[str] = set()
        frontier = set(
            (await session.scalars(select(Memory.id).where(Memory.id == memory_id, Memory.user_id == user_id))).all()
        )
        while frontier:
            doomed |= frontier
            frontier = set(
                (await session.scalars(
                    select(Memory.id).where(Memory.user_id == user_id, Memory.superseded_by_id.in_(frontier))
                )).all()
            ) - doomed
        if not doomed:
            return 0

        episode_ids = (
            await session.scalars(select(EpisodeMemory.episode_id).where(EpisodeMemory.memory_id.in_(doomed)))
        ).all()
        if episode_ids:
            await session.execute(delete(Episode).where(Episode.id.in_(set(episode_ids))))
        # Rows that were superseded by a deleted row keep existing only as history of it.
        await session.execute(delete(Memory).where(Memory.id.in_(doomed)))
        await prune_orphaned_entities(session, user_id)
        await session.execute(
            update(User).where(User.id == user_id).values(profile="", profile_updated_at=None)
        )
    return len(doomed)


async def delete_user(db: Database, user_id: str) -> bool:
    """Delete everything stored for ``user_id``: memories, versions, instances,
    entities, episodes and the profile."""
    async with db.session_scope() as session:
        result = await session.execute(delete(User).where(User.id == user_id))
        return result.rowcount > 0


# -- namespaces -----------------------------------------------------------------


async def create_namespace(
    db: Database,
    user_id: str,
    *,
    tenant_id: str = "default",
    kind: str = "user",
    name: str = "",
    timezone_name: str | None = None,
) -> User:
    """Create a namespace, or return it if it already exists with the same owner.

    Namespaces are also created implicitly (tenant "default", kind "user") the
    first time something is saved for a new ``user_id``; call this first to set
    the tenant or kind. Moving an existing namespace to another tenant raises
    instead of happening silently.
    """
    if kind not in NAMESPACE_KINDS:
        raise ValueError(f"kind must be one of {', '.join(NAMESPACE_KINDS)}")
    async with db.session_scope() as session:
        await session.execute(
            insert(User)
            .values(id=user_id, tenant_id=tenant_id, kind=kind, name=name or user_id,
                    timezone=timezone_name or "UTC")
            .on_conflict_do_nothing()
        )
        user = await session.scalar(select(User).where(User.id == user_id))
        if (user.tenant_id, user.kind) != (tenant_id, kind):
            raise ValueError(
                f"namespace {user_id!r} already exists as tenant={user.tenant_id!r}, kind={user.kind!r}"
            )
        return user


async def list_namespaces(
    db: Database, tenant_id: str | None = None, *, kind: str | None = None, limit: int = 100, offset: int = 0
) -> list[User]:
    """Namespaces, oldest first, optionally only one tenant's or one kind."""
    stmt = select(User).order_by(User.created_at, User.id).limit(limit).offset(offset)
    if tenant_id is not None:
        stmt = stmt.where(User.tenant_id == tenant_id)
    if kind is not None:
        stmt = stmt.where(User.kind == kind)
    async with db.session_scope() as session:
        return list((await session.scalars(stmt)).all())


async def delete_tenant(db: Database, tenant_id: str) -> int:
    """Delete every namespace of one tenant and everything in them. Returns namespaces deleted."""
    async with db.session_scope() as session:
        result = await session.execute(delete(User).where(User.tenant_id == tenant_id))
        return result.rowcount


# -- export / import ------------------------------------------------------------


def _plain(row, skip: tuple[str, ...] = ()) -> dict:
    out = {}
    for column in row.__table__.columns:
        if column.key in skip:
            continue
        value = getattr(row, column.key)
        out[column.key] = value.isoformat() if isinstance(value, datetime) else value
    return out


def _typed(model, data: dict) -> dict:
    """Back from JSON: ISO strings to datetimes, unknown keys dropped."""
    out = {}
    for column in model.__table__.columns:
        if column.key not in data:
            continue
        value = data[column.key]
        if value is not None and isinstance(column.type, DateTime):
            value = datetime.fromisoformat(value)
        out[column.key] = value
    return out


async def export_user(
    db: Database,
    user_id: str,
    target: str | Path | IO[str],
    *,
    include_embeddings: bool = False,
    since: datetime | None = None,
) -> dict:
    """Write one user's memory as JSONL. Returns counts per record type.

    Lines: one ``header``, then ``entity``, ``relation``, ``memory`` (with its
    ``instances`` and ``entity_ids``), ``episode`` (with ``memory_ids``).
    Forgotten and superseded memories are included, so the history survives.
    ``since``: only memories and episodes changed after it (incremental backup;
    entities and relations are always complete, they are small).
    Embeddings are left out by default; ``reembed_stale`` recreates them.
    """
    owns = not hasattr(target, "write")
    out = open(target, "w", encoding="utf-8") if owns else target
    counts = {"entity": 0, "relation": 0, "memory": 0, "episode": 0}

    def write(kind: str, data: dict) -> None:
        out.write(json.dumps({"_type": kind, **data}, ensure_ascii=False) + "\n")
        counts[kind] = counts.get(kind, 0) + 1

    try:
        async with db.session_scope() as session:
            user = await session.scalar(select(User).where(User.id == user_id))
            if user is None:
                raise ValueError(f"no such user: {user_id}")
            write("header", {
                "format": EXPORT_FORMAT, "version": EXPORT_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "since": since.isoformat() if since else None,
                "include_embeddings": include_embeddings,
                "user": _plain(user),
            })
            counts.pop("header", None)

            for entity in (await session.scalars(select(Entity).where(Entity.user_id == user_id))).all():
                write("entity", _plain(entity, skip=("user_id",)))
            for relation in (await session.scalars(
                select(EntityRelation).where(EntityRelation.user_id == user_id)
            )).all():
                write("relation", _plain(relation, skip=("user_id",)))

            stmt = select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at, Memory.id)
            if since:
                stmt = stmt.where(Memory.updated_at > since)
            if include_embeddings:
                stmt = stmt.options(undefer(Memory.embedding))
            stream = await session.stream_scalars(stmt.execution_options(yield_per=BATCH))
            async for page in stream.partitions(BATCH):
                ids = [m.id for m in page]
                instances: dict[str, list] = {}
                for inst in (await session.scalars(select(Instance).where(Instance.source_memory_id.in_(ids)))).all():
                    instances.setdefault(inst.source_memory_id, []).append(
                        _plain(inst, skip=("id", "user_id", "source_memory_id"))
                    )
                links: dict[str, list] = {}
                for link in (await session.scalars(select(MemoryEntity).where(MemoryEntity.memory_id.in_(ids)))).all():
                    links.setdefault(link.memory_id, []).append(link.entity_id)
                for memory in page:
                    write("memory", {
                        **_plain(memory, skip=("user_id",) if include_embeddings else ("user_id", "embedding")),
                        "instances": instances.get(memory.id, []),
                        "entity_ids": links.get(memory.id, []),
                    })

            stmt = select(Episode).where(Episode.user_id == user_id).order_by(Episode.created_at, Episode.id)
            if since:
                stmt = stmt.where(Episode.updated_at > since)
            if include_embeddings:
                stmt = stmt.options(undefer(Episode.embedding))
            for episode in (await session.scalars(stmt)).all():
                memory_ids = (await session.scalars(
                    select(EpisodeMemory.memory_id).where(EpisodeMemory.episode_id == episode.id)
                )).all()
                write("episode", {
                    **_plain(episode, skip=("user_id",) if include_embeddings else ("user_id", "embedding")),
                    "memory_ids": list(memory_ids),
                })
    finally:
        if owns:
            out.close()
    return counts


def _lines(source: str | Path | IO[str]) -> Iterator[dict]:
    owns = not hasattr(source, "read")
    handle = open(source, encoding="utf-8") if owns else source
    try:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"line {number} is not valid JSON: {exc}") from exc
    finally:
        if owns:
            handle.close()


async def import_user(
    db: Database,
    source: str | Path | IO[str],
    *,
    as_user: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Load a file written by :func:`export_user`. Returns counts per record type.

    Rows keep their ids and existing ones are skipped, so a re-run after an
    interruption (or importing the same backup twice) changes nothing.
    ``as_user``: import under another user id, with fresh ids for every row
    (a copy, e.g. into the same database).
    Vectors are kept only when the export has them and they were made by
    ``embedding_model`` (your current model); otherwise the memory is marked
    for ``reembed_stale``.
    """
    lines = _lines(source)
    header = next(lines, None)
    if not header or header.get("_type") != "header" or header.get("format") != EXPORT_FORMAT:
        raise ValueError("not a riki-memory export (missing header)")
    if header.get("version", 0) > EXPORT_VERSION:
        raise ValueError(f"export version {header['version']} is newer than this library understands")

    user_data = _typed(User, header["user"])
    user_id = as_user or user_data["id"]
    new_ids: dict[str, str] = {}

    def remap(old: str | None) -> str | None:
        if old is None or as_user is None:
            return old
        return new_ids.setdefault(old, uuid.uuid4().hex)

    counts = {"entity": 0, "relation": 0, "memory": 0, "episode": 0, "skipped_links": 0}
    superseded: dict[str, str] = {}
    pending: dict[type, list[dict]] = {}

    async def insert_rows(session, model, rows: list[dict]) -> None:
        for start in range(0, len(rows), BATCH):
            await session.execute(insert(model).values(rows[start:start + BATCH]).on_conflict_do_nothing())

    async def present(session, model, ids: set) -> set:
        found: set = set()
        ids = list(ids)
        for start in range(0, len(ids), BATCH):
            found |= set((await session.scalars(select(model.id).where(model.id.in_(ids[start:start + BATCH])))).all())
        return found

    async def flush(session) -> None:
        # Parents first. A parent can be skipped (this user already has the same text,
        # entity name or session), so only link to the ones that are really there.
        for model in (Entity, Memory, Episode):
            await insert_rows(session, model, pending.pop(model, []))
        children = {model: pending.pop(model, []) for model in (EntityRelation, Instance, MemoryEntity, EpisodeMemory)}
        entities = await present(session, Entity, {
            *(r["subject_id"] for r in children[EntityRelation]), *(r["target_id"] for r in children[EntityRelation]),
            *(r["entity_id"] for r in children[MemoryEntity]),
        })
        memories = await present(session, Memory, {
            *(r["source_memory_id"] for r in children[Instance]),
            *(r["memory_id"] for r in children[MemoryEntity]), *(r["memory_id"] for r in children[EpisodeMemory]),
        })
        episodes = await present(session, Episode, {r["episode_id"] for r in children[EpisodeMemory]})
        keep = {
            EntityRelation: lambda r: r["subject_id"] in entities and r["target_id"] in entities,
            Instance: lambda r: r["source_memory_id"] in memories,
            MemoryEntity: lambda r: r["memory_id"] in memories and r["entity_id"] in entities,
            EpisodeMemory: lambda r: r["episode_id"] in episodes and r["memory_id"] in memories,
        }
        for model, rows in children.items():
            kept = [r for r in rows if keep[model](r)]
            counts["skipped_links"] += len(rows) - len(kept)
            await insert_rows(session, model, kept)

    async with db.session_scope() as session:
        user_data["id"] = user_id
        await session.execute(insert(User).values(**user_data).on_conflict_do_nothing())

    for record in lines:
        kind = record.pop("_type", None)
        if kind == "entity":
            row = _typed(Entity, record)
            row.update(id=remap(row["id"]), user_id=user_id)
            pending.setdefault(Entity, []).append(row)
        elif kind == "relation":
            row = _typed(EntityRelation, record)
            row.update(id=remap(row["id"]), user_id=user_id,
                       subject_id=remap(row["subject_id"]), target_id=remap(row["target_id"]))
            pending.setdefault(EntityRelation, []).append(row)
        elif kind == "memory":
            row = _typed(Memory, record)
            if row.get("superseded_by_id"):
                superseded[remap(row["id"])] = remap(row.pop("superseded_by_id"))
            row.update(id=remap(row["id"]), user_id=user_id, superseded_by_id=None)
            if not (row.get("embedding") and embedding_model and row.get("embedding_model") == embedding_model):
                row.update(embedding=None, embedding_model=None, embedded_at=None)
            pending.setdefault(Memory, []).append(row)
            for inst in record.get("instances", []):
                pending.setdefault(Instance, []).append({
                    **_typed(Instance, inst), "id": uuid.uuid4().hex if as_user else _instance_id(row["id"], inst),
                    "user_id": user_id, "source_memory_id": row["id"],
                })
            for entity_id in record.get("entity_ids", []):
                pending.setdefault(MemoryEntity, []).append({"memory_id": row["id"], "entity_id": remap(entity_id)})
        elif kind == "episode":
            row = _typed(Episode, record)
            row.update(id=remap(row["id"]), user_id=user_id)
            if not (row.get("embedding") and embedding_model and row.get("embedding_model") == embedding_model):
                row.update(embedding=None, embedding_model=None)
            pending.setdefault(Episode, []).append(row)
            for memory_id in record.get("memory_ids", []):
                pending.setdefault(EpisodeMemory, []).append({"episode_id": row["id"], "memory_id": remap(memory_id)})
        else:
            log.warning("skipping unknown record type %r", kind)
            continue
        counts[kind] += 1
        if sum(len(rows) for rows in pending.values()) >= BATCH:
            async with db.session_scope() as session:
                await flush(session)

    async with db.session_scope() as session:
        await flush(session)
        # Versions point at newer versions, which may come later in the file (or, in an
        # incremental export, not at all): link them last, and only to rows that exist.
        targets = await present(session, Memory, set(superseded.values()))
        for old_id, new_id in superseded.items():
            if new_id in targets:
                await session.execute(
                    update(Memory).where(Memory.id == old_id, Memory.superseded_by_id.is_(None))
                    .values(superseded_by_id=new_id)
                )
    if counts["skipped_links"]:
        log.info("import: %d links skipped because their memory, entity or episode already "
                 "existed under another id", counts["skipped_links"])
    return counts


def _instance_id(memory_id: str, inst: dict) -> str:
    """Stable id for an imported instance, so a re-import finds it already there."""
    key = f"{memory_id}|{inst.get('kind')}|{inst.get('subject')}|{inst.get('instance_key')}"
    return uuid.uuid5(uuid.NAMESPACE_URL, key).hex
