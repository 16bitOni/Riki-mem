"""Writing to long-term memory: a row, sometimes an embedding, and graph links.

Everything goes through :func:`save_memory`, so every write has been validated.

Fixes applied while extracting from Riki (see the original teardown):

- an exact duplicate no longer *resurrects* a forgotten memory — the existing
  row is returned untouched;
- the exact-hash check and the insert are no longer racy: an ``IntegrityError``
  on the ``(user_id, content_hash)`` constraint is caught and the existing row
  returned;
- entity linking is optional (``MemoryConfig.enable_entities``);
- the hard-coded cognition trigger is now an ``on_important`` callback the host
  app supplies (or nothing at all).
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .config import MemoryConfig
from .db import Database
from .entities import link_entities, prune_orphaned_entities
from .llm import LLMProvider
from .models import Instance, Memory, User
from .schemas import InstanceItem, MemoryCreate
from .tags import normalize_subjects, normalize_tags

log = logging.getLogger(__name__)


async def _upsert_instances(
    session,
    user_id: str,
    memory: Memory,
    instances: list[InstanceItem],
) -> None:
    """Write one Instance row per extracted instance (idempotent via unique constraint).

    Not wrapped in a try/except: a failed write here means the counts would be
    wrong, so the whole save rolls back and the error surfaces.

    The same ``instance_key`` seen again under another memory is a new row, not
    a merge: each row is one mention with its own status and date, and
    aggregation groups rows by key. Matching keys never retire prose either —
    two bike repairs keyed ``bike-repair`` are two memories.
    """
    for inst in instances:
        stmt = (
            insert(Instance)
            .values(
                user_id=user_id,
                kind=inst.kind,
                subject=inst.subject,
                instance_key=inst.instance_key,
                display_name=inst.display_name,
                status=inst.status,
                date=inst.date or memory.event_at,
                numeric_value=inst.numeric_value,
                unit=inst.unit,
                source_memory_id=memory.id,
            )
            .on_conflict_do_nothing()
        )
        await session.execute(stmt)


async def _repoint_instances(
    session: AsyncSession, user_id: str, retired_ids: list[str], new_id: str
) -> None:
    """Move the Instance rows of retired memories onto the memory that replaced them.

    Without this, a superseded memory takes its instances' evidence with it
    (aggregation only cites active memories). Where the new memory already has
    a row for the same (kind, subject, key), the new row wins and the old one goes.
    """
    rows = (
        await session.scalars(
            select(Instance).where(
                Instance.user_id == user_id, Instance.source_memory_id.in_(retired_ids)
            )
        )
    ).all()
    if not rows:
        return
    taken = set(
        (
            await session.execute(
                select(Instance.kind, Instance.subject, Instance.instance_key).where(
                    Instance.user_id == user_id, Instance.source_memory_id == new_id
                )
            )
        ).all()
    )
    for row in rows:
        identity = (row.kind, row.subject, row.instance_key)
        if identity in taken:
            await session.delete(row)
        else:
            row.source_memory_id = new_id
            taken.add(identity)
    await session.flush()


def _slot_key(value: datetime | None) -> str | None:
    """Normalise an event time for same-slot comparison.

    A naive datetime (from a caller) is treated as UTC so it still compares
    correctly against the timezone-aware values we store.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


async def _find_near_duplicate(
    session: AsyncSession, db: Database, llm: LLMProvider, config: MemoryConfig,
    data: MemoryCreate, vector: list[float] | None,
) -> Memory | None:
    """An existing memory that says (almost) the same thing, or None.

    A candidate must be this user's, still active, the same kind, and about the
    same date (so "dentist on 10 March" is never merged with "dentist on
    12 April"). Two checks, both inside Postgres so nothing but the winner is
    loaded: similar spelling (pg_trgm, every memory), then similar meaning
    (pgvector, only vectors made by the current embedding model).
    """
    when = data.event_at
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    same_slot = [
        Memory.user_id == data.user_id,
        Memory.is_active.is_(True),
        Memory.kind == data.kind,
        Memory.event_at.is_(None) if when is None else Memory.event_at == when,
    ]
    spelling = func.similarity(Memory.content, data.content)
    best = (await session.execute(
        select(Memory, spelling).where(*same_slot).order_by(spelling.desc()).limit(1)
    )).first()
    if best is None:
        return None
    if best[1] >= config.dedup_trigram_threshold:
        return best[0]
    if vector is None:
        return None
    nearest = await db.vectors.nearest(
        session, Memory, same_slot, vector, llm.embedding_model_id, 1, config.dedup_vector_threshold,
        build_index=False,
    )
    return await session.get(Memory, nearest[0][0]) if nearest else None


def memory_hash(data: MemoryCreate) -> str:
    """Exact-duplicate identity: same kind, same date slot, same text (case-insensitive).
    The date is part of it so "went to the dentist" on two days is two visits."""
    key = f"{data.kind}|{_slot_key(data.event_at) or ''}|{data.content.lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def save_memory(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    data: MemoryCreate,
    on_important: Callable[[Memory], Awaitable[None]] | None = None,
    vector: list[float] | None = None,
) -> Memory:
    """Save one memory and return it. ``vector``: its embedding, if the caller
    already made it (extraction embeds a whole turn in one call).

    - The same sentence (same kind and date) twice returns the first one. A
      forgotten memory is not brought back by a retry of its own turn; said
      again in a new turn, it is saved as a new memory.
    - A near-duplicate (same thing, worded differently) is replaced by the
      newer wording; the older row is retired, not deleted, and the higher
      importance of the two is kept.
    - If the embedding fails, the memory is still saved and flagged so
      ``reembed_stale`` can finish the job.
    """
    if data.tags:
        data = data.model_copy(update={
            "tags": await normalize_tags(
                llm, db, data.user_id, data.tags, config.tag_similarity_threshold
            )
        })

    if data.instances:
        subjects = await normalize_subjects(
            llm, db, data.user_id, [inst.subject for inst in data.instances],
            config.tag_similarity_threshold,
        )
        data = data.model_copy(update={"instances": [
            inst.model_copy(update={"subject": subject})
            for inst, subject in zip(data.instances, subjects)
        ]})

    content_hash = memory_hash(data)

    # 1. Exact duplicate? Answer before paying for an embedding.
    async with db.session_scope() as session:
        existing = await session.scalar(
            select(Memory).where(Memory.user_id == data.user_id, Memory.content_hash == content_hash)
        )
        if existing and (existing.is_active or (
            data.source_key is not None and existing.source_key == data.source_key
        )):
            # A retry of the turn that produced a forgotten memory must not bring it back.
            return existing
        if existing:
            # Forgotten or replaced, and now said again in a new turn: a new statement.
            # The old row stays for audit; it only gives up its hash.
            existing.content_hash = hashlib.sha256(f"retired|{existing.id}".encode()).hexdigest()

    # 2. Embed outside any transaction (a slow network call must not hold the database).
    wants_vector = data.semantic if data.semantic is not None else config.embed_all_kinds
    if not wants_vector:
        vector = None
    elif vector is None:
        try:
            vector = await llm.embed(data.content)
        except Exception:
            log.exception("could not embed a new memory; it is saved and will be picked up by reembed")

    # 3. Write everything in one transaction. The insert gets its own scope so a
    # lost dedup race rolls back cleanly and the winner can be returned.
    try:
        async with db.session_scope() as session:
            # The memory's user row must exist first (Postgres enforces the foreign key).
            await session.execute(
                insert(User)
                .values(id=data.user_id, name=data.user_id, timezone=config.default_timezone)
                .on_conflict_do_nothing()
            )
            near = await _find_near_duplicate(session, db, llm, config, data, vector)
            memory = Memory(
                user_id=data.user_id,
                kind=data.kind,
                content=data.content,
                content_hash=content_hash,
                importance=max(data.importance, near.importance) if near else data.importance,
                event_at=data.event_at,
                source_key=data.source_key,
                tags=data.tags or [],
                attributes=data.attributes or {},
                valid_to=data.valid_to,
                source=data.source,
                wants_embedding=wants_vector,
                embedding=vector,
                embedding_model=llm.embedding_model_id if vector is not None else None,
                embedded_at=datetime.now(timezone.utc) if vector is not None else None,
            )
            session.add(memory)
            await session.flush()

            if data.instances:
                await _upsert_instances(session, data.user_id, memory, data.instances)

            if config.enable_entities:
                await link_entities(session, data.user_id, memory.id, data.entities, data.relations)

            retire = set(data.replaces)
            if near:
                log.debug("memory %s replaces near-duplicate %s", memory.id, near.id)
                retire.add(near.id)
            retired_ids = [rid for rid in retire if rid != memory.id]
            if retired_ids:
                await session.execute(
                    update(Memory)
                    .where(Memory.id.in_(retired_ids), Memory.user_id == data.user_id)
                    .values(is_active=False, superseded_by_id=memory.id)
                )
                changed = [rid for rid in data.replaces if rid in retired_ids]
                if data.changed_at and changed:
                    # No longer true, but it was: keep it findable as history.
                    await session.execute(
                        update(Memory)
                        .where(Memory.id.in_(changed), Memory.user_id == data.user_id,
                               Memory.valid_to.is_(None))
                        .values(valid_to=data.changed_at)
                    )
                await _repoint_instances(session, data.user_id, retired_ids, memory.id)
    except IntegrityError:
        # Lost a race with an identical save: return the winner instead of crashing.
        log.debug("dedup race on '%s…'; returning the existing row", data.content[:40])
        async with db.session_scope() as session:
            existing = await session.scalar(
                select(Memory).where(
                    Memory.user_id == data.user_id, Memory.content_hash == content_hash
                )
            )
        if existing is None:  # pragma: no cover - defensive; the constraint says it exists
            raise
        return existing

    # 4. The host app decides what "important" means. Awaiting the callback keeps
    # delivery honest (the original fire-and-forget task could die silently).
    if memory.importance >= 4 and on_important is not None:
        try:
            await on_important(memory)
        except Exception:
            log.exception("on_important callback failed for memory %s", memory.id)

    return memory


async def get_memory(db: Database, memory_id: str, user_id: str) -> Memory | None:
    async with db.session_scope() as session:
        return await session.scalar(
            select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
        )


async def forget_memory(db: Database, config: MemoryConfig, user_id: str, memory_id: str) -> bool:
    """Soft delete: the row stays for audit but is never retrieved again.

    Its Instance rows are hard-deleted, so a forgotten thing stops being
    counted (the Memory row remains as the audit record).
    """
    async with db.session_scope() as session:
        result = await session.execute(
            update(Memory)
            .where(Memory.id == memory_id, Memory.user_id == user_id)
            .values(is_active=False, forgotten_at=datetime.now(timezone.utc))
        )
        if result.rowcount:
            await session.execute(
                delete(Instance).where(
                    Instance.user_id == user_id, Instance.source_memory_id == memory_id
                )
            )
            if config.enable_entities:
                await prune_orphaned_entities(session, user_id)
        return result.rowcount > 0


async def list_memories(
    db: Database, user_id: str, *, active_only: bool = True, limit: int = 100, offset: int = 0,
    attributes: dict | None = None, source: str | None = None,
) -> list[Memory]:
    """Newest first. ``attributes``: only memories whose attributes contain these
    key/values (e.g. ``{"type": "medication"}``). ``source``: only ``"stated"``,
    ``"inferred"`` (what was guessed about the user) or ``"assistant"`` memories."""
    async with db.session_scope() as session:
        stmt = select(Memory).where(Memory.user_id == user_id)
        if source:
            stmt = stmt.where(Memory.source == source)
        if attributes:
            stmt = stmt.where(Memory.attributes.contains(attributes))
        if active_only:
            stmt = stmt.where(Memory.is_active.is_(True))
        return list(
            (await session.scalars(
                stmt.order_by(Memory.created_at.desc(), Memory.id).limit(limit).offset(offset)
            )).all()
        )
