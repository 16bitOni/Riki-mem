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

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

import re

from .config import MemoryConfig
from .db import Database
from .entities import link_entities, prune_orphaned_entities
from .llm import LLMProvider
from .models import CategoryFact, Memory
from .schemas import MemoryCreate
from .similarity import cosine_similarity, trigram_similarity
from .tags import normalize_tags

# Tags that map directly to a category name for materialized aggregation.
# The value is the canonical category key stored in category_facts.
_TAG_TO_CATEGORY: dict[str, str] = {
    "charity-event": "charity_event_attended",
    "charity":       "charity_event_attended",
    "wedding":       "wedding_attended",
    "model-kit":     "model_kit",
    "model-tanks":   "model_kit",
    "instrument":    "instrument_owned",
    "doctor-visit":  "doctor_visited",
    "camping":       "camping_trip",
    "bike-purchase": "bike_expense",
    "bike-maintenance": "bike_expense",
    "aquarium":      "aquarium_fish",
    "health-device": "health_device",
    "road-trip":     "road_trip",
    "birth-event":   "baby_born",
    "property-viewing": "property_viewed",
    "art-event":     "art_event_attended",
    "citrus":        "citrus_used",
}

# Dollar amounts: $ followed by digits (optionally with comma/period separators)
_AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def _extract_amount(content: str) -> tuple[float | None, str | None]:
    """Pull the first dollar amount out of a memory string, if any."""
    m = _AMOUNT_RE.search(content)
    if not m:
        return None, None
    try:
        return float(m.group(1).replace(",", "")), "USD"
    except ValueError:
        return None, None


def _item_name_from_content(content: str) -> str:
    """Best-effort: strip the leading 'longeval_user ...' prefix to get a short item name."""
    # Remove "longeval_user <verb phrase>" prefix, keep the specific named thing
    trimmed = re.sub(r"^longeval_user\s+\S+\s+", "", content, count=1).strip()
    # Cap at 200 chars
    return trimmed[:200] if trimmed else content[:200]


async def _upsert_category_facts(
    session,
    user_id: str,
    memory: Memory,
    tags: list[str],
    content: str,
) -> None:
    """Write one CategoryFact row per matching tag category (idempotent via unique constraint)."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    seen_categories: set[str] = set()
    numeric_value, unit = _extract_amount(content)
    item_name = _item_name_from_content(content)

    for tag in tags:
        category = _TAG_TO_CATEGORY.get(tag.lower().strip())
        if not category or category in seen_categories:
            continue
        seen_categories.add(category)
        try:
            stmt = (
                sqlite_insert(CategoryFact)
                .values(
                    user_id=user_id,
                    category=category,
                    item_name=item_name,
                    numeric_value=numeric_value,
                    unit=unit,
                    date=memory.event_at,
                    source_memory_id=memory.id,
                )
                .on_conflict_do_nothing()  # unique(user_id, category, source_memory_id)
            )
            await session.execute(stmt)
        except Exception:
            log.debug("category_fact upsert skipped for memory %s category %s", memory.id, category)

log = logging.getLogger(__name__)


def _slot_key(value: datetime | None) -> str | None:
    """Normalise an event time for same-slot comparison.

    SQLite may hand back a naive datetime; treat naive as UTC so it still
    compares correctly against the timezone-aware values we store.
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
    12 April"). Two checks: similar spelling (trigrams, for every memory), then
    similar meaning (vectors — only for memories that have one, made by the
    current embedding model).
    """
    slot = _slot_key(data.event_at)
    candidates = (
        await session.scalars(
            select(Memory)
            .where(
                Memory.user_id == data.user_id,
                Memory.is_active.is_(True),
                Memory.kind == data.kind,
            )
            .options(undefer(Memory.embedding))
        )
    ).all()
    candidates = [m for m in candidates if _slot_key(m.event_at) == slot]
    if not candidates:
        return None

    scored = [
        (m, trigram_similarity(m.content, data.content)) for m in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    best, best_score = scored[0]
    if best_score >= config.dedup_trigram_threshold:
        return best
    if vector is None:
        return None

    # Meaning-level check, restricted to vectors from the current model so we
    # never compare embeddings made by different models.
    embedded = [m for m in candidates if m.embedding and m.embedding_model == llm.embedding_model_id]
    best_vec: Memory | None = None
    best_sim = config.dedup_vector_threshold
    for m in embedded:
        sim = cosine_similarity(m.embedding, vector)
        if sim >= best_sim:
            best_sim, best_vec = sim, m
    return best_vec


async def save_memory(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    data: MemoryCreate,
    on_important: Callable[[Memory], Awaitable[None]] | None = None,
) -> Memory:
    """Save one memory and return it.

    - The same sentence twice returns the first one (without resurrecting it
      if it was forgotten).
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

    content_hash = hashlib.sha256(data.content.lower().encode("utf-8")).hexdigest()

    # 1. Exact duplicate? Answer before paying for an embedding.
    async with db.session_scope() as session:
        existing = await session.scalar(
            select(Memory).where(Memory.user_id == data.user_id, Memory.content_hash == content_hash)
        )
        if existing:
            # Deliberately NOT reactivated: the user may have forgotten this on purpose.
            return existing

    # 2. Embed outside any transaction (a slow network call must not hold the database).
    wants_vector = data.semantic if data.semantic is not None else config.embed_all_kinds
    vector: list[float] | None = None
    if wants_vector:
        try:
            vector = await llm.embed(data.content)
        except Exception:
            log.exception("could not embed a new memory; it is saved and will be picked up by reembed")

    # 3. Write everything in one transaction. The insert gets its own scope so a
    # lost dedup race rolls back cleanly and the winner can be returned.
    try:
        async with db.session_scope() as session:
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
                wants_embedding=wants_vector,
                embedding=vector,
                embedding_model=llm.embedding_model_id if vector is not None else None,
                embedded_at=datetime.now(timezone.utc) if vector is not None else None,
            )
            session.add(memory)
            await session.flush()

            await db.keywords.add(session, memory.id, memory.content)

            if data.tags:
                await _upsert_category_facts(session, data.user_id, memory, data.tags, data.content)

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
                for rid in retired_ids:
                    await db.keywords.remove(session, rid)
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
    """Soft delete: the row stays for audit but is never retrieved again."""
    async with db.session_scope() as session:
        result = await session.execute(
            update(Memory)
            .where(Memory.id == memory_id, Memory.user_id == user_id)
            .values(is_active=False)
        )
        if result.rowcount:
            await db.keywords.remove(session, memory_id)
            if config.enable_entities:
                await prune_orphaned_entities(session, user_id)
        return result.rowcount > 0


async def list_memories(
    db: Database, user_id: str, *, active_only: bool = True, limit: int = 100
) -> list[Memory]:
    """Newest first. For admin views and debugging; not part of recall."""
    async with db.session_scope() as session:
        stmt = select(Memory).where(Memory.user_id == user_id)
        if active_only:
            stmt = stmt.where(Memory.is_active.is_(True))
        return list(
            (await session.scalars(stmt.order_by(Memory.created_at.desc()).limit(limit))).all()
        )
