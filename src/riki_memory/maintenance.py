"""Keeping the index healthy: check it, re-embed after a model change, merge
paraphrase drift, and rebuild from history.

``consolidate_memories`` used to be reachable only from an admin CLI — nothing
ever called it, so drift accumulated silently. It is now a first-class method
on :class:`riki_memory.RikiMemory`; schedule it (weekly cron is plenty) or call
it by hand.
"""

import asyncio
import logging
import uuid as uuid_mod
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import undefer

from .config import MemoryConfig
from .db import Database
from .entities import copy_entity_links
from .extraction import propose_memories
from .llm import LLMProvider
from .models import Instance, Memory, MemoryEntity
from .schemas import MemoryCreate
from .similarity import cosine_similarity, trigram_similarity
from .store import save_memory

log = logging.getLogger(__name__)

BATCH = 50


def _needs_embedding(llm: LLMProvider):
    """A vector that needs (re)computing: wanted but missing, or made by a different model."""
    return Memory.wants_embedding.is_(True) & (
        Memory.embedding.is_(None) | (Memory.embedding_model != llm.embedding_model_id)
    )


async def memory_health(db: Database, llm: LLMProvider, user_id: str | None = None) -> dict:
    """Counts that show whether search can see all the memories that should be searchable."""
    scope = [Memory.user_id == user_id] if user_id else []
    async with db.session_scope() as session:
        total = await session.scalar(select(func.count()).select_from(Memory).where(*scope))
        active = await session.scalar(
            select(func.count()).select_from(Memory).where(*scope, Memory.is_active.is_(True))
        )
        pending = await session.scalar(
            select(func.count()).select_from(Memory).where(*scope, _needs_embedding(llm))
        )
        by_model = (
            await session.execute(
                select(Memory.embedding_model, func.count())
                .where(*scope, Memory.embedding.is_not(None))
                .group_by(Memory.embedding_model)
            )
        ).all()
    models = {model: count for model, count in by_model}
    return {
        "current_model": llm.embedding_model_id,
        "memories": total,
        "active": active,
        "embedded_by_model": models,
        "needs_embedding": pending,  # missing, or made by an older model
        "healthy": pending == 0,
    }


async def reembed_stale(
    db: Database, llm: LLMProvider, config: MemoryConfig, user_id: str | None = None,
    limit: int | None = None,
) -> dict:
    """Compute vectors for every memory that needs one. Safe to stop and run again.

    Each memory is updated on its own, so a crash loses nothing. A memory that fails is
    skipped for the rest of this run (so one bad row cannot loop forever) and counted.
    """
    scope = [Memory.user_id == user_id] if user_id else []
    done, failed_ids = 0, set()
    gate = asyncio.Semaphore(config.reembed_concurrency)

    async def refresh(memory_id: str, content: str) -> bool:
        async with gate:
            try:
                vector = await llm.embed(content)
            except Exception:
                log.exception("re-embedding %s failed", memory_id)
                return False
        async with db.session_scope() as session:
            await session.execute(
                update(Memory)
                .where(Memory.id == memory_id)
                .values(
                    embedding=vector,
                    embedding_model=llm.embedding_model_id,
                    embedded_at=datetime.now(timezone.utc),
                )
            )
        return True

    while limit is None or done + len(failed_ids) < limit:
        take = BATCH if limit is None else min(BATCH, limit - done - len(failed_ids))
        async with db.session_scope() as session:
            rows = (
                await session.execute(
                    select(Memory.id, Memory.content)
                    .where(*scope, _needs_embedding(llm), *([Memory.id.not_in(failed_ids)] if failed_ids else []))
                    .order_by(Memory.is_active.desc(), Memory.created_at.desc())  # live memories first
                    .limit(take)
                )
            ).all()
        if not rows:
            break
        outcomes = await asyncio.gather(*(refresh(i, c) for i, c in rows))
        for (memory_id, _), ok in zip(rows, outcomes):
            if ok:
                done += 1
            else:
                failed_ids.add(memory_id)
    return {"embedded": done, "failed": len(failed_ids)}


async def reextract_turns(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    turns: list[dict],
    dry_run: bool = False,
) -> dict:
    """Run the extraction prompt again over past turns (after improving the prompt or model).

    ``turns`` is a list of ``{"user": ..., "assistant": ..., "at": datetime, "key": ...}``,
    oldest first. Turns are replayed with their original timestamp, so "tomorrow"
    means the right day. Memories already saved are recognised (same sentence, or a
    near-duplicate that the newer wording replaces), so running this twice does not
    pile up copies. With ``dry_run`` nothing is saved; the result lists what would have been.
    """
    saved = proposed = failed = 0
    preview: list[str] = []
    for turn in turns:
        try:
            found = await propose_memories(
                db, llm, config, user_id, turn["user"], turn["assistant"],
                known=[], turn_key=turn.get("key"), at=turn.get("at"),
            )
            proposed += len(found)
            for data in found:
                if dry_run:
                    preview.append(data.content)
                else:
                    await save_memory(db, llm, config, data)
                    saved += 1
        except Exception:
            log.exception("re-extraction failed for turn %s", turn.get("key"))
            failed += 1
    return {
        "turns": len(turns), "proposed": proposed, "saved": saved, "failed": failed,
        "dry_run": dry_run, "preview": preview[:20],
    }


# ---------------------------------------------------------------- memory consolidation


class ConsolidatedMemory(BaseModel):
    content: str = Field(description="Consolidated memory preserving all key information")
    kind: str = Field(description="Memory kind: fact, preference, event, or episode")
    importance: int = Field(ge=1, le=5, description="Importance of the consolidated memory")


class ConsolidationResult(BaseModel):
    memories: list[ConsolidatedMemory] = Field(description="1-2 consolidated memories")


CONSOLIDATION_PROMPT = """You are consolidating related memories about a person into fewer, richer entries.

Below are {count} related memories. Merge them into 1-2 concise memories that preserve ALL important information.
- Keep specific names, dates, numbers, and details
- Prefer the most recent phrasing when wording conflicts
- The result should be self-contained and understandable without the originals
- Set importance to the maximum of the originals

Memories to consolidate:
{memories}"""


async def _find_memory_groups(
    db: Database, config: MemoryConfig, user_id: str
) -> list[list[Memory]]:
    """Group active memories by shared entity (or by kind when entities are disabled)."""
    async with db.session_scope() as session:
        if config.enable_entities:
            rows = (
                await session.execute(
                    select(MemoryEntity.entity_id, Memory)
                    .join(Memory, Memory.id == MemoryEntity.memory_id)
                    .where(Memory.user_id == user_id, Memory.is_active.is_(True))
                    .order_by(MemoryEntity.entity_id, Memory.created_at)
                    .options(undefer(Memory.embedding))
                )
            ).all()
            groups: dict[str, list[Memory]] = defaultdict(list)
            for entity_id, memory in rows:
                groups[entity_id].append(memory)
            # Memories with no entity links still deserve a look: group them by kind.
            linked_ids = {m.id for ms in groups.values() for m in ms}
            unlinked = (
                await session.scalars(
                    select(Memory)
                    .where(Memory.user_id == user_id, Memory.is_active.is_(True))
                    .options(undefer(Memory.embedding))
                )
            ).all()
            by_kind: dict[str, list[Memory]] = defaultdict(list)
            for m in unlinked:
                if m.id not in linked_ids:
                    by_kind[m.kind].append(m)
            return [g for g in list(groups.values()) + list(by_kind.values()) if len(g) >= 2]

        rows = (
            await session.scalars(
                select(Memory)
                .where(Memory.user_id == user_id, Memory.is_active.is_(True))
                .options(undefer(Memory.embedding))
            )
        ).all()
        by_kind = defaultdict(list)
        for m in rows:
            by_kind[m.kind].append(m)
        return [g for g in by_kind.values() if len(g) >= 2]


def _cluster_vectors(memories: list[Memory], threshold: float) -> list[list[Memory]]:
    """Simple greedy clustering by cosine similarity on existing embeddings."""
    assigned: set[str] = set()
    clusters: list[list[Memory]] = []
    for i, anchor in enumerate(memories):
        if anchor.id in assigned:
            continue
        cluster = [anchor]
        assigned.add(anchor.id)
        for candidate in memories[i + 1 :]:
            if candidate.id in assigned:
                continue
            if cosine_similarity(anchor.embedding, candidate.embedding) >= threshold:
                cluster.append(candidate)
                assigned.add(candidate.id)
        if len(cluster) >= 2:
            clusters.append(cluster)
    return clusters


def _cluster_by_kind(memories: list[Memory]) -> list[list[Memory]]:
    """Fallback clustering: group by kind when no vectors are available."""
    by_kind: dict[str, list[Memory]] = defaultdict(list)
    for m in memories:
        by_kind[m.kind].append(m)
    return [group for group in by_kind.values() if len(group) >= 2]


async def _cluster_by_similarity(
    memories: list[Memory], threshold: float
) -> list[list[Memory]]:
    if len(memories) < 2:
        return []
    embedded = [m for m in memories if m.embedding]
    if len(embedded) >= 2:
        return _cluster_vectors(embedded, threshold)
    # Trigram fallback before giving up on content similarity entirely.
    trigram_clusters: list[list[Memory]] = []
    assigned: set[str] = set()
    for i, anchor in enumerate(memories):
        if anchor.id in assigned:
            continue
        cluster = [anchor]
        assigned.add(anchor.id)
        for candidate in memories[i + 1 :]:
            if candidate.id in assigned:
                continue
            if trigram_similarity(anchor.content, candidate.content) >= threshold:
                cluster.append(candidate)
                assigned.add(candidate.id)
        if len(cluster) >= 2:
            trigram_clusters.append(cluster)
    return trigram_clusters or _cluster_by_kind(memories)


async def _consolidate_cluster(
    db: Database, llm: LLMProvider, cluster: list[Memory], user_id: str
) -> list[MemoryCreate]:
    """Use the LLM to merge a cluster into 1-2 consolidated summaries."""
    contents = "\n".join(
        f"- [{m.kind}, {m.created_at.strftime('%Y-%m-%d')}, importance={m.importance}] {m.content}"
        for m in cluster
    )
    result = await llm.chat_json(
        system=CONSOLIDATION_PROMPT.format(count=len(cluster), memories=contents),
        user="Consolidate these memories.",
        schema=ConsolidationResult,
        fast=True,
    )

    original_ids = [m.id for m in cluster]
    batch_id = uuid_mod.uuid4().hex[:8]
    max_importance = max(m.importance for m in cluster)

    creates = []
    for cm in result.memories:
        creates.append(
            MemoryCreate(
                user_id=user_id,
                content=cm.content[:1000],
                kind=cm.kind if cm.kind in ("fact", "preference", "event", "episode") else "fact",
                importance=min(max(cm.importance, max_importance), 5),
                source_key=f"consolidation/{batch_id}",
                replaces=original_ids,
            )
        )
    return creates


async def consolidate_memories(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    dry_run: bool = False,
    min_cluster_size: int | None = None,
    similarity_threshold: float | None = None,
) -> dict:
    """Merge clusters of related active memories into consolidated summaries.

    Returns stats: {clusters_found, memories_merged, consolidated_created, dry_run}.
    Schedule this (e.g. weekly) — paraphrase drift accumulates otherwise.
    """
    min_size = min_cluster_size or config.consolidation_min_cluster_size
    threshold = similarity_threshold or config.consolidation_similarity_threshold

    groups = await _find_memory_groups(db, config, user_id)

    # Exclude memories that back Instance rows — consolidation destroys the
    # per-instance granularity that counting depends on.
    instance_memory_ids: set[str] = set()
    try:
        async with db.session_scope() as session:
            ids = (await session.scalars(
                select(Instance.source_memory_id).where(Instance.user_id == user_id)
            )).all()
            instance_memory_ids = set(ids)
    except Exception:
        pass
    if instance_memory_ids:
        groups = [[m for m in g if m.id not in instance_memory_ids] for g in groups]
        groups = [g for g in groups if len(g) >= 2]

    log.info("consolidation: %d candidate groups", len(groups))

    processed_ids: set[str] = set()
    clusters_found = memories_merged = consolidated_created = 0
    preview: list[str] = []

    for memories in groups:
        unprocessed = [m for m in memories if m.id not in processed_ids]
        if len(unprocessed) < min_size:
            continue
        clusters = await _cluster_by_similarity(unprocessed, threshold)
        for cluster in clusters:
            cluster = [m for m in cluster if m.id not in processed_ids]
            if len(cluster) < min_size:
                continue
            clusters_found += 1
            if dry_run:
                preview.append(
                    f"Cluster ({len(cluster)} memories): "
                    + " | ".join(m.content[:60] for m in cluster[:3])
                )
                processed_ids.update(m.id for m in cluster)
                memories_merged += len(cluster)
                continue
            try:
                creates = await _consolidate_cluster(db, llm, cluster, user_id)
                original_ids = [m.id for m in cluster]
                for data in creates:
                    new_memory = await save_memory(db, llm, config, data)
                    if config.enable_entities:
                        async with db.session_scope() as session:
                            await copy_entity_links(session, original_ids, new_memory.id)
                    consolidated_created += 1
                processed_ids.update(m.id for m in cluster)
                memories_merged += len(cluster)
            except Exception:
                log.exception("consolidation failed for a cluster of %d memories", len(cluster))

    return {
        "clusters_found": clusters_found,
        "memories_merged": memories_merged,
        "consolidated_created": consolidated_created,
        "dry_run": dry_run,
        "preview": preview[:20] if dry_run else [],
    }
