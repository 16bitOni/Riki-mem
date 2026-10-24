"""The light knowledge graph: people, places, orgs and projects mentioned in
memories, and the relations between them ("Riya --sister of--> Subhadip").

This is optional. Disable it with ``MemoryConfig(enable_entities=False)`` and
the write path skips entity linking entirely, and retrieval skips the graph
channel — the teardown found the graph write-heavy for its payoff at small
scale, so it should be a choice, not a tax.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Entity, EntityRelation, MemoryEntity
from .schemas import EntityMention, RelationMention

log = logging.getLogger(__name__)


async def get_or_create_entity(
    session: AsyncSession, user_id: str, name: str, type_: str = "other"
) -> Entity:
    found = await session.scalar(
        select(Entity).where(Entity.user_id == user_id, func.lower(Entity.name) == name.lower())
    )
    if found:
        return found
    entity = Entity(user_id=user_id, name=name, type=type_)
    session.add(entity)
    await session.flush()
    return entity


async def link_entities(
    session: AsyncSession,
    user_id: str,
    memory_id: str,
    entities: list[EntityMention],
    relations: list[RelationMention],
) -> None:
    entity_ids: set[str] = set()
    for mention in entities:
        entity = await get_or_create_entity(session, user_id, mention.name, mention.type)
        entity_ids.add(entity.id)

    for rel in relations:
        subject = await get_or_create_entity(session, user_id, rel.subject)
        target = await get_or_create_entity(session, user_id, rel.target)
        entity_ids.update([subject.id, target.id])
        exists = await session.scalar(
            select(EntityRelation.id).where(
                EntityRelation.subject_id == subject.id,
                EntityRelation.predicate == rel.predicate,
                EntityRelation.target_id == target.id,
            )
        )
        if not exists:
            session.add(
                EntityRelation(
                    user_id=user_id, subject_id=subject.id, predicate=rel.predicate, target_id=target.id
                )
            )

    session.add_all(MemoryEntity(memory_id=memory_id, entity_id=eid) for eid in entity_ids)


async def copy_entity_links(
    session: AsyncSession, source_memory_ids: list[str], target_memory_id: str
) -> int:
    """Copy all entity links from source memories to the target (union). Returns new links created."""
    existing = set(
        (
            await session.scalars(
                select(MemoryEntity.entity_id).where(MemoryEntity.memory_id.in_(source_memory_ids))
            )
        ).all()
    )
    already = set(
        (
            await session.scalars(
                select(MemoryEntity.entity_id).where(MemoryEntity.memory_id == target_memory_id)
            )
        ).all()
    )
    new_links = existing - already
    if new_links:
        session.add_all(MemoryEntity(memory_id=target_memory_id, entity_id=eid) for eid in new_links)
    return len(new_links)


async def prune_orphaned_entities(session: AsyncSession, user_id: str) -> int:
    """Delete entities no memory links to any more.

    The original never cleaned these up, so ``forget_memory`` left orphans
    forever. Call it after forgets, or periodically from maintenance.
    """
    orphan_ids = (
        await session.scalars(
            select(Entity.id)
            .where(Entity.user_id == user_id)
            .where(
                ~select(MemoryEntity.entity_id)
                .where(MemoryEntity.entity_id == Entity.id)
                .exists()
            )
        )
    ).all()
    count = 0
    for entity_id in orphan_ids:
        entity = await session.get(Entity, entity_id)
        if entity is not None:
            await session.delete(entity)
            count += 1
    if count:
        log.debug("pruned %d orphaned entities", count)
    return count
