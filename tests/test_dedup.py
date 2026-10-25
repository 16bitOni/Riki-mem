"""Tests for identity-based deduplication via Instance model."""

import pytest
from datetime import datetime, timezone

from riki_memory import InstanceItem, MemoryCreate
from riki_memory.models import Instance, Memory
from riki_memory.store import save_memory

from sqlalchemy import select


async def test_instance_dedup_supersedes_old_memory(memory, provider):
    """When a new memory matches an existing instance_key, the old memory is superseded."""
    inst = InstanceItem(
        category="wedding-attended",
        instance_key="rachel-mike-wedding",
        display_name="Rachel and Mike wedding",
        status="occurred",
        date=datetime(2023, 8, 15, tzinfo=timezone.utc),
    )

    data1 = MemoryCreate(
        user_id="u1",
        content="Went to my cousin's wedding.",
        kind="event",
        importance=3,
        event_at=datetime(2023, 8, 15, tzinfo=timezone.utc),
        instances=[inst],
    )
    data2 = MemoryCreate(
        user_id="u1",
        content="Attended Rachel and Mike's wedding on August 15, 2023 in Boston.",
        kind="event",
        importance=4,
        event_at=datetime(2023, 8, 15, tzinfo=timezone.utc),
        instances=[inst],
    )

    mem1 = await save_memory(memory.db, provider, memory.config, data1)
    mem2 = await save_memory(memory.db, provider, memory.config, data2)

    # The second save should supersede the first
    assert mem2.id != mem1.id
    async with memory.db.session_scope() as session:
        old = await session.scalar(select(Memory).where(Memory.id == mem1.id))
        assert old.is_active is False


async def test_distinct_events_same_day_not_collapsed(memory, provider):
    """Two different events on the same day with different instance_keys are both kept."""
    data1 = MemoryCreate(
        user_id="u1",
        content="Bought a B-29 model kit at the hobby store.",
        kind="event",
        importance=3,
        event_at=datetime(2023, 6, 10, tzinfo=timezone.utc),
        instances=[InstanceItem(
            category="model-kit",
            instance_key="b-29-model",
            display_name="B-29 model kit",
            status="occurred",
        )],
    )
    data2 = MemoryCreate(
        user_id="u1",
        content="Also picked up a Camaro model on the same trip.",
        kind="event",
        importance=3,
        event_at=datetime(2023, 6, 10, tzinfo=timezone.utc),
        instances=[InstanceItem(
            category="model-kit",
            instance_key="camaro-model",
            display_name="Camaro model kit",
            status="occurred",
        )],
    )

    mem1 = await save_memory(memory.db, provider, memory.config, data1)
    mem2 = await save_memory(memory.db, provider, memory.config, data2)

    # Both memories should be active
    async with memory.db.session_scope() as session:
        active = (await session.scalars(
            select(Memory).where(
                Memory.user_id == "u1",
                Memory.is_active.is_(True),
            )
        )).all()

    assert len(active) == 2

    # And both instance rows exist with distinct keys
    async with memory.db.session_scope() as session:
        instances = (await session.scalars(
            select(Instance).where(Instance.user_id == "u1", Instance.category == "model-kit")
        )).all()

    keys = {i.instance_key for i in instances}
    assert "b-29-model" in keys
    assert "camaro-model" in keys


async def test_non_instance_memories_use_content_dedup(memory, provider):
    """Memories without instances still use the old content-hash dedup path."""
    data1 = MemoryCreate(
        user_id="u1",
        content="My sister's name is Riya.",
        kind="fact",
        importance=3,
    )
    data2 = MemoryCreate(
        user_id="u1",
        content="My sister's name is Riya.",
        kind="fact",
        importance=3,
    )

    mem1 = await save_memory(memory.db, provider, memory.config, data1)
    mem2 = await save_memory(memory.db, provider, memory.config, data2)

    # Exact duplicate returns same row
    assert mem1.id == mem2.id
