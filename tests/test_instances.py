"""Tests for the Instance model, upsert, dedup, and schema validation."""

import pytest
from datetime import datetime, timezone

from riki_memory import InstanceItem, MemoryCreate
from riki_memory.models import Instance
from riki_memory.store import save_memory

from sqlalchemy import select


@pytest.fixture
def make_memory_with_instances():
    """Factory for MemoryCreate objects with instances attached."""
    def _make(
        content: str,
        user_id: str = "u1",
        instances: list | None = None,
        event_at: datetime | None = None,
    ):
        return MemoryCreate(
            user_id=user_id,
            content=content,
            kind="event",
            importance=3,
            event_at=event_at,
            instances=instances or [],
            tags=["wedding"],
        )
    return _make


async def test_save_memory_creates_instance_rows(memory, provider, make_memory_with_instances):
    """Saving a memory with instances attached creates Instance rows in the DB."""
    data = make_memory_with_instances(
        content="Attended Rachel and Mike's wedding on August 15, 2023.",
        instances=[
            InstanceItem(
                kind="event", subject="wedding",
                instance_key="rachel-mike-wedding",
                display_name="Rachel and Mike wedding",
                status="occurred",
                date=datetime(2023, 8, 15, tzinfo=timezone.utc),
            )
        ],
        event_at=datetime(2023, 8, 15, tzinfo=timezone.utc),
    )

    saved = await save_memory(memory.db, provider, memory.config, data)
    assert saved is not None

    async with memory.db.session_scope() as session:
        rows = (await session.scalars(
            select(Instance).where(Instance.user_id == "u1")
        )).all()

    assert len(rows) == 1
    assert (rows[0].kind, rows[0].subject) == ("event", "wedding")
    assert rows[0].instance_key == "rachel-mike-wedding"
    assert rows[0].status == "occurred"
    assert rows[0].source_memory_id == saved.id


async def test_instance_dedup_same_key(memory, provider, make_memory_with_instances):
    """Two memories about the same instance_key don't create duplicate Instance rows."""
    inst = InstanceItem(
        kind="item", subject="instrument",
        instance_key="fender-stratocaster",
        display_name="Fender Stratocaster",
        status="current",
    )

    data1 = make_memory_with_instances(
        content="I bought a Fender Stratocaster guitar last month.",
        instances=[inst],
    )
    data2 = make_memory_with_instances(
        content="My Fender Stratocaster is my favorite instrument to play.",
        instances=[inst],
    )

    await save_memory(memory.db, provider, memory.config, data1)
    await save_memory(memory.db, provider, memory.config, data2)

    async with memory.db.session_scope() as session:
        rows = (await session.scalars(
            select(Instance).where(
                Instance.user_id == "u1",
                Instance.subject == "instrument",
            )
        )).all()

    # Two rows (one per source_memory_id), but same instance_key
    keys = {r.instance_key for r in rows}
    assert keys == {"fender-stratocaster"}


async def test_instance_status_filtering(memory, provider, make_memory_with_instances):
    """Instances with different statuses are stored correctly."""
    data1 = make_memory_with_instances(
        content="I own a ukulele.",
        instances=[InstanceItem(
            kind="item", subject="instrument",
            instance_key="ukulele",
            display_name="Ukulele",
            status="current",
        )],
    )
    data2 = make_memory_with_instances(
        content="I sold my old drum set.",
        instances=[InstanceItem(
            kind="item", subject="instrument",
            instance_key="drum-set",
            display_name="Drum set",
            status="former",
        )],
    )
    data3 = make_memory_with_instances(
        content="I want to buy a piano someday.",
        instances=[InstanceItem(
            kind="item", subject="instrument",
            instance_key="piano",
            display_name="Piano",
            status="desired",
        )],
    )

    await save_memory(memory.db, provider, memory.config, data1)
    await save_memory(memory.db, provider, memory.config, data2)
    await save_memory(memory.db, provider, memory.config, data3)

    async with memory.db.session_scope() as session:
        current = (await session.scalars(
            select(Instance).where(
                Instance.user_id == "u1",
                Instance.subject == "instrument",
                Instance.status.in_(["current", "occurred"]),
            )
        )).all()
        all_inst = (await session.scalars(
            select(Instance).where(
                Instance.user_id == "u1",
                Instance.subject == "instrument",
            )
        )).all()

    assert len(current) == 1
    assert current[0].instance_key == "ukulele"
    assert len(all_inst) == 3


async def test_instance_item_slugify():
    """InstanceItem slugifies the key and makes the subject a singular slug."""
    item = InstanceItem(
        kind="Attendance",
        subject="Family Weddings",
        instance_key="Rachel & Mike's Wedding!!",
        display_name="Rachel and Mike wedding",
        status="occurred",
    )
    assert item.kind == "event"
    assert item.subject == "family-wedding"
    assert item.instance_key == "rachel-mike-s-wedding"


async def test_instance_item_unknown_kind_becomes_other():
    item = InstanceItem(kind="hobbyhorse", subject="x", instance_key="k", display_name="d",
                        status="occurred")
    assert item.kind == "other"


async def test_instance_item_status_validation():
    """InstanceItem rejects invalid status values."""
    with pytest.raises(Exception):
        InstanceItem(
            kind="event", subject="wedding",
            instance_key="test-wedding",
            display_name="Test",
            status="invalid_status",
        )
