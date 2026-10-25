"""Category facts backfill for exhaustive aggregation retrieval.

The write-time path (store._upsert_category_facts) already populates category_facts
for every new memory.  This module provides a one-shot backfill that does the same
for memories that were ingested before the category_facts table existed.

Read-time aggregation is handled by recall._category_facts_memories, which JOINs
category_facts to memories and appends the matching source memories to the found set.
"""

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .db import Database
from .models import CategoryFact, Memory
from .store import _TAG_TO_CATEGORY, _extract_amount, _item_name_from_content

log = logging.getLogger(__name__)


async def backfill_user_facts(db: Database, user_id: str) -> int:
    """Populate category_facts for all existing memories that haven't been processed.

    Uses the same rule-based extraction as write-time (_upsert_category_facts) so
    results are consistent.  Idempotent — skips memories whose source_memory_id is
    already present in the table.

    Returns the number of new CategoryFact rows inserted.
    """
    async with db.session_scope() as session:
        existing_source_ids: set[str] = set(
            (await session.scalars(
                select(CategoryFact.source_memory_id).where(
                    CategoryFact.user_id == user_id,
                )
            )).all()
        )
        memories = (await session.scalars(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.is_active.is_(True),
                Memory.tags.is_not(None),
            )
        )).all()

    to_process = [
        m for m in memories
        if m.id not in existing_source_ids
        and any(t.lower().strip() in _TAG_TO_CATEGORY for t in (m.tags or []))
    ]

    saved = 0
    for memory in to_process:
        tags = memory.tags or []
        numeric_value, unit = _extract_amount(memory.content)
        item_name = _item_name_from_content(memory.content)
        seen_categories: set[str] = set()

        for tag in tags:
            category = _TAG_TO_CATEGORY.get(tag.lower().strip())
            if not category or category in seen_categories:
                continue
            seen_categories.add(category)
            try:
                async with db.session_scope() as session:
                    session.add(CategoryFact(
                        user_id=user_id,
                        category=category,
                        item_name=item_name,
                        numeric_value=numeric_value,
                        unit=unit,
                        date=memory.event_at,
                        source_memory_id=memory.id,
                    ))
                saved += 1
            except IntegrityError:
                pass  # already exists

    log.info("backfill: processed %d memories, inserted %d category_facts for %s",
             len(to_process), saved, user_id)
    return saved
