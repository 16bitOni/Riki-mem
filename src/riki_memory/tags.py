"""Tag embedding cache and normalization utilities.

Write-time convergence: proposed tags are matched against the user's existing
tag vocabulary by embedding similarity.  "fundraiser" becomes "charity-event"
if "charity-event" already exists and is close enough.

Query-time expansion: query tags are broadened to include semantically similar
tags from the vocabulary so "charity" also finds memories tagged "fundraiser".

Both paths share a process-level embedding cache so each tag string is embedded
at most once per process lifetime.
"""

import logging

from .similarity import cosine_similarity

log = logging.getLogger(__name__)

_tag_embedding_cache: dict[str, list[float]] = {}


async def get_tag_embedding(llm, tag: str) -> list[float] | None:
    if tag in _tag_embedding_cache:
        return _tag_embedding_cache[tag]
    try:
        vec = await llm.embed(tag)
        _tag_embedding_cache[tag] = vec
        return vec
    except Exception:
        log.debug("failed to embed tag %r", tag)
        return None


async def get_user_tag_vocabulary(db, user_id: str) -> set[str]:
    from sqlalchemy import select
    from .models import Memory

    async with db.session_scope() as session:
        rows = (
            await session.execute(
                select(Memory.tags).where(
                    Memory.user_id == user_id,
                    Memory.is_active.is_(True),
                    Memory.tags.is_not(None),
                )
            )
        ).scalars().all()

    tags: set[str] = set()
    for tag_list in rows:
        if isinstance(tag_list, list):
            tags.update(t for t in tag_list if isinstance(t, str))
    return tags


async def expand_tags_semantically(
    llm, db, user_id: str, query_tags: list[str], threshold: float
) -> list[str]:
    """Expand query tags with semantically similar tags from the user's vocabulary."""
    expanded = set(query_tags)
    existing = await get_user_tag_vocabulary(db, user_id)

    for qtag in query_tags:
        qtag_vec = await get_tag_embedding(llm, qtag)
        if qtag_vec is None:
            continue
        for etag in existing:
            if etag in expanded:
                continue
            etag_vec = await get_tag_embedding(llm, etag)
            if etag_vec is None:
                continue
            if cosine_similarity(qtag_vec, etag_vec) >= threshold:
                expanded.add(etag)

    return list(expanded)


async def normalize_tags(
    llm, db, user_id: str, tags: list[str], threshold: float
) -> list[str]:
    """Normalize tags against existing vocabulary, converging to canonical forms."""
    if not tags:
        return tags

    existing = await get_user_tag_vocabulary(db, user_id)
    if not existing:
        return tags

    normalized = []
    for tag in tags:
        if tag in existing:
            normalized.append(tag)
            continue

        tag_vec = await get_tag_embedding(llm, tag)
        if tag_vec is None:
            normalized.append(tag)
            continue

        best_match = None
        best_sim = threshold
        for etag in existing:
            etag_vec = await get_tag_embedding(llm, etag)
            if etag_vec is None:
                continue
            sim = cosine_similarity(tag_vec, etag_vec)
            if sim > best_sim:
                best_sim = sim
                best_match = etag

        normalized.append(best_match if best_match else tag)

    return normalized
