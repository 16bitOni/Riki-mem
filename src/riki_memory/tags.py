"""Tag embedding cache and normalization utilities.

Write-time convergence: proposed tags are matched against the user's existing
tag vocabulary by embedding similarity.  "fundraiser" becomes "charity-event"
if "charity-event" already exists and is close enough.

Query-time expansion: query tags are broadened to include semantically similar
tags from the vocabulary so "charity" also finds memories tagged "fundraiser".

Both paths share a process-level embedding cache, keyed by (embedding model, term),
so each term is embedded at most once per model per process, and every term not yet
cached is embedded in one batched call.
"""

import logging

from .llm import embed_texts
from .similarity import cosine_similarity

log = logging.getLogger(__name__)

# shortcut: unbounded per-process cache (a few KB per term); move to a DB table with
# pgvector if vocabularies reach tens of thousands of terms per process.
_tag_embedding_cache: dict[tuple[str, str], list[float]] = {}


async def tag_vectors(llm, terms) -> dict[str, list[float]]:
    """Vectors for ``terms`` (missing ones embedded in one call); failed terms are left out."""
    model = getattr(llm, "embedding_model_id", "")
    terms = list(dict.fromkeys(terms))
    missing = [t for t in terms if (model, t) not in _tag_embedding_cache]
    if missing:
        try:
            for term, vec in zip(missing, await embed_texts(llm, missing)):
                _tag_embedding_cache[(model, term)] = vec
        except Exception:
            log.warning("could not embed %d tag(s); matching them by exact name only", len(missing))
    return {t: _tag_embedding_cache[(model, t)] for t in terms if (model, t) in _tag_embedding_cache}


async def get_tag_embedding(llm, tag: str) -> list[float] | None:
    return (await tag_vectors(llm, [tag])).get(tag)


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
    vectors = await tag_vectors(llm, [*query_tags, *existing])
    for qtag in query_tags:
        if qtag not in vectors:
            continue
        for etag in existing:
            if etag not in expanded and etag in vectors and \
                    cosine_similarity(vectors[qtag], vectors[etag]) >= threshold:
                expanded.add(etag)
    return list(expanded)


async def get_user_subjects(db, user_id: str) -> set[str]:
    """Every instance subject this user has, e.g. {"instrument", "wedding"}."""
    from sqlalchemy import select
    from .models import Instance

    async with db.session_scope() as session:
        rows = await session.scalars(
            select(Instance.subject).where(Instance.user_id == user_id).distinct()
        )
        return set(rows.all())


async def normalize_subjects(
    llm, db, user_id: str, subjects: list[str], threshold: float
) -> list[str]:
    """Converge instance subjects onto the user's existing ones.

    Subjects arrive already singular ("instruments" -> "instrument"), so this
    only has to catch different words for the same class ("musical-instrument"
    vs "instrument"), by embedding similarity — the same rule tags use.
    """
    if not subjects:
        return subjects
    existing = await get_user_subjects(db, user_id)
    if not existing:
        return subjects
    return [
        subject if subject in existing else await _closest(llm, subject, existing, threshold) or subject
        for subject in subjects
    ]


async def _closest(llm, term: str, vocabulary: set[str], threshold: float) -> str | None:
    vectors = await tag_vectors(llm, [term, *vocabulary])
    if term not in vectors:
        return None
    best, best_sim = None, threshold
    for candidate in vocabulary:
        if candidate in vectors:
            sim = cosine_similarity(vectors[term], vectors[candidate])
            if sim > best_sim:
                best, best_sim = candidate, sim
    return best


async def normalize_tags(
    llm, db, user_id: str, tags: list[str], threshold: float
) -> list[str]:
    """Normalize tags against existing vocabulary, converging to canonical forms."""
    if not tags:
        return tags

    existing = await get_user_tag_vocabulary(db, user_id)
    if not existing:
        return tags

    if all(tag in existing for tag in tags):
        return tags
    return [tag if tag in existing else await _closest(llm, tag, existing, threshold) or tag for tag in tags]
