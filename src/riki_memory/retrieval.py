"""Hybrid retrieval.

Five independent "channels" each return a ranked list of memory ids:

    keyword  - exact words (Postgres full-text search, ts_rank_cd ranked)
    semantic - similar meaning (pgvector HNSW, cosine)
    time     - inside a date window
    graph    - linked to a named person/place/project (and their direct neighbours)
    episode  - memories from sessions whose summary matches the query

The lists are merged with weighted Reciprocal Rank Fusion, so a memory found by
several channels wins. Important memories get a small boost, and an optional
LLM reranker can reorder the final shortlist.

Per-channel failure isolation is kept from the original: a broken channel
(say, the embedding API is down) returns no hits instead of breaking recall.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import TypeVar

from sqlalchemy import String, func, literal, or_, select, union_all

from .config import MemoryConfig
from .db import Database
from .keyword import CHANNEL_DEPTH, live_clauses
from .keyword import keyword_search as _keyword_search
from .llm import LLMProvider
from .models import Entity, EntityRelation, Episode, EpisodeMemory, Memory, MemoryEntity
from .schemas import RetrievalPlan

log = logging.getLogger(__name__)

# Re-exported for backwards compatibility; the canonical home is keyword.py.
MAX_TERMS = 10  # keywords or entity names taken from one plan

T = TypeVar("T")


def fuse_scores(
    rankings: list[list[T]], weights: list[float] | None = None, k: int = 60
) -> dict[T, float]:
    """Reciprocal Rank Fusion scores. A channel's `weight` scales what it contributes."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[T, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + position + 1)
    return scores


def reciprocal_rank_fusion(
    rankings: list[list[T]], weights: list[float] | None = None, k: int = 60
) -> list[T]:
    """Merge several ranked id lists into one. Higher in any list = higher overall."""
    scores = fuse_scores(rankings, weights, k)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def importance_boost(importance: int, weight: float) -> float:
    """Multiplier around 1.0: importance 3 is neutral, 5 gets +2*weight, 1 gets -2*weight."""
    return 1.0 + weight * (importance - 3)


async def keyword_search(
    db: Database, user_id: str, keywords: list[str], start: datetime | None, end: datetime | None
) -> list[str]:
    """Keyword channel: Postgres full-text search, ranked with ts_rank_cd."""
    return await _keyword_search(db, user_id, keywords, start, end)


async def semantic_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    text_query: str,
    start: datetime | None,
    end: datetime | None,
    vector: list[float] | None = None,
) -> list[str]:
    if vector is None:
        vector = await llm.embed(text_query)
    async with db.session_scope() as session:
        # Only vectors from the current model are compared (see vectors.py).
        scored = await db.vectors.nearest(
            session, Memory, live_clauses(user_id, start, end), vector,
            llm.embedding_model_id, CHANNEL_DEPTH, config.min_similarity,
        )
    return [mid for mid, _ in scored]


async def time_window(
    db: Database, user_id: str, start: datetime | None, end: datetime | None
) -> list[str]:
    async with db.session_scope() as session:
        stmt = (
            select(Memory.id)
            .where(*live_clauses(user_id, start, end))
            .order_by(
                Memory.importance.desc(),
                func.coalesce(Memory.event_at, Memory.created_at).desc(),
            )
            .limit(CHANNEL_DEPTH)
        )
        return list((await session.scalars(stmt)).all())


async def graph_search(
    db: Database, config: MemoryConfig, user_id: str, names: list[str], relation_words=()
) -> list[str]:
    """Memories about people/places/projects connected to the question, up to
    `graph_max_hops` away.

    Start points ("seeds") are the named entities plus both ends of any relation
    whose label contains one of the relationship words ("sister" finds
    "sister of" links). From the seeds we walk along relations in both
    directions. Memories about closer entities rank first.
    """
    wanted = [n.strip().lower() for n in list(names)[:MAX_TERMS] if n.strip()]
    words = [w.strip().lower() for w in list(relation_words)[:MAX_TERMS] if w.strip()]
    if not wanted and not words:
        return []

    async with db.session_scope() as session:
        seeds: set[str] = set()
        if wanted:
            seeds.update(
                await session.scalars(
                    select(Entity.id).where(Entity.user_id == user_id, func.lower(Entity.name).in_(wanted))
                )
            )
        if words:
            ends = await session.execute(
                select(EntityRelation.subject_id, EntityRelation.target_id).where(
                    EntityRelation.user_id == user_id,
                    or_(
                        *(
                            func.lower(EntityRelation.predicate).like(f"%{w}%", escape="\\")
                            for w in words
                        )
                    ),
                )
            )
            for subject_id, target_id in ends:
                seeds.update([subject_id, target_id])
        if not seeds:
            return []

        # Relations can be followed in either direction ("Riya sister of Subhadip" works both ways).
        edges = union_all(
            select(
                EntityRelation.subject_id.label("src"), EntityRelation.target_id.label("dst")
            ).where(EntityRelation.user_id == user_id),
            select(EntityRelation.target_id, EntityRelation.subject_id).where(
                EntityRelation.user_id == user_id
            ),
        ).cte("edges")

        # Recursive walk: depth 0 = the seeds, depth 1 = their neighbours, and so on.
        walk = (
            select(Entity.id.label("entity_id"), literal(0).label("depth"))
            .where(Entity.id.in_(seeds))
            .cte("walk", recursive=True)
        )
        walk = walk.union(
            select(edges.c.dst, walk.c.depth + 1)
            .select_from(walk)
            .join(edges, edges.c.src == walk.c.entity_id)
            .where(walk.c.depth < config.graph_max_hops)
        )
        closest = (
            select(walk.c.entity_id, func.min(walk.c.depth).label("depth"))
            .group_by(walk.c.entity_id)
            .order_by("depth")
            .limit(config.graph_fanout_limit)  # a very connected graph cannot explode the query
            .subquery()
        )

        hops = func.min(closest.c.depth).label("hops")
        rows = await session.execute(
            select(Memory.id)
            .join(MemoryEntity, MemoryEntity.memory_id == Memory.id)
            .join(closest, closest.c.entity_id == MemoryEntity.entity_id)
            .where(*live_clauses(user_id, None, None))
            .group_by(Memory.id, Memory.importance, Memory.created_at)
            .order_by(hops, Memory.importance.desc(), Memory.created_at.desc())
            .limit(CHANNEL_DEPTH)
        )
        return [row[0] for row in rows]


async def tag_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    tags: list[str],
) -> list[str]:
    """Tag channel: return ALL active memories matching ANY of the given tags.

    Unlike other channels this is exhaustive — no CHANNEL_DEPTH cap — because
    aggregation queries ("how many weddings?") need every instance.

    Uses semantic expansion: query tags are broadened to include similar tags
    from the user's vocabulary so "charity" also matches "fundraiser".
    """
    if not tags:
        return []
    normalised = [t.strip().lower().replace(" ", "-") for t in tags if t.strip()]
    if not normalised:
        return []

    from .tags import expand_tags_semantically

    try:
        search_tags = await expand_tags_semantically(
            llm, db, user_id, normalised, config.tag_similarity_threshold
        )
    except Exception:
        log.exception("semantic tag expansion failed; falling back to exact tags")
        search_tags = normalised

    async with db.session_scope() as session:
        conditions = [Memory.tags.cast(String).like(f'%"{tag}"%') for tag in search_tags]
        stmt = (
            select(Memory.id)
            .where(
                Memory.user_id == user_id,
                Memory.is_active.is_(True),
                Memory.tags.is_not(None),
                or_(*conditions),
            )
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .limit(200)
        )
        return list((await session.scalars(stmt)).all())


async def episode_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    text_query: str,
    vector: list[float] | None = None,
) -> list[str]:
    """Episode channel: sessions whose summary matches the query."""
    if vector is None:
        vector = await llm.embed(text_query)
    async with db.session_scope() as session:
        matching = await db.vectors.nearest(
            session, Episode, [Episode.user_id == user_id], vector,
            llm.embedding_model_id, CHANNEL_DEPTH, config.episode_similarity_threshold,
        )
    if not matching:
        return []

    async with db.session_scope() as session:
        memory_ids = (
            await session.scalars(
                select(EpisodeMemory.memory_id)
                .join(Memory, Memory.id == EpisodeMemory.memory_id)
                .where(
                    EpisodeMemory.episode_id.in_([eid for eid, _ in matching]),
                    Memory.is_active.is_(True),
                    Memory.user_id == user_id,
                )
                .limit(CHANNEL_DEPTH)
            )
        ).all()

    return list(memory_ids)


async def _timed(name: str, coroutine):
    """Run one channel. A broken channel must not break the whole recall:
    it returns no hits, and the failure is logged."""
    t0 = time.perf_counter()
    try:
        ids, error = await coroutine, None
    except Exception as exc:
        log.exception("memory channel '%s' failed", name)
        ids, error = [], type(exc).__name__
    return name, ids, round((time.perf_counter() - t0) * 1000), error


async def hybrid_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    plan: RetrievalPlan,
    limit: int | None = None,
    rerank_query: str | None = None,
) -> tuple[list[Memory], dict]:
    """Run every channel the plan asks for and return ``(memories, report)``.

    1. channels run in parallel; a failing one is skipped
    2. weighted RRF merges them
    3. an importance prior nudges the order (multiplicative, so it can only break near-ties)
    4. optionally an LLM reranker reorders the shortlist (``config.rerank = "llm"``)

    The report carries per-channel hits/timings and every returned memory with
    its score and which channels found it — the same data the original
    dashboard's Retrieval tab read from trace spans.
    """
    limit = limit or config.recall_limit

    # Embed once, share the vector between semantic and episode channels.
    query_vector = None
    needs_vector = bool(plan.semantic_query)
    if needs_vector:
        try:
            query_vector = await llm.embed(plan.semantic_query)
        except Exception:
            log.exception("query embedding failed; semantic and episode channels will be skipped")

    channels = {}
    if plan.keywords:
        channels["keyword"] = keyword_search(db, user_id, plan.keywords, None, None)
    if plan.semantic_query and query_vector is not None:
        channels["semantic"] = semantic_search(
            db, llm, config, user_id, plan.semantic_query, None, None,
            vector=query_vector,
        )
    if plan.start or plan.end:
        channels["time"] = time_window(db, user_id, plan.start, plan.end)
    if config.enable_entities and (plan.entities or plan.relation_words):
        channels["graph"] = graph_search(db, config, user_id, plan.entities, plan.relation_words)
    if config.enable_episodes and plan.semantic_query and query_vector is not None:
        channels["episode"] = episode_search(
            db, llm, config, user_id, plan.semantic_query,
            vector=query_vector,
        )
    if plan.tags:
        channels["tag"] = tag_search(db, llm, config, user_id, plan.tags)

    weights = {
        "keyword": config.weight_keyword,
        "semantic": config.weight_semantic,
        "graph": config.weight_graph,
        "time": config.weight_time,
        "episode": config.weight_episode,
        "tag": config.weight_tag,
    }
    outcomes = await asyncio.gather(*(_timed(name, task) for name, task in channels.items()))
    results = [ids for _, ids, _, _ in outcomes]
    report: dict = {
        "limit": limit,
        "channels": {
            name: {"hits": len(ids), "ms": ms, "error": error, "weight": weights[name]}
            for name, ids, ms, error in outcomes
        },
        "plan": {
            "keywords": plan.keywords,
            "semantic_query": plan.semantic_query,
            "entities": plan.entities,
            "relation_words": plan.relation_words,
            "tags": plan.tags,
            "start": plan.start.isoformat() if plan.start else None,
            "end": plan.end.isoformat() if plan.end else None,
        },
        "candidates": 0,
        "reranked": False,
        "returned": 0,
        "results": [],
    }
    scores = fuse_scores(list(results), [weights[name] for name in channels])
    report["candidates"] = len(scores)
    if not scores:
        return [], report

    use_reranker = bool(config.rerank == "llm" and rerank_query)
    shortlist = max(limit, config.rerank_candidates) if use_reranker else limit * 2
    candidate_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:shortlist]

    async with db.session_scope() as session:
        rows = (await session.scalars(select(Memory).where(Memory.id.in_(candidate_ids)))).all()
    ranked = sorted(
        rows,
        key=lambda m: scores[m.id] * importance_boost(m.importance, config.importance_weight),
        reverse=True,
    )
    before_rerank = [m.id for m in ranked]

    if use_reranker:
        from .rerank import rerank  # local import: the LLM stack is only needed when enabled

        ranked = await rerank(llm, rerank_query or "", ranked)
        report["reranked"] = True
    final = ranked[:limit]

    positions = {name: {mid: i + 1 for i, mid in enumerate(ids)} for name, ids, _, _ in outcomes}
    report["returned"] = len(final)
    report["results"] = [
        {
            "id": str(m.id),
            "rank": rank,
            "kind": m.kind,
            "importance": m.importance,
            "rrf": round(scores[m.id], 5),
            "score": round(
                scores[m.id] * importance_boost(m.importance, config.importance_weight), 5
            ),
            "found_by": {name: pos[m.id] for name, pos in positions.items() if m.id in pos},
            "moved_by_rerank": before_rerank.index(m.id) + 1 - rank if use_reranker else 0,
            "text": m.content[:160],
        }
        for rank, m in enumerate(final, start=1)
    ]
    return final, report
