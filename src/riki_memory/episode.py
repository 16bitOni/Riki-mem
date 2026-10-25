"""Episodic memory: session-level summaries that link atomic memories.

After all turns of a conversation session have been ingested (via
``remember()``), call ``close_session()`` to generate a summary that
captures the session's topics and themes. During recall, the episode
channel finds matching sessions and returns their linked atomic memories,
giving retrieval breadth across an entire conversation.
"""

import logging

from sqlalchemy import delete, select

from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .models import Episode, EpisodeMemory, Memory

log = logging.getLogger(__name__)

EPISODE_SUMMARY_PROMPT = """You summarise a conversation session for a personal assistant's long-term memory.
You will receive a list of atomic memories extracted from one session.

Write a 2-4 sentence summary capturing:
- The main topics discussed
- Key decisions, events, or facts mentioned
- The overall context (what the user was doing / asking about)

Be specific — include names, dates, and details. The summary will be used to find this session
later when a related question comes up."""


async def close_session(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    session_key: str,
    date=None,
) -> Episode | None:
    """Generate an episode summary for a completed session.

    Finds memories whose ``source_key`` starts with ``session_key``,
    summarises them, embeds the summary, and links the episode to its
    atomic memories. Idempotent: calling again for the same session_key
    updates the existing episode.
    """
    async with db.session_scope() as session:
        memories = (
            await session.scalars(
                select(Memory).where(
                    Memory.user_id == user_id,
                    Memory.is_active.is_(True),
                    Memory.source_key.like(f"{session_key}%"),
                ).order_by(Memory.created_at)
            )
        ).all()

    if not memories:
        return None

    listing = "\n".join(f"- [{m.kind}] {m.content}" for m in memories)
    try:
        summary = await llm.chat_text(
            system=EPISODE_SUMMARY_PROMPT,
            user=f"Session memories:\n{listing}",
            fast=True,
        )
    except Exception:
        log.exception("episode summary generation failed for session_key=%s", session_key)
        return None

    try:
        embedding = await llm.embed(summary)
    except Exception:
        log.warning("episode embedding failed for session_key=%s; saving without vector", session_key)
        embedding = None

    memory_ids = [m.id for m in memories]

    async with db.session_scope() as session:
        existing = (
            await session.scalars(
                select(Episode).where(
                    Episode.user_id == user_id,
                    Episode.session_key == session_key,
                )
            )
        ).first()

        if existing:
            existing.summary = summary
            existing.date = date
            existing.embedding = embedding
            existing.embedding_model = llm.embedding_model_id if embedding else None
            episode = existing
            await session.execute(
                delete(EpisodeMemory).where(EpisodeMemory.episode_id == episode.id)
            )
        else:
            episode = Episode(
                user_id=user_id,
                session_key=session_key,
                summary=summary,
                date=date,
                embedding=embedding,
                embedding_model=llm.embedding_model_id if embedding else None,
            )
            session.add(episode)
            await session.flush()

        session.add_all(
            [EpisodeMemory(episode_id=episode.id, memory_id=mid) for mid in memory_ids]
        )

    log.debug(
        "episode closed: session_key=%s, %d memories, summary=%s",
        session_key, len(memory_ids), summary[:80],
    )
    return episode
