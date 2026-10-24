"""The profile is a short summary of who the user is, rebuilt from their best memories."""

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select, update

from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .models import Memory, User
from .users import ensure_user

log = logging.getLogger(__name__)

PROFILE_PROMPT = """Write a profile of {name} for a personal assistant, from the memories below.
Plain prose, under 250 words, third person. Cover who they are, what they do, how they like to be helped,
the commitments or goals that matter most, and significant recent experiences that shape who they are.
Only use what the memories say. Do not invent anything."""


async def refresh_profile(
    db: Database, llm: LLMProvider, config: MemoryConfig, user_id: str
) -> str:
    async with db.session_scope() as session:
        memories = (
            await session.scalars(
                select(Memory)
                .where(Memory.user_id == user_id, Memory.is_active.is_(True))
                .order_by(Memory.importance.desc(), Memory.created_at.desc())
                .limit(80)
            )
        ).all()
    if not memories:
        async with db.session_scope() as session:
            user = await session.scalar(select(User).where(User.id == user_id))
            return user.profile if user else ""

    bullet_list = "\n".join(f"- {m.content}" for m in memories)
    text = (
        await llm.chat_text(
            system=PROFILE_PROMPT.format(name=(await ensure_user(db, user_id)).name),
            user=bullet_list,
            fast=True,
        )
    ).strip()
    if not text:
        async with db.session_scope() as session:
            user = await session.scalar(select(User).where(User.id == user_id))
            return user.profile if user else ""

    async with db.session_scope() as session:
        await session.execute(
            update(User)
            .where(User.id == user_id)
            .values(profile=text, profile_updated_at=datetime.now(timezone.utc))
        )
    return text


async def refresh_profile_if_stale(
    db: Database, llm: LLMProvider, config: MemoryConfig, user_id: str
) -> None:
    """Rebuild the profile once enough new memories have piled up.

    The original counted against a request-scoped user object, which could
    misfire; here the timestamp is read fresh from the database.
    """
    async with db.session_scope() as session:
        updated_at = await session.scalar(select(User.profile_updated_at).where(User.id == user_id))
    since = updated_at or datetime(1970, 1, 1, tzinfo=timezone.utc)
    async with db.session_scope() as session:
        new_count = await session.scalar(
            select(func.count())
            .select_from(Memory)
            .where(Memory.user_id == user_id, Memory.created_at > since)
        )
    if (new_count or 0) >= config.profile_refresh_after:
        try:
            await refresh_profile(db, llm, config, user_id)
        except Exception:
            log.exception("profile refresh failed")
