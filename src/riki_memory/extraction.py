"""Decide what (if anything) from an exchange deserves to become a long-term memory.

Call :func:`extract_and_save_turn` after each turn, or collect several turns
and call it once per session — the cadence is the host app's choice. (In the
original Riki this per-turn function existed but nothing called it; here it is
the primary write path, so there is no dead code.)
"""

import logging

from pydantic import ValidationError
from sqlalchemy import select

from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .models import Memory
from .profile import refresh_profile_if_stale
from .recall import format_memories
from .schemas import Extraction, MemoryCreate
from .store import save_memory
from .timeutil import localize, now_in
from .users import ensure_user

log = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You maintain the long-term memory of a personal assistant for {name}.
Current date and time: {now} (timezone {timezone}).
Read the latest exchange and save what is worth remembering for months or years.

Save: stable facts about {name} and people in their life, preferences, plans and commitments,
important events (with dates), decisions, goals, things {name} is actively working on or
looking for (job search, projects, applications), recommendations or advice {name} asked
for and received, emotional states and moods when they're clearly expressed or strongly implied
("exhausted", "excited about X", "stressed about the deadline"), and active intentions or
near-term focus areas that shape what {name} cares about right now.
Do not save: small talk, one-off questions, general knowledge that anyone could look up,
assistant-only reasoning with no user context, secrets such as passwords or card numbers.

Rules:
- `kind` must be exactly one of: fact, preference, event, episode. There is no "plan" kind:
  a plan or commitment is an `event` (with event_at) if it has a date, otherwise a `fact`.
- One self-contained sentence per memory, naming who it is about ("{name} prefers ...").
- If the exchange changes or contradicts a known memory, put that memory's id in `replaces`.
- Never repeat anything listed under KNOWN MEMORIES or SAVED THIS TURN.
- CRITICAL: Convert ALL relative dates ("next Friday", "last Saturday", "two months ago",
  "recently") into absolute dates in BOTH the content text AND event_at. For example,
  if today is 2024-02-15 and the user says "last Saturday", write "on February 10th, 2024"
  in content and set event_at to that date. Never store relative dates like "last week" or
  "two months ago" — they become meaningless later.
- Preserve the specific details and terminology from the conversation. If the user describes
  a "GPS system not functioning correctly", use those exact words, not a paraphrase like
  "GPS issue". Specificity matters for later retrieval.
- Return an empty list when nothing is worth saving. That is the most common correct answer."""


async def propose_memories(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    user_message: str,
    reply: str,
    known: list[Memory] | None = None,
    turn_key: str | None = None,
    at=None,
) -> list[MemoryCreate]:
    """Ask the model what is worth remembering from one exchange.

    Returns validated, *unsaved* memories. ``known`` is the memories already
    shown to the model (from recall) — the model may only retire those.
    ``at`` is "now" for the prompt; pass the turn's original time when
    re-extracting history so "next Friday" resolves correctly.
    """
    user = await ensure_user(db, user_id, timezone=config.default_timezone)
    known = known or []

    if turn_key:
        async with db.session_scope() as session:
            saved_this_turn = (
                await session.scalars(select(Memory).where(Memory.source_key == turn_key))
            ).all()
    else:
        saved_this_turn = []

    result = await llm.chat_json(
        system=EXTRACTION_PROMPT.format(
            name=user.name, now=(at or now_in(user.timezone)).isoformat(), timezone=user.timezone
        ),
        user=(
            f"KNOWN MEMORIES:\n{format_memories(known)}\n\n"
            f"SAVED THIS TURN:\n{format_memories(saved_this_turn)}\n\n"
            f"{user.name}: {user_message}\nAssistant: {reply}"
        ),
        schema=Extraction,
        fast=True,
    )

    shown_ids = {str(m.id) for m in known}  # the model may only retire memories it was shown
    memories = []
    for item in result.memories:
        try:
            memories.append(
                MemoryCreate(
                    user_id=user.id,
                    content=item.content,
                    kind=item.kind,
                    importance=item.importance,
                    event_at=localize(item.event_at, user.timezone),
                    semantic=item.semantic,
                    source_key=turn_key,
                    entities=item.entities,
                    relations=item.relations,
                    replaces=[r for r in item.replaces if r in shown_ids],
                )
            )
        except ValidationError as error:
            log.warning("skipped an invalid memory from the extractor: %s", error)
    return memories


async def extract_and_save_turn(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    user_message: str,
    reply: str,
    known: list[Memory] | None = None,
    turn_key: str | None = None,
    on_important=None,
    at=None,
) -> list[Memory]:
    """Propose memories from one exchange, save them, and refresh the profile.

    Run this in the background after the reply is sent — it never needs to
    block the chat loop. Failures are logged, never raised.
    """
    try:
        proposed = await propose_memories(
            db, llm, config, user_id, user_message, reply, known, turn_key, at=at
        )
        saved = []
        for data in proposed:
            memory = await save_memory(db, llm, config, data, on_important=on_important)
            saved.append(memory)
        await refresh_profile_if_stale(db, llm, config, user_id)
        return saved
    except Exception:
        log.exception("memory extraction failed")
        return []
