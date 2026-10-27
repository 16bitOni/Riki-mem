"""Core memory blocks: a few short, labelled lists of what matters most about the user.

Like Letta's core memory: small enough to sit in the prompt, edited a line at a time
(never regenerated), and editable by the host agent (``get_blocks`` / ``set_block``).

- Kept on ``users.blocks`` as ``{label: [{"text": ..., "from": [memory ids]}]}``, so
  deleting a memory also removes the lines written from it.
- Updated in the background once ``profile_refresh_after`` new memories have piled up:
  one fast-model call that sees the current blocks and only the *new* memories, and
  returns add/remove edits. (The old profile re-read 80 memories and rewrote everything.)
- ``users.profile`` holds the rendered blocks, for code that reads the profile text.
"""

import logging
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select, update

from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .models import Memory, User

log = logging.getLogger(__name__)

BlockLabel = Literal["identity", "preferences", "constraints", "people", "projects"]
BLOCK_LABELS: tuple[str, ...] = BlockLabel.__args__
BLOCK_HINTS = {
    "identity": "who they are: name, work, location, life situation",
    "preferences": "how they like things and like to be helped",
    "constraints": "never violate: allergies, diet, health limits, hard no's",
    "people": "the important people in their life and who they are",
    "projects": "goals and things they are actively working on",
}
MAX_LINES = 12  # per block; the oldest line goes when a block is full
MAX_LINE_CHARS = 200
NEW_MEMORIES_SHOWN = 40

BLOCKS_PROMPT = """You keep a personal assistant's core memory about the user: five short blocks of one-line facts.
""" + "\n".join(f"- {label}: {hint}" for label, hint in BLOCK_HINTS.items()) + """

You get the current blocks and memories saved since the last update. Return edits:
- add: a durable, important fact that belongs in a block and is not there yet (one short line,
  third person, with the memory ids it comes from);
- remove: a line (copied exactly) that the new memories show is no longer true.
Most memories need no edit: trivia, one-off events and moods stay out of the blocks.
Return an empty list when nothing changes."""


class BlockEdit(BaseModel):
    block: BlockLabel
    op: Literal["add", "remove"]
    text: str = Field(description="The line to add, or the exact line to remove.")
    memory_ids: list[str] = Field(default_factory=list, description="Ids of the memories it comes from.")


class BlockEdits(BaseModel):
    edits: list[BlockEdit]


def render(blocks: dict | None) -> str:
    """The blocks as prompt text, non-empty blocks only."""
    parts = []
    for label in BLOCK_LABELS:
        lines = (blocks or {}).get(label) or []
        if lines:
            parts.append(f"{label}:\n" + "\n".join(f"- {line['text']}" for line in lines))
    return "\n".join(parts)


def apply_edits(blocks: dict | None, edits: list[BlockEdit], known_ids: set[str]) -> dict:
    """New blocks with ``edits`` applied. Sources are kept only if they are real memory ids."""
    out = {label: list((blocks or {}).get(label) or []) for label in BLOCK_LABELS}
    for edit in edits:
        lines = out[edit.block]
        text = " ".join(edit.text.split())[:MAX_LINE_CHARS]
        if edit.op == "remove":
            out[edit.block] = [line for line in lines if line["text"] != text]
        elif text and all(line["text"] != text for line in lines):
            lines.append({"text": text, "from": [i for i in edit.memory_ids if i in known_ids]})
            del lines[:-MAX_LINES]
    return {label: lines for label, lines in out.items() if lines}


def without_sources(blocks: dict | None, memory_ids: set[str]) -> dict:
    """Blocks minus every line written from one of ``memory_ids`` (after a hard delete)."""
    return {
        label: kept for label, lines in (blocks or {}).items()
        if (kept := [line for line in lines if not set(line.get("from") or []) & memory_ids])
    }


async def save_blocks(db: Database, user_id: str, blocks: dict, *, refreshed: bool = False) -> None:
    values = {"blocks": blocks, "profile": render(blocks)}
    if refreshed:
        values["profile_updated_at"] = datetime.now(timezone.utc)
    async with db.session_scope() as session:
        await session.execute(update(User).where(User.id == user_id).values(**values))


async def get_blocks(db: Database, user_id: str) -> dict:
    async with db.session_scope() as session:
        return (await session.scalar(select(User.blocks).where(User.id == user_id))) or {}


async def refresh_profile(db: Database, llm: LLMProvider, config: MemoryConfig, user_id: str) -> str:
    """Edit the blocks from the memories saved since the last update; returns the rendered text."""
    async with db.session_scope() as session:
        user = await session.scalar(select(User).where(User.id == user_id))
        if user is None:
            return ""
        since = user.profile_updated_at or datetime(1970, 1, 1, tzinfo=timezone.utc)
        new = (await session.scalars(
            select(Memory)
            .where(Memory.user_id == user_id, Memory.is_active.is_(True), Memory.created_at > since)
            .order_by(Memory.importance.desc(), Memory.created_at.desc())
            .limit(NEW_MEMORIES_SHOWN)
        )).all()
    blocks = user.blocks or {}
    if not new:
        return render(blocks)
    result = await llm.chat_json(
        system=BLOCKS_PROMPT,
        user=(
            f"User's name: {user.name}\n\nCURRENT BLOCKS:\n{render(blocks) or '(empty)'}\n\n"
            "NEW MEMORIES:\n" + "\n".join(f"- {m.content}  (id: {m.id})" for m in new)
        ),
        schema=BlockEdits,
        fast=True,
    )
    blocks = apply_edits(blocks, result.edits, {m.id for m in new})
    await save_blocks(db, user_id, blocks, refreshed=True)
    return render(blocks)


async def refresh_profile_if_stale(
    db: Database, llm: LLMProvider, config: MemoryConfig, user_id: str
) -> None:
    """Update the blocks once ``profile_refresh_after`` new memories have piled up."""
    from sqlalchemy import func

    async with db.session_scope() as session:
        updated_at = await session.scalar(select(User.profile_updated_at).where(User.id == user_id))
        new_count = await session.scalar(
            select(func.count()).select_from(Memory).where(
                Memory.user_id == user_id,
                Memory.created_at > (updated_at or datetime(1970, 1, 1, tzinfo=timezone.utc)),
            )
        )
    if (new_count or 0) >= config.profile_refresh_after:
        try:
            await refresh_profile(db, llm, config, user_id)
        except Exception:
            log.exception("core memory update failed")
