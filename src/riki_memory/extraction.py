"""Decide what (if anything) from an exchange deserves to become a long-term memory.

Call :func:`extract_and_save_turn` after each turn, or collect several turns
and call it once per session — the cadence is the host app's choice. (In the
original Riki this per-turn function existed but nothing called it; here it is
the primary write path, so there is no dead code.)
"""

import logging

from pydantic import ValidationError
from sqlalchemy import select

from . import telemetry
from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .models import Instance, Memory
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
- Preserve the specific details and terminology from the conversation. If the user says
  "the oven runs about 20 degrees hot", keep those words, not a paraphrase like
  "oven problem". Specificity matters for later retrieval.
- Tag each memory with 2-5 short lowercase topic labels in the `tags` field. Tags describe
  WHAT the memory is about — be specific. Examples: "job-interview", "kitchen-renovation",
  "marathon-training", "tax-filing", "school-admission", "visa-application", "food-allergy",
  "rent". Use hyphens for multi-word tags. Good tags enable later queries like
  "how many interviews" or "total spent on the renovation".
- NEVER drop numbers that belong to the memory. If the user says "$1,200 for the sofa",
  the memory MUST contain "$1,200". If "ran 21 km", the memory MUST say "21 km".
- NEVER normalize tense in a way that loses occurrence information. "attended a workshop"
  must stay past tense. "plans to buy" must NOT become "bought". "used to own" must
  NOT become "owns". The tense encodes whether something happened vs. is planned.
- Resolve entity references within the session: if turn 1 mentions "my laptop" and
  turn 3 identifies it as "the ThinkPad X1", use the specific name in the memory.
  "My manager's farewell" should become "Priya's farewell dinner" if names are given.
- Return an empty list when nothing is worth saving. That is the most common correct answer.

INSTANCE EXTRACTION (in the `instances` list INSIDE each memory):
For each memory that mentions a countable or aggregatable thing, list its instances
inside that memory. An instance belongs only to the memory that mentions it — never
copy an instance onto other memories from the same turn.
- kind: exactly one of event, item, visit, trip, media, person, activity, money, other.
- subject: the everyday noun you would count it by, singular — the X in "how many X".
  Prefer the broader class a person would ask about ("appliance" for a dishwasher,
  "course" for a Python bootcamp, "trip" for a weekend in Goa). Reuse a KNOWN SUBJECT
  whenever one fits; invent a new one only when none does.
- instance_key: normalized lowercase-hyphenated identity string. The same
  real-world thing MUST always get the same key. "Priya's farewell dinner"
  and "my manager's farewell" (if same event) = "priya-farewell-dinner".
  "Dr. Mehta the dentist" = "dr-mehta-dentist". Resolve pronouns.
- display_name: human-readable name for this specific instance.
- status: MANDATORY. 'occurred' for completed events/purchases/attendances.
  'current' for things currently owned or active. 'planned' for stated future
  intentions ("plans to return"). 'desired' for wishes ("wants to try").
  'former' for sold/ended/no-longer-true ("sold the old car", "used to have").
  NEVER default to 'occurred' — match the tense of the source text.
- numeric_value + unit: ONLY if the source text explicitly states a number.
  Never infer amounts. "$1,200" → numeric_value=1200, unit="USD".
  "ran 21 km" → numeric_value=21, unit="km".
- One sentence can yield zero, one, or many instances. "Bought a standing desk
  and an office chair" = two instances. "Went to a few concerts this year" = zero
  instances (too vague — only extract specific named things).
- date: when the instance occurred, if stated. Use absolute dates."""


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

    # Load existing instances so the extractor can resolve identity across sessions.
    known_instances_text = ""
    try:
        async with db.session_scope() as session:
            existing_instances = (
                await session.scalars(
                    select(Instance)
                    .where(Instance.user_id == user.id)
                    .order_by(Instance.created_at.desc())
                    .limit(100)
                )
            ).all()
        if existing_instances:
            lines = []
            for inst in existing_instances:
                date_str = inst.date.strftime("%Y-%m-%d") if inst.date else ""
                lines.append(
                    f"- [{inst.kind}/{inst.subject}] {inst.instance_key} "
                    f"({inst.status}{', ' + date_str if date_str else ''})"
                )
            subjects = sorted({inst.subject for inst in existing_instances})
            known_instances_text = (
                "\n\nKNOWN SUBJECTS: " + ", ".join(subjects)
                + "\n\nKNOWN INSTANCES (use the same instance_key for the same thing):\n"
                + "\n".join(lines)
            )
    except Exception:
        log.warning("could not load known instances for %s; extracting without them", user.id, exc_info=True)

    result = await llm.chat_json(
        system=EXTRACTION_PROMPT.format(
            name=user.name, now=(at or now_in(user.timezone)).isoformat(), timezone=user.timezone
        ),
        user=(
            f"KNOWN MEMORIES:\n{format_memories(known)}\n\n"
            f"SAVED THIS TURN:\n{format_memories(saved_this_turn)}"
            f"{known_instances_text}\n\n"
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
                    tags=item.tags,
                    replaces=[r for r in item.replaces if r in shown_ids],
                    instances=item.instances,
                )
            )
        except ValidationError as error:
            log.warning("skipped an invalid memory from the extractor: %s", error)
            telemetry.record_dropped()
    telemetry.record_proposed(len(memories))
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
    except Exception as error:
        log.exception("memory extraction failed")
        telemetry.record_failure(error)
        return []
