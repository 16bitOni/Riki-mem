"""Decide what (if anything) from an exchange deserves to become a long-term memory.

Call :func:`extract_and_save_turn` after each turn, or collect several turns
and call it once per session — the cadence is the host app's choice. (In the
original Riki this per-turn function existed but nothing called it; here it is
the primary write path, so there is no dead code.)
"""

import logging
import re
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from sqlalchemy import select

from . import telemetry
from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider, embed_texts
from .models import Instance, Memory
from .privacy import redact_secrets
from .profile import add_lines, refresh_profile_if_stale
from .recall import format_memories
from .router import PREGATE_RE
from .schemas import Extraction, MemoryCreate
from .store import save_memory
from .timeutil import localize, now_in
from .users import ensure_user

log = logging.getLogger(__name__)

# Byte-identical on every call (no name, date or timezone in it: those go in the user
# message), so providers that cache prompt prefixes bill it at the cached rate.
EXTRACTION_PROMPT = """You maintain the long-term memory of a personal assistant.
The user's name, the current date and time, and their timezone are under CONTEXT in the message.
Read the latest exchange and save what is worth remembering for months or years.

Save: stable facts about the user and people in their life, preferences, plans and commitments,
important events (with dates), decisions, goals, things the user is actively working on or
looking for (job search, projects, applications), recommendations or advice the user asked
for and received, emotional states and moods when they're clearly expressed or strongly implied
("exhausted", "excited about X", "stressed about the deadline"), and active intentions or
near-term focus areas that shape what the user cares about right now.
Do not save: small talk, one-off questions, general knowledge that anyone could look up,
assistant-only reasoning with no user context, secrets such as passwords or card numbers.

Rules:
- `kind` must be exactly one of: fact, preference, event, episode. There is no "plan" kind:
  a plan or commitment is an `event` (with event_at) if it has a date, otherwise a `fact`.
- Write memories in English, translating if the chat is in another language (Hinglish,
  Bengali, ...), so they can be found whatever language the next question is in. Keep
  names, places and quoted titles as they were said.
- One self-contained sentence per memory, naming who it is about with the user's name
  from CONTEXT ("Sub prefers ...", never "the user prefers ...").
- If the exchange changes or contradicts a known memory, put that memory's id in `replaces`.
- `source`: "stated" when the user said it, "inferred" when you concluded it from how they
  spoke (a mood, an implied preference), "assistant" when it comes from the assistant's
  reply (a recommendation the user received). Prefer saving what was stated.
- `valid_until` only for a temporary state with a known end ("I'm in Goa until Sunday",
  "on leave this week"): the absolute date it stops being true. Leave it empty otherwise.
- Never repeat anything listed under KNOWN MEMORIES or SAVED THIS TURN.
- CRITICAL: Convert ALL relative dates ("next Friday", "last Saturday", "two months ago",
  "recently") into absolute dates in BOTH the content text AND event_at. For example,
  if today is 2024-02-15 and the user says "last Saturday", write "on February 10th, 2024"
  in content and set event_at to that date. Never store relative dates like "last week" or
  "two months ago" — they become meaningless later.
- Preserve the specific details and terminology from the conversation. If the user says
  "the oven runs about 20 degrees hot", keep those words, not a paraphrase like
  "oven problem". Specificity matters for later retrieval.
- `importance`: 1 trivia ... 5 core to who they are. `semantic`: true for opinions,
  feelings, preferences and ideas; false for names, dates and numbers.
- `entities`: the people, places, organisations and projects named (type person, place, org,
  project or other). `relations`: how they connect, as a short verb phrase
  ({"subject": "Riya", "predicate": "sister of", "target": "Sub"}).
- `core`: only for the few facts worth having in every conversation, the block they belong
  in: identity (name, work, city, life situation), preferences (how they like things and
  like to be helped), constraints (allergies, diet, health limits, hard no's: anything an
  answer must never violate), people (who matters in their life), projects (active goals).
  Leave it empty for everything else; most memories have none.
- `attributes`: empty unless the APP-SPECIFIC INSTRUCTIONS ask for structured fields.
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


_WORD_RE = re.compile(r"[a-z0-9]+")
NEIGHBOURS = 8  # existing memories shown to the extractor when the caller passes none
SUBJECTS_SHOWN = 60
INSTANCES_SHOWN = 20


def _local(at, tz: str):
    """The turn's time in the user's timezone (a stored turn time is in UTC)."""
    return localize(at, tz).astimezone(ZoneInfo(tz)) if at else now_in(tz)


def _words(text: str) -> set[str]:
    """Lowercase words of 3+ letters, plus a crude singular ("plants" -> "plant")."""
    words = {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3}
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}


async def find_neighbours(
    db: Database, llm: LLMProvider, config: MemoryConfig, user, text: str
) -> list[Memory]:
    """The user's memories closest to ``text`` (keywords + meaning, no model call), so the
    extractor can retire a fact the turn changes even when the caller passed no ``known``.
    Costs one embedding and two indexed queries; a failing channel is skipped."""
    from .heuristic import make_heuristic_plan
    from .retrieval import keyword_search, reciprocal_rank_fusion, semantic_search

    rankings = []
    keywords = make_heuristic_plan(text, user.timezone).keywords
    if keywords:
        try:
            rankings.append(await keyword_search(db, user.id, keywords, None, None))
        except Exception:
            log.warning("neighbour keyword search failed", exc_info=True)
    try:
        rankings.append(await semantic_search(db, llm, config, user.id, text, None, None))
    except Exception:
        log.warning("neighbour semantic search failed", exc_info=True)
    ids = reciprocal_rank_fusion(rankings)[:NEIGHBOURS]
    if not ids:
        return []
    async with db.session_scope() as session:
        rows = {m.id: m for m in (await session.scalars(select(Memory).where(Memory.id.in_(ids)))).all()}
    return [rows[i] for i in ids if i in rows]


async def _known_instances_text(db: Database, user_id: str, turn_text: str) -> str:
    """Known subjects (short) plus only the instances this turn might be about.

    Sending every recent instance cost up to ~1.5k tokens per write, almost all irrelevant.
    shortcut: matching is by shared words over the 500 newest instances; use FTS or
    vectors if users with thousands of instances start missing key reuse.
    """
    async with db.session_scope() as session:
        rows = (await session.execute(
            select(Instance.kind, Instance.subject, Instance.instance_key, Instance.status, Instance.date)
            .where(Instance.user_id == user_id)
            .order_by(Instance.created_at.desc())
            .limit(500)
        )).all()
    if not rows:
        return ""
    subjects = sorted({r.subject for r in rows})[:SUBJECTS_SHOWN]
    words = _words(turn_text)
    relevant = [r for r in rows if words & _words(f"{r.subject} {r.instance_key.replace('-', ' ')}")]
    text = "\n\nKNOWN SUBJECTS: " + ", ".join(subjects)
    if relevant:
        lines = [
            f"- [{r.kind}/{r.subject}] {r.instance_key} ({r.status}"
            f"{', ' + r.date.strftime('%Y-%m-%d') if r.date else ''})"
            for r in relevant[:INSTANCES_SHOWN]
        ]
        text += "\n\nKNOWN INSTANCES (use the same instance_key for the same thing):\n" + "\n".join(lines)
    return text


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
    shown to the model (from recall) — the model may only retire those. When
    ``known`` is ``None`` the closest existing memories are looked up here
    (no model call), so updates work without the caller's help; pass ``[]``
    to show none. ``at`` is "now" for the prompt; pass the turn's original time
    when re-extracting history so "next Friday" resolves correctly.
    """
    user = await ensure_user(db, user_id, timezone=config.default_timezone)
    user_message, reply = redact_secrets(user_message), redact_secrets(reply)  # never sent to the model
    if known is None:
        known = await find_neighbours(db, llm, config, user, user_message)

    if turn_key:
        async with db.session_scope() as session:
            saved_this_turn = (
                await session.scalars(
                    select(Memory).where(Memory.user_id == user.id, Memory.source_key == turn_key)
                )
            ).all()
    else:
        saved_this_turn = []

    # Existing instances let the extractor resolve identity across sessions.
    try:
        known_instances_text = await _known_instances_text(db, user.id, f"{user_message} {reply}")
    except Exception:
        known_instances_text = ""
        log.warning("could not load known instances for %s; extracting without them", user.id, exc_info=True)

    system = EXTRACTION_PROMPT
    if config.extraction_instructions.strip():
        system += "\n\nAPP-SPECIFIC INSTRUCTIONS (follow these too):\n" + config.extraction_instructions.strip()
    result = await llm.chat_json(
        system=system,
        user=(
            f"CONTEXT\nUser's name: {user.name}\n"
            f"Current date and time: {_local(at, user.timezone).isoformat()} (timezone {user.timezone})\n\n"
            f"KNOWN MEMORIES:\n{format_memories(known)}\n\n"
            f"SAVED THIS TURN:\n{format_memories(saved_this_turn)}"
            f"{known_instances_text}\n\n"
            f"{user.name}: {user_message}\nAssistant: {reply}"
        ),
        schema=Extraction,
        fast=True,
    )

    shown_ids = {str(m.id) for m in known}  # the model may only retire memories it was shown
    stated_ids = {str(m.id) for m in known if m.source == "stated"}
    turn_time = _local(at, user.timezone)
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
                    # a guess never overrules what the user actually said
                    replaces=[r for r in item.replaces if r in shown_ids
                              and not (item.source != "stated" and r in stated_ids)],
                    source=item.source,
                    core=item.core,
                    # what it replaces was true until this happened (or until this turn)
                    changed_at=localize(item.event_at, user.timezone) or turn_time,
                    valid_to=localize(item.valid_until, user.timezone),
                    instances=item.instances,
                    attributes=item.attributes,
                )
            )
        except ValidationError as error:
            log.warning("skipped an invalid memory from the extractor: %s", error)
            telemetry.record_dropped()
    telemetry.record_proposed(len(memories))
    return memories


async def save_proposed(
    db: Database, llm: LLMProvider, config: MemoryConfig, proposed: list[MemoryCreate],
    on_important=None,
) -> list[Memory]:
    """Save extracted memories, embedding all that want a vector in one call."""
    wanted = [
        i for i, data in enumerate(proposed)
        if (data.semantic if data.semantic is not None else config.embed_all_kinds)
    ]
    vectors: dict[int, list[float]] = {}
    if wanted:
        try:
            made = await embed_texts(llm, [proposed[i].content for i in wanted])
            vectors = dict(zip(wanted, made))
        except Exception:
            log.warning("batch embedding failed; embedding one memory at a time", exc_info=True)
    saved, core = [], []
    for i, data in enumerate(proposed):
        memory = await save_memory(db, llm, config, data, on_important=on_important, vector=vectors.get(i))
        saved.append(memory)
        if data.core and memory.is_active:
            core.append((data.core, memory.content, memory.id))
    if core:
        try:
            await add_lines(db, proposed[0].user_id, core)
        except Exception:
            log.warning("could not add core-memory lines; the next update will", exc_info=True)
    return saved


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
    block the chat loop. Failures are logged, never raised. A user message that
    is only chitchat ("ok thanks") is skipped without any model call.
    """
    if config.heuristic_pregate and PREGATE_RE.match(user_message or ""):
        telemetry.record_skipped()
        return []
    try:
        proposed = await propose_memories(
            db, llm, config, user_id, user_message, reply, known, turn_key, at=at
        )
        saved = await save_proposed(db, llm, config, proposed, on_important=on_important)
        await refresh_profile_if_stale(db, llm, config, user_id)
        return saved
    except Exception as error:
        log.exception("memory extraction failed")
        telemetry.record_failure(error)
        return []
