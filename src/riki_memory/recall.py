"""The memory router: decide what to look up, look it up, and format it for the LLM.

The plan can be built two ways (``MemoryConfig.planner_mode``):

- ``"llm"`` — the planner model rewrites the query (best at resolving
  references like "that role" from recent chat; costs one fast-model call);
- ``"heuristic"`` — keywords/entities/dates from regex, zero chat tokens;
- ``"escalate"`` (default) — heuristic first, LLM planner only when cheap
  retrieval finds nothing.

Fixes applied while extracting from Riki:

- the planner runs on the **fast** model (the original burned the main model on
  every message, even "hi");
- a heuristic pre-gate answers greetings/thanks/farewells without any
  LLM call at all (``MemoryConfig.heuristic_pregate``);
- the user profile is injected only when recall found nothing by default
  (``MemoryConfig.profile_injection``), instead of ~350 tokens every turn.
"""

import logging
import re
from datetime import datetime

from sqlalchemy import select

from .config import MemoryConfig
from .db import Database
from .heuristic import extract_temporal_anchors, is_aggregation_query, is_temporal_arithmetic, make_heuristic_plan
from .llm import LLMProvider
from .models import CategoryFact, Memory
from .retrieval import hybrid_search
from .schemas import RetrievalPlan
from .store import _TAG_TO_CATEGORY
from .timeutil import localize, now_in
from .users import ensure_user

log = logging.getLogger(__name__)

PLANNER_PROMPT = """You decide how to search a person's long-term memory so an assistant can answer their latest message.
Current date and time: {now} (timezone {timezone}).

First: resolve references. If the latest message says "these companies", "that role", "the thing
I mentioned" — look at the recent chat to figure out what they mean. Use the resolved names and
topics (not the pronouns) when filling in the fields below.

- needs_memory: false for greetings, thanks, or messages that need nothing personal.
- keywords: exact words likely stored in a memory (names, places, objects). Use a few strong ones.
- semantic_query: a rewrite capturing the meaning, for things like preferences, opinions or plans.
- entities: people, places, organisations or projects mentioned or referenced.
- relation_words: relationship words ("my sister's husband's company" -> sister, husband, company). Use the user's own words.
- tags: topic/category tags to search for. CRITICAL for counting or aggregation questions
  ("how many weddings?", "total bike expenses", "all my doctor visits"). Use specific
  lowercase hyphenated labels like "wedding", "bike-purchase", "doctor-visit", "model-kit".
- start / end: only if the message refers to a time ("last week", "in March", "yesterday"). Use ISO 8601 with the timezone offset.
Leave fields empty rather than guessing."""

# Messages that obviously need nothing personal. Checked before any LLM call.
_PREGATE = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|sup|good\s?(morning|afternoon|evening|night|day)|"
    r"thanks?|thank\s?you|thx|bye+|goodbye|see\s?you|ok|okay|k|cool|nice|great|"
    r"lol|haha)+\s*[!?.…]*\s*$",
    re.IGNORECASE,
)


def _pregated_plan() -> RetrievalPlan:
    return RetrievalPlan(needs_memory=False)


async def make_plan(
    llm: LLMProvider,
    config: MemoryConfig,
    message: str,
    recent_chat: str,
    timezone: str,
) -> RetrievalPlan:
    if config.heuristic_pregate and _PREGATE.match(message or ""):
        return _pregated_plan()
    plan = await llm.chat_json(
        system=PLANNER_PROMPT.format(now=now_in(timezone).isoformat(), timezone=timezone),
        user=f"Recent chat:\n{recent_chat or '(none)'}\n\nLatest message:\n{message}",
        schema=RetrievalPlan,
        fast=True,  # routing never needs the main model
    )
    plan.start = localize(plan.start, timezone)
    plan.end = localize(plan.end, timezone)
    return plan


def format_memories(memories: list[Memory]) -> str:
    if not memories:
        return "(nothing found)"
    lines = []
    for m in memories:
        when = (m.event_at or m.created_at).strftime("%Y-%m-%d")
        lines.append(f"- [{m.kind}, {when}] {m.content}  (id: {m.id})")
    return "\n".join(lines)


def _apply_not_after(plan: RetrievalPlan, not_after: datetime | None) -> RetrievalPlan:
    """Cap the plan's end date to not_after if it is tighter (or if end is unset)."""
    if not_after is None:
        return plan
    if plan.end is None or not_after < plan.end:
        return plan.model_copy(update={"end": not_after})
    return plan


def _filter_by_not_after(memories: list[Memory], not_after: datetime | None) -> list[Memory]:
    """Remove memories whose event_at post-dates not_after.

    Only event_at is checked — memories without event_at are always kept (they
    are timeless facts). This is a post-retrieval safety net: the time channel
    respects plan.end, but keyword/semantic channels ignore it.
    """
    if not_after is None:
        return memories
    kept = []
    for m in memories:
        ea = m.event_at
        if ea is None:
            kept.append(m)
            continue
        # Normalise: if event_at is naive, treat as UTC for comparison.
        if ea.tzinfo is None:
            from datetime import timezone as _tz
            ea = ea.replace(tzinfo=_tz.utc)
        if ea <= not_after:
            kept.append(m)
    return kept


def _dedup_by_event(memories: list[Memory]) -> list[Memory]:
    """Collapse same-event duplicates: keep one memory per (event_at_date, kind).

    When multiple memories describe the same event from different angles, they
    waste recall slots that should go to distinct events. We keep the most
    informative one (longest content). Memories without event_at are never
    deduplicated — they are timeless facts that cannot be clustered by date.
    """
    dated: dict[tuple, Memory] = {}
    undated: list[Memory] = []
    for m in memories:
        if m.event_at is None:
            undated.append(m)
        else:
            key = (m.event_at.date() if hasattr(m.event_at, "date") else m.event_at, m.kind)
            existing = dated.get(key)
            if existing is None or len(m.content) > len(existing.content):
                dated[key] = m
    # Re-merge in original rank order so dedup doesn't scramble scores.
    deduped_ids = {m.id for m in dated.values()}
    result = []
    seen_keys: set[tuple] = set()
    for m in memories:
        if m.event_at is None:
            result.append(m)
        else:
            key = (m.event_at.date() if hasattr(m.event_at, "date") else m.event_at, m.kind)
            if key not in seen_keys and m.id in deduped_ids:
                seen_keys.add(key)
                result.append(m)
    return result


async def _category_facts_memories(
    db: Database,
    user_id: str,
    tags: list[str],
    not_after: datetime | None,
) -> list[Memory]:
    """Return the source memories for all CategoryFact rows matching the given tags.

    This is the structured aggregation path: O(1) index lookup → JOIN to memories.
    Falls back to an empty list if the category_facts table doesn't exist yet
    (pre-migration databases) or no matching facts are found.
    """
    categories = list({
        _TAG_TO_CATEGORY[t.lower().strip()]
        for t in tags
        if t.lower().strip() in _TAG_TO_CATEGORY
    })
    if not categories:
        return []
    try:
        async with db.session_scope() as session:
            stmt = (
                select(Memory)
                .join(CategoryFact, CategoryFact.source_memory_id == Memory.id)
                .where(
                    CategoryFact.user_id == user_id,
                    CategoryFact.category.in_(categories),
                    Memory.is_active.is_(True),
                )
                .order_by(Memory.event_at.asc().nullslast())
            )
            if not_after is not None:
                from sqlalchemy import or_, func
                stmt = stmt.where(
                    or_(
                        Memory.event_at.is_(None),
                        Memory.event_at <= not_after,
                    )
                )
            rows = (await session.scalars(stmt)).all()
        return list(rows)
    except Exception:
        log.debug("category_facts query failed (table may not exist yet); falling back")
        return []


async def _dispatch_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    plan: RetrievalPlan,
    query: str,
    limit: int | None = None,
) -> tuple[list[Memory], dict]:
    """Route to agentic or single-shot hybrid search based on config."""
    if config.retrieval_mode == "agentic":
        from .agentic import agentic_search
        return await agentic_search(db, llm, config, user_id, plan, rerank_query=query,
                                    limit=limit)
    return await hybrid_search(db, llm, config, user_id, plan, rerank_query=query,
                               limit=limit)


async def _temporal_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    query: str,
    not_after: datetime | None = None,
) -> tuple[list[Memory], dict] | None:
    """Decompose a temporal-arithmetic question into per-anchor searches.

    Returns (memories, report) with a computed date difference, or None
    if anchor extraction fails (caller falls through to normal search).
    """
    anchors = extract_temporal_anchors(query)
    if len(anchors) < 2:
        return None

    all_memories: list[Memory] = []
    anchor_dates: list[datetime | None] = []

    for anchor in anchors:
        plan = make_heuristic_plan(anchor, config.default_timezone)
        plan.needs_memory = True
        plan = _apply_not_after(plan, not_after)
        found, _ = await _dispatch_search(db, llm, config, user_id, plan, anchor, limit=5)
        found = _filter_by_not_after(found, not_after)

        best_date: datetime | None = None
        for m in found:
            if m.event_at is not None:
                best_date = m.event_at
                break
        anchor_dates.append(best_date)
        all_memories.extend(found)

    # Dedup by memory id
    seen = set()
    unique = []
    for m in all_memories:
        if m.id not in seen:
            seen.add(m.id)
            unique.append(m)

    report: dict = {
        "needs_memory": True,
        "planner": "temporal",
        "temporal_arithmetic": True,
        "anchors": anchors,
        "anchor_dates": [d.isoformat() if d else None for d in anchor_dates],
        "candidates": len(unique),
        "returned": len(unique),
    }

    if anchor_dates[0] and anchor_dates[1]:
        diff = abs((anchor_dates[1] - anchor_dates[0]).days)
        report["computed_days"] = diff
        report["computed_answer"] = f"{diff} days"

    return unique, report


async def search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    query: str,
    recent_chat: str = "",
    not_after: datetime | None = None,
) -> tuple[list[Memory], dict]:
    """Plan + hybrid search. Returns ``(memories, report)``.

    ``report["planner"]`` says which plan built the query: ``"heuristic"``
    (zero chat tokens) or ``"llm"``.

    ``not_after``: cap retrieval to memories dated at or before this datetime.
    Pass the question date so future-dated memories don't pollute temporal questions.
    """
    user = await ensure_user(db, user_id, timezone=config.default_timezone)
    mode = config.planner_mode

    # Aggregation queries ("how many X", "total", "all my") need the LLM planner
    # so it can infer temporal bounds from named events ("before Run for the Cure").
    # The heuristic escalate path short-circuits as soon as it finds *any* result,
    # which means it never generates a proper date window for counting questions.
    # Temporal arithmetic: decompose into per-anchor searches before anything else.
    if is_temporal_arithmetic(query or ""):
        result = await _temporal_search(db, llm, config, user.id, query, not_after)
        if result is not None:
            return result

    is_aggregation = is_aggregation_query(query or "")

    if mode == "heuristic":
        plan = make_heuristic_plan(
            query, user.timezone, pregated=bool(config.heuristic_pregate and _PREGATE.match(query or ""))
        )
        planner = "heuristic"
    elif mode == "escalate" and not is_aggregation:
        plan = make_heuristic_plan(
            query, user.timezone, pregated=bool(config.heuristic_pregate and _PREGATE.match(query or ""))
        )
        if plan.needs_memory:
            plan = _apply_not_after(plan, not_after)
            found, report = await _dispatch_search(db, llm, config, user.id, plan, query)
            if report["candidates"]:
                report["needs_memory"] = True
                report["planner"] = "heuristic"
                found = _filter_by_not_after(found, not_after)
                return found, report
            log.debug("heuristic plan found nothing; escalating to the LLM planner")
        plan = await make_plan(llm, config, query, recent_chat, user.timezone)
        planner = "llm"
    else:  # "llm" or "escalate" with aggregation query
        if mode == "escalate" and is_aggregation:
            log.debug("aggregation query detected; skipping heuristic and using LLM planner directly")
        plan = await make_plan(llm, config, query, recent_chat, user.timezone)
        planner = "llm"

    if not plan.needs_memory:
        return [], {"needs_memory": False, "planner": planner}
    plan = _apply_not_after(plan, not_after)

    # SQL aggregation path: bypass RRF for counting/summing questions.
    if is_aggregation and config.enable_sql_aggregation and plan.tags:
        from .aggregation import (
            aggregation_to_memories,
            execute_aggregation,
            parse_aggregation_query,
        )
        agg_query = parse_aggregation_query(query, plan, not_after=not_after)
        if agg_query:
            result = await execute_aggregation(db, user.id, agg_query)
            if result.evidence:
                memories = await aggregation_to_memories(db, result.evidence)
                memories = _filter_by_not_after(memories, not_after)
                report = {
                    "needs_memory": True,
                    "planner": planner,
                    "aggregation": True,
                    "sql_result": result.value,
                    "sql_count": result.count,
                    "candidates": len(result.evidence),
                    "returned": len(memories),
                }
                return memories, report

    # Use a higher recall limit for aggregation so all instances fit in the window.
    effective_limit = config.aggregation_recall_limit if is_aggregation else None

    found, report = await _dispatch_search(db, llm, config, user.id, plan, query,
                                           limit=effective_limit)
    report["needs_memory"] = True
    report["planner"] = planner

    # Post-retrieval safety net: filter out memories that post-date the question.
    # Keyword/semantic channels don't respect plan.end, so this catches stragglers.
    found = _filter_by_not_after(found, not_after)

    # Collapse same-date/same-kind duplicates for aggregation queries so every
    # recall slot holds a distinct event.
    if is_aggregation:
        found = _dedup_by_event(found)

    return found, report


async def build_memory_context(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    message: str,
    recent_chat: str = "",
    name: str = "",
) -> tuple[str, list[Memory]]:
    """Text to inject into the system prompt, plus the memories it contains.

    The profile block follows ``config.profile_injection``: by default it is
    only included when recall found nothing (the profile usually restates what
    the memories already say).
    """
    user = await ensure_user(db, user_id, name=name, timezone=config.default_timezone)
    memories, _ = await search(db, llm, config, user.id, message, recent_chat)

    inject_profile = (
        config.profile_injection == "always"
        or (config.profile_injection == "when_empty" and not memories)
    )
    profile_block = f"## About {user.name}\n{user.profile or '(no profile yet)'}\n\n" if inject_profile else ""
    context = (
        f"{profile_block}"
        f"## Memories that may be relevant to this message\n{format_memories(memories)}"
    )
    return context, memories
