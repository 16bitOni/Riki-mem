"""The memory router: decide what to look up, look it up, and format it for the LLM.

Every message is routed first (``router.route_query``, zero tokens):

- ``chitchat``    — nothing is looked up;
- ``temporal``    — anchors are dated and the difference is computed in code;
- ``aggregate``   — counted/summed from the Instance table in SQL;
- ``exploratory`` and ``lookup`` — hybrid retrieval.

A computed answer (count, sum, day difference) travels in ``report["computed"]``
and is rendered at the top of ``context()``, with what it counted and what it
left out, so the host LLM never has to count prose itself. A route that cannot
compute (no matching subject, an undated anchor) falls back to ``lookup`` and
the report says why.

The search plan can be built three ways (``MemoryConfig.planner_mode``):

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
from datetime import datetime

from sqlalchemy import select

from . import temporal
from .aggregation import (
    aggregation_to_memories,
    computed_view,
    execute_aggregation,
    parse_aggregation_query,
)
from .config import MemoryConfig
from .db import Database
from .heuristic import make_heuristic_plan
from .llm import LLMProvider
from .models import Instance, Memory
from .retrieval import hybrid_search
from .router import PREGATE_RE, route_query
from .schemas import RetrievalPlan
from .similarity import trigram_similarity
from .store import _slot_key
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
- tags: topic tags to search for. CRITICAL for counting or aggregation questions
  ("how many weddings?", "total bike expenses", "all my doctor visits"): put the plain
  noun being counted FIRST, singular and lowercase ("wedding", "bike", "doctor"), then
  any other useful topic labels.
- start / end: only if the message refers to a time ("last week", "in March", "yesterday"). Use ISO 8601 with the timezone offset.
Leave fields empty rather than guessing."""


def _pregated_plan() -> RetrievalPlan:
    return RetrievalPlan(needs_memory=False)


async def make_plan(
    llm: LLMProvider,
    config: MemoryConfig,
    message: str,
    recent_chat: str,
    timezone: str,
    now: datetime | None = None,
) -> RetrievalPlan:
    if config.heuristic_pregate and PREGATE_RE.match(message or ""):
        return _pregated_plan()
    # Use the question's reference time (not_after) so that temporal phrases like
    # "this year" or "last month" resolve relative to when the question was asked,
    # not to today's date (which would exclude historical conversations).
    ref_now = now or now_in(timezone)
    plan = await llm.chat_json(
        system=PLANNER_PROMPT.format(now=ref_now.isoformat(), timezone=timezone),
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


def _instance_line(item: dict, reason: str | None = None) -> str:
    parts = [item["name"], reason or item["status"]]
    if item.get("date"):
        parts.append(item["date"])
    if item.get("value") is not None:
        parts.append(f"{item['value']:g} {item.get('unit') or ''}".strip())
    parts.append(f"memory {item['memory_id']}")
    return "- " + " | ".join(parts)


def format_computed(computed: dict | None) -> str:
    """The deterministic answer, its evidence, and its exclusions, for the prompt."""
    if not computed:
        return ""
    lines = ["## Computed from memory (deterministic; use this instead of counting the memories yourself)"]
    if computed["type"] == "temporal":
        a, b = computed["anchors"][0], computed["anchors"][1]
        lines.append(
            f'{computed["unit"].upper()} between "{a["text"]}" ({a["date"]}) and '
            f'"{b["text"]}" ({b["date"]}) = {computed["value"]}'
            + (f' ({computed["days"]} days)' if computed["unit"] != "days" else "")
        )
        for anchor in (a, b):
            if anchor["memory_id"]:
                lines.append(f'- "{anchor["text"]}" dated from memory {anchor["memory_id"]}')
        return "\n".join(lines)

    operation = {"COUNT_DISTINCT": "COUNT", "SUM": "SUM", "AVG": "AVERAGE", "LIST": "LIST"}[computed["operation"]]
    statuses = "/".join(computed["status_filter"]) or "any"
    rule = ", newest mention decides" if computed["current_state"] else ""
    unit = f' {computed["unit"]}' if computed.get("unit") else ""
    lines.append(
        f'{operation} of {", ".join(computed["subjects"])} (status: {statuses}{rule}) = {computed["value"]}{unit}'
    )
    lines.append("Counted:" if computed["included"] else "Counted: nothing")
    lines.extend(_instance_line(item) for item in computed["included"])
    if computed["excluded"]:
        lines.append("Left out:")
        lines.extend(_instance_line(item, item["reason"]) for item in computed["excluded"])
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


async def _dedup(db: Database, config: MemoryConfig, memories: list[Memory]) -> list[Memory]:
    """Drop results that repeat an earlier (higher-ranked) one, keeping rank order.

    A result is a repeat when either
    - every instance it backs is already backed by a kept result (same subject,
      key and day — so two different bike repairs under one key both stay), or
    - it says nearly the same thing on the same date slot (the rule save-time
      dedup uses).

    Distinct things on the same day are never collapsed.
    """
    if len(memories) < 2:
        return memories
    async with db.session_scope() as session:
        rows = (await session.scalars(
            select(Instance).where(Instance.source_memory_id.in_([m.id for m in memories]))
        )).all()
    identities: dict[str, set[tuple]] = {}
    for row in rows:
        identities.setdefault(row.source_memory_id, set()).add(
            (row.subject, row.instance_key, row.date.date() if row.date else None)
        )

    kept: list[Memory] = []
    covered: set[tuple] = set()
    for m in memories:
        mine = identities.get(m.id, set())
        if mine and mine <= covered:
            continue
        if any(
            _slot_key(k.event_at) == _slot_key(m.event_at)
            and trigram_similarity(k.content, m.content) >= config.dedup_trigram_threshold
            for k in kept
        ):
            continue
        kept.append(m)
        covered |= mine
    return kept


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


async def _temporal(
    db: Database, llm: LLMProvider, config: MemoryConfig, user, query: str,
    not_after: datetime | None,
) -> tuple[list[Memory], dict]:
    async def search_anchor(anchor: str) -> list[Memory]:
        plan = make_heuristic_plan(anchor, user.timezone, now=not_after)
        plan = _apply_not_after(plan.model_copy(update={"needs_memory": True}), not_after)
        found, _ = await _dispatch_search(db, llm, config, user.id, plan, anchor, limit=5)
        return _filter_by_not_after(found, not_after)

    now = not_after or now_in(user.timezone)
    memories, report = await temporal.compute(db, llm, query, search_anchor, now)
    report["needs_memory"] = True
    report["planner"] = "temporal"
    return memories, report


async def _lookup(
    db: Database, llm: LLMProvider, config: MemoryConfig, user, query: str, recent_chat: str,
    not_after: datetime | None, limit: int | None,
) -> tuple[list[Memory], dict]:
    """Plan + hybrid search, honouring ``planner_mode``."""
    pregated = bool(config.heuristic_pregate and PREGATE_RE.match(query or ""))
    mode = config.planner_mode
    if mode == "heuristic":
        plan, planner = make_heuristic_plan(query, user.timezone, pregated=pregated, now=not_after), "heuristic"
    else:
        if mode == "escalate":
            cheap = make_heuristic_plan(query, user.timezone, pregated=pregated, now=not_after)
            if cheap.needs_memory:
                cheap = _apply_not_after(cheap, not_after)
                found, report = await _dispatch_search(db, llm, config, user.id, cheap, query, limit=limit)
                if report["candidates"]:
                    report["planner"] = "heuristic"
                    return _filter_by_not_after(found, not_after), report
                log.debug("heuristic plan found nothing; escalating to the LLM planner")
        plan = await make_plan(llm, config, query, recent_chat, user.timezone, now=not_after)
        planner = "llm"

    if not plan.needs_memory:
        return [], {"needs_memory": False, "planner": planner}
    plan = _apply_not_after(plan, not_after)
    found, report = await _dispatch_search(db, llm, config, user.id, plan, query, limit=limit)
    report["planner"] = planner
    return _filter_by_not_after(found, not_after), report


async def _aggregate(
    db: Database, llm: LLMProvider, config: MemoryConfig, user, query: str, recent_chat: str,
    not_after: datetime | None,
) -> tuple[list[Memory], dict]:
    limit = config.aggregation_recall_limit
    if config.planner_mode == "heuristic":
        plan, planner = make_heuristic_plan(query, user.timezone, now=not_after), "heuristic"
    else:
        plan = await make_plan(llm, config, query, recent_chat, user.timezone, now=not_after)
        planner = "llm"
    if not plan.needs_memory:
        return [], {"needs_memory": False, "planner": planner}
    plan = _apply_not_after(plan, not_after)

    sql_report: dict = {"aggregation": "disabled"}
    if config.enable_sql_aggregation:
        agg_query = parse_aggregation_query(query, plan, not_after=not_after)
        if agg_query is None:
            sql_report = {"aggregation": "unparsed"}
        else:
            result = await execute_aggregation(
                db, user.id, agg_query, llm=llm, subject_threshold=config.subject_match_threshold
            )
            if not result.resolved:
                sql_report = {"aggregation": "no_matching_subject", "sql_terms": agg_query.terms}
            else:
                return await _aggregate_answer(
                    db, llm, config, user, query, plan, planner, not_after, limit, agg_query, result
                )

    # Nothing computable: rank memories instead, with a window big enough for every instance.
    found, report = await _dispatch_search(db, llm, config, user.id, plan, query, limit=limit)
    found = _filter_by_not_after(found, not_after)
    report["planner"] = planner
    if not found and planner == "llm":
        # The LLM plan may have a wrong window or missing tags; try the regex plan once.
        fb_plan = make_heuristic_plan(query, user.timezone, now=not_after)
        fb_plan = _apply_not_after(fb_plan.model_copy(update={"needs_memory": True}), not_after)
        fb_found, fb_report = await _dispatch_search(db, llm, config, user.id, fb_plan, query, limit=limit)
        fb_found = _filter_by_not_after(fb_found, not_after)
        if fb_found:
            log.debug("aggregation fallback heuristic search found %d results", len(fb_found))
            found, report = fb_found, fb_report
            report["planner"] = "heuristic_fallback"
    report.update(sql_report)
    return found, report


async def _aggregate_answer(
    db, llm, config, user, query, plan, planner, not_after, limit, agg_query, result
) -> tuple[list[Memory], dict]:
    """The SQL answer plus the memories behind it — never an empty list when something was counted."""
    cited = result.evidence + [row for row, _ in result.excluded]
    memories = _filter_by_not_after(await aggregation_to_memories(db, cited), not_after)
    by_id = {m.id: m for m in memories}
    # Counted instances first (in date order), then the ones left out.
    ordered = []
    for row in cited:
        m = by_id.pop(row.source_memory_id, None)
        if m is not None:
            ordered.append(m)

    expected = {row.source_memory_id for row in result.evidence}
    missing = expected - {m.id for m in ordered}
    if missing:
        log.warning(
            "aggregation counted instances whose memories are not retrievable (%s); "
            "adding ranked results so the answer is not empty", sorted(missing),
        )
        found, _ = await _dispatch_search(db, llm, config, user.id, plan, query, limit=limit)
        seen = {m.id for m in ordered}
        ordered += [m for m in _filter_by_not_after(found, not_after) if m.id not in seen]

    report = {
        "needs_memory": True,
        "planner": planner,
        "aggregation": True,
        "sql_result": result.value,
        "sql_count": result.count,
        "sql_subjects": result.subjects,
        "computed": computed_view(result, agg_query),
        "candidates": len(cited),
        "returned": len(ordered),
    }
    if missing:
        report["evidence_missing"] = sorted(missing)
    return ordered, report


async def search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    query: str,
    recent_chat: str = "",
    not_after: datetime | None = None,
) -> tuple[list[Memory], dict]:
    """Route, then search. Returns ``(memories, report)``.

    ``report["route"]`` is the path the router chose; ``report["planner"]``
    says which plan built the search (``"heuristic"`` = zero chat tokens).
    ``report["computed"]``, when present, is a deterministic answer with its
    evidence (see :func:`format_computed`).

    ``not_after``: cap retrieval to memories dated at or before this datetime.
    Pass the question date so future-dated memories don't pollute temporal questions.
    """
    user = await ensure_user(db, user_id, timezone=config.default_timezone)
    route = route_query(query, pregate=config.heuristic_pregate)

    if route == "chitchat":
        return [], {"needs_memory": False, "planner": "pregate", "route": route}

    fallback: dict = {}
    if route == "temporal":
        found, report = await _temporal(db, llm, config, user, query, not_after)
        if "computed" in report:
            report["route"] = route
            return await _dedup(db, config, found), report
        fallback = {k: report[k] for k in ("temporal", "anchors", "anchor_dates") if k in report}

    if route == "aggregate":
        found, report = await _aggregate(db, llm, config, user, query, recent_chat, not_after)
    else:
        limit = config.aggregation_recall_limit if route == "exploratory" else None
        found, report = await _lookup(db, llm, config, user, query, recent_chat, not_after, limit)

    report.setdefault("needs_memory", True)
    report["route"] = route
    report.update(fallback)
    if not report["needs_memory"]:
        return [], report
    return await _dedup(db, config, found), report


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

    A computed answer (count, sum, date difference) goes first, with what it
    counted and what it left out. The profile block follows
    ``config.profile_injection``: by default it is only included when recall
    found nothing (the profile usually restates what the memories already say).
    """
    user = await ensure_user(db, user_id, name=name, timezone=config.default_timezone)
    memories, report = await search(db, llm, config, user.id, message, recent_chat)

    inject_profile = (
        config.profile_injection == "always"
        or (config.profile_injection == "when_empty" and not memories)
    )
    profile_block = f"## About {user.name}\n{user.profile or '(no profile yet)'}\n\n" if inject_profile else ""
    computed = format_computed(report.get("computed"))
    computed_block = f"{computed}\n\n" if computed else ""
    context = (
        f"{profile_block}{computed_block}"
        f"## Memories that may be relevant to this message\n{format_memories(memories)}"
    )
    return context, memories
