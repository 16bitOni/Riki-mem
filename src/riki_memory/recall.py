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

from .config import MemoryConfig
from .db import Database
from .heuristic import make_heuristic_plan
from .llm import LLMProvider
from .models import Memory
from .retrieval import hybrid_search
from .schemas import RetrievalPlan
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


async def search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    query: str,
    recent_chat: str = "",
) -> tuple[list[Memory], dict]:
    """Plan + hybrid search. Returns ``(memories, report)``.

    ``report["planner"]`` says which plan built the query: ``"heuristic"``
    (zero chat tokens) or ``"llm"``.
    """
    user = await ensure_user(db, user_id, timezone=config.default_timezone)
    mode = config.planner_mode

    if mode == "heuristic":
        plan = make_heuristic_plan(
            query, user.timezone, pregated=bool(config.heuristic_pregate and _PREGATE.match(query or ""))
        )
        planner = "heuristic"
    elif mode == "escalate":
        plan = make_heuristic_plan(
            query, user.timezone, pregated=bool(config.heuristic_pregate and _PREGATE.match(query or ""))
        )
        if plan.needs_memory:
            found, report = await hybrid_search(db, llm, config, user.id, plan, rerank_query=query)
            if report["candidates"]:
                report["needs_memory"] = True
                report["planner"] = "heuristic"
                return found, report
            log.debug("heuristic plan found nothing; escalating to the LLM planner")
        plan = await make_plan(llm, config, query, recent_chat, user.timezone)
        planner = "llm"
    else:  # "llm"
        plan = await make_plan(llm, config, query, recent_chat, user.timezone)
        planner = "llm"

    if not plan.needs_memory:
        return [], {"needs_memory": False, "planner": planner}
    found, report = await hybrid_search(db, llm, config, user.id, plan, rerank_query=query)
    report["needs_memory"] = True
    report["planner"] = planner
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
