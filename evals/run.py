#!/usr/bin/env python
"""Run riki-memory against our own personal-agent scenarios.

    python evals/run.py --dry-run                 # check the scenario files, no model calls
    python evals/run.py                           # default settings
    python evals/run.py --ablation                # each optional feature on/off, side by side
    python evals/run.py --variant cheap:planner_mode=heuristic,rerank=off
    python evals/run.py --only update-city,count-interviews --verbose

Each scenario is replayed through the public API only: ``remember()`` turn by
turn (with the session's real date), optional ``forget``/``delete`` actions,
then ``context()`` for every question. A fixed reader model answers from that
context alone and a fixed judge grades the answer, so variants differ only in
the memory layer. Secrets listed in ``must_not_store`` are checked
deterministically in storage, the profile and every context.

Uses longeval/.env (RIKI_* provider settings, RIKI_DB_URL). Reader and judge
default to RIKI_CHAT_MODEL; override with RIKI_READER_MODEL / RIKI_JUDGE_MODEL.
Every model call is cached on disk (longeval/cache), so re-runs and variants
that don't change writes replay for free.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "longeval"))

from harness import build_memory, env_get, load_env  # noqa: E402

from riki_memory import OpenAICompatibleProvider  # noqa: E402

UNKNOWN = "UNKNOWN"

ABLATION = {
    "default": {},
    "planner-llm": {"planner_mode": "llm"},
    "planner-heuristic": {"planner_mode": "heuristic"},
    "rerank-llm": {"rerank": "llm"},  # rerank is off by default; does turning it on earn its call?
    "profile-always": {"profile_injection": "always"},  # core memory in every context
    "agentic": {"retrieval_mode": "agentic"},
    "episodes": {"enable_episodes": True},
    "no-entities": {"enable_entities": False},
    "no-sql-aggregation": {"enable_sql_aggregation": False},
}

READER_PROMPT = """You are a personal assistant. Below is what you remember about the user.
Answer their question using only this memory. Be brief. If the memory does not contain
the answer, say you don't know; never guess or invent details.
Today is {today}.

MEMORY:
{context}"""

JUDGE_PROMPT = """You grade a personal assistant's answer against the expected answer.
- EXPECTED "UNKNOWN": correct only if the answer says it doesn't know or has no such
  information. Any specific claim (a name, a place, a value) is wrong.
- EXPECTED describes behaviour ("should not ..."): judge whether the answer behaves so.
- Otherwise: correct if the answer conveys the expected key facts. Extra harmless detail
  is fine; a missing key fact, a wrong value, or a contradiction is wrong.
Dates are equivalent in any format."""


class Verdict(BaseModel):
    correct: bool
    reason: str = ""


# Library model calls made while answering one question (planner, reranker, ...).
_read_calls: ContextVar[list | None] = ContextVar("read_calls", default=None)


class CountingProvider:
    """Counts the library's model calls during reads (cache hits included: they are
    calls the library asked for)."""

    def __init__(self, inner):
        self.inner = inner
        self.embedding_model_id = inner.embedding_model_id

    def _count(self):
        calls = _read_calls.get()
        if calls is not None:
            calls[0] += 1

    async def chat_json(self, system, user, schema, *, fast=True):
        self._count()
        return await self.inner.chat_json(system, user, schema, fast=fast)

    async def chat_text(self, system, user, *, fast=True):
        self._count()
        return await self.inner.chat_text(system, user, fast=fast)

    async def embed(self, text):
        return await self.inner.embed(text)

    async def close(self):
        await self.inner.close()


# -- scenarios -----------------------------------------------------------------


def load_scenarios(folder: Path, only: set[str] | None = None) -> list[dict]:
    scenarios = []
    for path in sorted(folder.glob("*.json")):
        for scenario in json.loads(path.read_text(encoding="utf-8")):
            scenario["_file"] = path.name
            scenarios.append(scenario)
    problems = []
    seen = set()
    for s in scenarios:
        sid = s.get("id")
        if not sid or sid in seen:
            problems.append(f"{s['_file']}: missing or duplicate id {sid!r}")
        seen.add(sid)
        for key in ("category", "sessions", "questions"):
            if not s.get(key):
                problems.append(f"{sid}: missing {key}")
        for session in s.get("sessions", []):
            if not isinstance(session.get("days_ago"), int) or not session.get("turns"):
                problems.append(f"{sid}: every session needs an int days_ago and turns")
            for turn in session.get("turns", []):
                if not (isinstance(turn, list) and len(turn) == 2):
                    problems.append(f"{sid}: a turn must be [user message, assistant reply]")
        for q in s.get("questions", []):
            if not q.get("q") or not q.get("expected"):
                problems.append(f"{sid}: every question needs q and expected")
        for action in s.get("actions", []):
            if set(action) - {"forget", "delete"}:
                problems.append(f"{sid}: unknown action {action}")
    if problems:
        raise SystemExit("scenario problems:\n  " + "\n  ".join(problems))
    if only:
        missing = only - seen
        if missing:
            raise SystemExit(f"unknown scenario ids: {', '.join(sorted(missing))}")
        scenarios = [s for s in scenarios if s["id"] in only]
    return scenarios


def render(expected: str, now: datetime) -> str:
    """``{d:N}`` -> the date N days before now, e.g. "September 10, 2026"."""
    out = expected
    while "{d:" in out:
        start = out.index("{d:")
        end = out.index("}", start)
        day = now - timedelta(days=int(out[start + 3:end]))
        out = out[:start] + f"{day:%B} {day.day}, {day.year}" + out[end + 1:]
    return out


# -- running -------------------------------------------------------------------


@dataclass
class QuestionResult:
    scenario: str
    category: str
    question: str
    expected: str
    answer: str = ""
    correct: bool = False
    reason: str = ""
    context_chars: int = 0
    read_model_calls: int = 0
    latency_ms: float = 0.0


@dataclass
class ScenarioResult:
    scenario: str
    questions: list[QuestionResult] = field(default_factory=list)
    privacy_violations: list[str] = field(default_factory=list)
    unmatched_actions: list[str] = field(default_factory=list)


def namespace(variant: str, scenario: str, which: str = "main") -> str:
    base = f"scn/{variant}/{scenario}"
    return base if which == "main" else f"{base}/{which}"


async def run_scenario(memory, reader, judge, variant: str, scenario: dict, now: datetime) -> ScenarioResult:
    sid = scenario["id"]
    result = ScenarioResult(sid)
    namespaces = {namespace(variant, sid, s.get("namespace", "main")) for s in scenario["sessions"]}
    namespaces.add(namespace(variant, sid))
    for ns in namespaces:
        await memory.delete_user(ns)

    for index, session in enumerate(sorted(scenario["sessions"], key=lambda s: -s["days_ago"])):
        user = namespace(variant, sid, session.get("namespace", "main"))
        at = (now - timedelta(days=session["days_ago"])).replace(hour=18, minute=0, second=0, microsecond=0)
        for turn_index, (message, reply) in enumerate(session["turns"]):
            await memory.remember(user, message, reply, turn_key=f"{user}/s{index}/t{turn_index}", at=at)
        if memory.config.enable_episodes:
            await memory.close_session(user, f"{user}/s{index}", date=at)

    main = namespace(variant, sid)
    for action in scenario.get("actions", []):
        verb, needle = next(iter(action.items()))
        matches = [m for m in await memory.list_memories(main, limit=1000) if needle.lower() in m.content.lower()]
        if not matches:
            result.unmatched_actions.append(f"{verb} {needle!r}: no memory mentions it")
        for m in matches:
            await (memory.forget if verb == "forget" else memory.delete)(main, m.id)

    secrets = scenario.get("must_not_store", [])
    for q in scenario["questions"]:
        expected = render(q["expected"], now)
        calls = [0]
        token = _read_calls.set(calls)
        started = time.perf_counter()
        try:
            context, _ = await memory.context(main, q["q"])
        finally:
            _read_calls.reset(token)
        latency = (time.perf_counter() - started) * 1000
        answer = await reader.chat_text(
            READER_PROMPT.format(today=f"{now:%B} {now.day}, {now.year}", context=context or "(nothing)"),
            q["q"], fast=False,
        )
        verdict = await judge.chat_json(
            JUDGE_PROMPT, f"QUESTION: {q['q']}\nEXPECTED: {expected}\nANSWER: {answer}", Verdict, fast=False,
        )
        result.questions.append(QuestionResult(
            sid, scenario["category"], q["q"], expected, answer, verdict.correct, verdict.reason,
            len(context or ""), calls[0], round(latency, 1),
        ))
        for secret in secrets:
            if secret.lower() in (context or "").lower() or secret.lower() in answer.lower():
                result.privacy_violations.append(f"{secret!r} appeared answering {q['q']!r}")

    if secrets:
        stored = [m.content for ns in namespaces for m in await memory.list_memories(ns, active_only=False, limit=1000)]
        profiles = [n.profile for n in await memory.list_namespaces(limit=1000) if n.id in namespaces]
        from sqlalchemy import select

        from riki_memory.models import Turn

        async with memory.db.session_scope() as session:  # the raw turn log counts as stored too
            raw = [f"{t.user_text}\n{t.reply_text}" for t in (await session.scalars(
                select(Turn).where(Turn.user_id.in_(namespaces)))).all()]
        for secret in secrets:
            if any(secret.lower() in text.lower() for text in stored + profiles + raw):
                result.privacy_violations.append(f"{secret!r} is stored")
    return result


async def run_variant(env, name: str, overrides: dict, scenarios: list[dict], concurrency: int, reader, judge):
    memory = build_memory(env, **overrides)
    memory.llm = CountingProvider(memory.llm)
    await memory.init()
    now = datetime.now(timezone.utc)
    gate = asyncio.Semaphore(concurrency)

    async def one(scenario):
        async with gate:
            return await run_scenario(memory, reader, judge, name, scenario, now)

    try:
        results = await asyncio.gather(*(one(s) for s in scenarios))
        stats = memory.write_stats.as_dict()
    finally:
        await memory.close()
    return results, stats


def summarise(name: str, overrides: dict, results: list[ScenarioResult], writes: dict) -> dict:
    questions = [q for r in results for q in r.questions]
    by_category: dict[str, list[QuestionResult]] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)
    n = len(questions) or 1
    return {
        "variant": name,
        "overrides": overrides,
        "accuracy": round(sum(q.correct for q in questions) / n, 3),
        "correct": sum(q.correct for q in questions),
        "questions": len(questions),
        "by_category": {c: f"{sum(q.correct for q in qs)}/{len(qs)}" for c, qs in sorted(by_category.items())},
        "privacy_violations": [v for r in results for v in r.privacy_violations],
        "unmatched_actions": [a for r in results for a in r.unmatched_actions],
        "read_model_calls_per_question": round(sum(q.read_model_calls for q in questions) / n, 2),
        "context_chars_per_question": round(sum(q.context_chars for q in questions) / n),
        "write": {k: writes[k] for k in (
            "writes", "failed_writes", "memories_saved", "dropped_invalid", "model_calls", "embed_calls",
            "prompt_tokens", "cached_tokens", "completion_tokens", "skipped_writes",
        )},
        "details": [q.__dict__ for q in questions],
    }


def print_report(summaries: list[dict], verbose: bool) -> None:
    categories = sorted({c for s in summaries for c in s["by_category"]})
    print()
    header = f"{'variant':20} {'score':>7}  {'read calls/q':>12} {'context/q':>10}  " + "  ".join(f"{c[:10]:>10}" for c in categories)
    print(header)
    print("-" * len(header))
    for s in summaries:
        cells = "  ".join(f"{s['by_category'].get(c, '-'):>10}" for c in categories)
        print(f"{s['variant']:20} {s['correct']:>3}/{s['questions']:<3}  {s['read_model_calls_per_question']:>12} "
              f"{s['context_chars_per_question']:>10}  {cells}")
    for s in summaries:
        problems = s["privacy_violations"] + s["unmatched_actions"]
        if s["write"]["failed_writes"]:
            problems.append(f"{s['write']['failed_writes']} writes failed")
        if problems:
            print(f"\n[{s['variant']}] " + "\n  ".join(["problems:"] + problems))
        if verbose:
            for q in s["details"]:
                if not q["correct"]:
                    print(f"\n[{s['variant']}] MISS {q['scenario']}: {q['question']}\n"
                          f"  expected: {q['expected']}\n  answer:   {q['answer']}\n  judge:    {q['reason']}")


def parse_variant(text: str) -> tuple[str, dict]:
    name, _, body = text.partition(":")
    overrides = {}
    for pair in filter(None, body.split(",")):
        key, _, value = pair.partition("=")
        overrides[key.strip()] = {"true": True, "false": False}.get(value.strip().lower(), value.strip())
    return name, overrides


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default=str(ROOT / "evals" / "scenarios"))
    parser.add_argument("--env", default=str(ROOT / "longeval" / ".env"))
    parser.add_argument("--only", default="", help="Comma-separated scenario ids.")
    parser.add_argument("--variant", action="append", default=[],
                        help="name:key=value,key=value (MemoryConfig overrides). Repeatable.")
    parser.add_argument("--ablation", action="store_true", help="Run every preset in ABLATION.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--save", default=None, help="Write full results (answers, verdicts) to this JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate scenarios and show the plan; no model calls.")
    parser.add_argument("--verbose", action="store_true", help="Show every missed question.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    scenarios = load_scenarios(Path(args.scenarios), set(filter(None, args.only.split(","))) or None)
    variants = dict(ABLATION) if args.ablation else {}
    variants.update(parse_variant(v) for v in args.variant)
    variants = variants or {"default": {}}

    if args.dry_run:
        now = datetime.now(timezone.utc)
        questions = sum(len(s["questions"]) for s in scenarios)
        turns = sum(len(x["turns"]) for s in scenarios for x in s["sessions"])
        print(f"{len(scenarios)} scenarios, {turns} turns, {questions} questions; variants: {', '.join(variants)}")
        for s in scenarios:
            for q in s["questions"]:
                print(f"  [{s['category']}] {s['id']}: {q['q']}  ->  {render(q['expected'], now)}")
        return

    env = load_env(args.env)
    from cache import CachingProvider

    def fixed_model(var: str):
        model = env_get(env, var) or env_get(env, "RIKI_CHAT_MODEL", "gpt-4o-mini")
        provider = OpenAICompatibleProvider(
            api_key=env_get(env, "RIKI_API_KEY"), base_url=env_get(env, "RIKI_BASE_URL", "https://api.openai.com/v1"),
            chat_model=model, fast_model=model,
            embedding_model=env_get(env, "RIKI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        return CachingProvider(provider, env_get(env, "RIKI_LLM_CACHE", str(ROOT / "longeval" / "cache")),
                               chat_model=model, fast_model=model)

    reader, judge = fixed_model("RIKI_READER_MODEL"), fixed_model("RIKI_JUDGE_MODEL")
    summaries = []
    try:
        for name, overrides in variants.items():
            print(f"running {name} {overrides or ''} ...", flush=True)
            results, writes = await run_variant(env, name, overrides, scenarios, args.concurrency, reader, judge)
            summaries.append(summarise(name, overrides, results, writes))
    finally:
        await reader.close()
        await judge.close()

    print_report(summaries, args.verbose)
    if args.save:
        Path(args.save).write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nfull results: {args.save}")


if __name__ == "__main__":
    asyncio.run(main())
