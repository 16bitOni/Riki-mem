"""LongMemEval benchmark harness for riki-memory.

Handles dataset downloading/parsing, session ingestion, querying,
scoring (substring + LLM judge), and aggregate reporting.

Dataset: https://huggingface.co/datasets/xiaowu0162/LongMemEval
Paper:   "LongMemEval: Benchmarking Chat Assistants on Long-Term
          Interactive Memory" (ICLR 2025)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

_AGGREGATION_JUDGE_RE = re.compile(
    r"\b(how many|how much|total|average|sum|count|how long|how far|how old)\b",
    re.IGNORECASE,
)
_TEMPORAL_CALC_JUDGE_RE = re.compile(
    r"\bhow (?:many|much|long|far|old)\s+(?:days?|months?|years?|hours?|minutes?|weeks?)\s+"
    r"(?:before|after|since|ago|until|between|apart|did it take|does it take|will it take)",
    re.IGNORECASE,
)


def _is_aggregation_judge(question: str) -> bool:
    if not _AGGREGATION_JUDGE_RE.search(question):
        return False
    if _TEMPORAL_CALC_JUDGE_RE.search(question):
        return False
    return True

import httpx

log = logging.getLogger("longeval")

# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    date: str
    turns: list[Turn]


@dataclass
class Question:
    question_id: str
    question: str
    answers: list[str]
    category: str
    question_date: str
    evidence_session_ids: list[str]


@dataclass
class Dataset:
    sessions: dict[str, Session]
    questions: list[Question]


@dataclass
class QuestionResult:
    question_id: str
    category: str
    question: str
    expected_answers: list[str]
    retrieved_contents: list[str]
    hit: bool
    reciprocal_rank: float

# ---------------------------------------------------------------------------
# .env loader (no external deps)
# ---------------------------------------------------------------------------

def load_env(env_path: str | Path) -> dict[str, str]:
    env: dict[str, str] = {}
    p = Path(env_path)
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip("\"'")
    return env


def env_get(env: dict[str, str], key: str, default: str = "") -> str:
    return env.get(key) or os.environ.get(key, default)

# ---------------------------------------------------------------------------
# Provider + RikiMemory builder
# ---------------------------------------------------------------------------

def build_memory(env: dict[str, str], **config_overrides):
    from riki_memory import RikiMemory, OpenAICompatibleProvider, MemoryConfig

    api_key = env_get(env, "RIKI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise SystemExit(
            "RIKI_API_KEY not configured. Copy longeval/.env.example to "
            "longeval/.env and fill in your API key."
        )

    llm = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=env_get(env, "RIKI_BASE_URL", "https://api.openai.com/v1"),
        chat_model=env_get(env, "RIKI_CHAT_MODEL", "gpt-4o-mini"),
        fast_model=env_get(env, "RIKI_FAST_MODEL", "gpt-4o-mini"),
        embedding_model=env_get(env, "RIKI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    db_url = env_get(env, "RIKI_DB_URL", "sqlite+aiosqlite:///longeval/eval.db")

    return RikiMemory(database_url=db_url, llm=llm, config=MemoryConfig(**config_overrides))

# ---------------------------------------------------------------------------
# Dataset downloading
# ---------------------------------------------------------------------------

DATASET_REPO = "xiaowu0162/longmemeval-cleaned"
HF_RAW_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{filename}"

SPLIT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


async def list_hf_splits() -> dict[str, str]:
    """Return {split_name: filename} for available splits."""
    return dict(SPLIT_FILES)


async def download_dataset(
    split: str = "oracle",
    hf_token: str = "",
    cache_dir: str = "longeval/data",
) -> Path:
    """Download a raw JSON file from HuggingFace and cache it locally."""
    if split not in SPLIT_FILES:
        raise SystemExit(
            f"Unknown split '{split}'. Choose from: {', '.join(SPLIT_FILES)}.\n"
            f"  oracle  — evidence sessions only, ~15 MB (fastest)\n"
            f"  s       — ~40 sessions per question, ~277 MB\n"
            f"  m       — ~500 sessions per question, ~2.7 GB"
        )

    filename = SPLIT_FILES[split]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / filename

    if cached.exists():
        log.info("using cached dataset: %s", cached)
        return cached

    url = HF_RAW_URL.format(repo=DATASET_REPO, filename=filename)
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    log.info("downloading %s (%s) ...", filename, url)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=600, write=30, pool=30),
        headers=headers,
        follow_redirects=True,
    ) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(cached, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  downloading: {pct}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)
            print()

    log.info("cached to %s", cached)
    return cached

# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------

def _parse_turns(raw_turns: list) -> list[Turn]:
    turns = []
    for t in raw_turns:
        if isinstance(t, dict):
            turns.append(Turn(role=t.get("role", "user"), content=t.get("content", "")))
        elif isinstance(t, str):
            turns.append(Turn(role="user", content=t))
    return turns


def parse_dataset(raw: list[dict], limit: int | None = None) -> Dataset:
    """Parse raw LongMemEval rows into our internal format.

    Actual schema per row:
      question_id, question_type, question, answer (str|int|list),
      question_date, haystack_dates (list[str]),
      haystack_session_ids (list[str]),
      haystack_sessions (list[list[{role, content, has_answer}]])
    """
    sessions: dict[str, Session] = {}
    questions: list[Question] = []

    items = raw[:limit] if limit else raw

    for i, row in enumerate(items):
        qid = str(row.get("question_id") or row.get("id") or i)
        q_text = row.get("question", "")

        ans = row.get("answer") or row.get("answers") or row.get("gold_answer") or []
        if isinstance(ans, (str, int, float)):
            ans = [str(ans)]
        else:
            ans = [str(a) for a in ans]

        category = row.get("question_type") or row.get("category") or "unknown"
        q_date = row.get("question_date") or ""

        session_ids = row.get("haystack_session_ids") or []
        session_dates = row.get("haystack_dates") or []
        haystack = row.get("haystack_sessions") or []

        if isinstance(haystack, str):
            haystack = json.loads(haystack)

        for idx, sess_turns in enumerate(haystack):
            sid = str(session_ids[idx]) if idx < len(session_ids) else f"{qid}_s{idx}"
            if sid in sessions:
                continue
            s_date = session_dates[idx] if idx < len(session_dates) else ""

            if isinstance(sess_turns, list):
                turns = _parse_turns(sess_turns)
            elif isinstance(sess_turns, dict):
                raw_turns = (
                    sess_turns.get("conversation")
                    or sess_turns.get("turns")
                    or sess_turns.get("messages")
                    or []
                )
                turns = _parse_turns(raw_turns)
                sid = str(sess_turns.get("session_id") or sess_turns.get("id") or sid)
                s_date = s_date or sess_turns.get("session_date") or sess_turns.get("date") or ""
            else:
                continue

            sessions[sid] = Session(session_id=sid, date=s_date, turns=turns)

        evidence_ids = (
            row.get("answer_session_ids")
            or row.get("evidence_session_ids")
            or row.get("gold_session_ids")
            or session_ids
        )
        if isinstance(evidence_ids, str):
            evidence_ids = json.loads(evidence_ids) if evidence_ids.startswith("[") else [evidence_ids]

        questions.append(Question(
            question_id=qid,
            question=q_text,
            answers=ans,
            category=category,
            question_date=q_date,
            evidence_session_ids=[str(x) for x in evidence_ids],
        ))

    return Dataset(sessions=sessions, questions=questions)

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

ProgressFn = Callable[[int, int, str], None]


def _parse_session_date(date_str: str) -> datetime | None:
    """Parse a session date string like '2023/05/28 (Sun) 21:04' into a timezone-aware datetime."""
    if not date_str:
        return None
    import re
    # Strip day-of-week parenthetical: '2023/05/28 (Sun) 21:04' -> '2023/05/28 21:04'
    cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=ZoneInfo("UTC"))
        except ValueError:
            continue
    return None


async def ingest_sessions(
    memory,
    user_id: str,
    sessions: dict[str, Session],
    progress_fn: ProgressFn | None = None,
) -> dict:
    """Feed conversation sessions into riki-memory via remember()."""
    total_pairs = sum(len(s.turns) // 2 for s in sessions.values())
    ingested = 0
    errors = 0
    t0 = time.time()

    episodes_created = 0
    for sid, session in sessions.items():
        session_at = _parse_session_date(session.date)
        for i in range(0, len(session.turns) - 1, 2):
            user_turn = session.turns[i]
            asst_turn = session.turns[i + 1]

            if user_turn.role != "user":
                continue

            turn_key = f"longeval/{sid}/{i}"
            try:
                await memory.remember(
                    user_id=user_id,
                    user_message=user_turn.content,
                    reply=asst_turn.content,
                    turn_key=turn_key,
                    at=session_at,
                )
                ingested += 1
            except Exception as e:
                log.warning("ingest error session=%s turn=%d: %s", sid, i, e)
                errors += 1

            if progress_fn:
                progress_fn(ingested + errors, total_pairs, sid)

        try:
            ep = await memory.close_session(
                user_id=user_id,
                session_key=f"longeval/{sid}",
                date=session_at,
            )
            if ep:
                episodes_created += 1
        except Exception as e:
            log.warning("close_session error session=%s: %s", sid, e)

    return {
        "ingested": ingested,
        "errors": errors,
        "sessions": len(sessions),
        "episodes": episodes_created,
        "elapsed_s": round(time.time() - t0, 1),
    }

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SCORE_STOP = frozenset("a an the is are was were of in on at to for and or by".split())


def _answer_words(answer: str) -> set[str]:
    """Significant words from an expected answer (lowercase, no punctuation, no stopwords)."""
    import re
    tokens = re.findall(r"[a-z0-9]+", answer.lower())
    return {t for t in tokens if len(t) >= 2 and t not in _SCORE_STOP}


def score_substring(answers: list[str], retrieved: list[str]) -> tuple[bool, float]:
    for rank, content in enumerate(retrieved, start=1):
        low = content.lower()
        for a in answers:
            if a.lower() in low:
                return True, 1.0 / rank
            words = _answer_words(a)
            if words and words.issubset(_answer_words(content)):
                return True, 1.0 / rank
    return False, 0.0


async def score_llm_judge(
    llm, question: str, expected: list[str], retrieved: list[str],
    question_date: str = "",
) -> tuple[bool, float]:
    if not retrieved:
        return False, 0.0

    context = "\n".join(f"[{i+1}] {m}" for i, m in enumerate(retrieved[:30]))
    expected_str = " OR ".join(expected)

    date_hint = (
        f"The question was asked on {question_date}. "
        "Use this as 'today' / 'now' for any relative time calculations "
        "('X months ago', 'last week', etc.).\n"
    ) if question_date else ""

    is_aggregation = _is_aggregation_judge(question)

    if is_aggregation:
        system = (
            "You are an evaluation judge for a personal memory system.\n"
            f"{date_hint}"
            "For counting / aggregation questions use this TWO-STEP process:\n\n"
            "STEP 1 — DEDUPLICATE & EXTRACT\n"
            "Read every numbered memory. List only the DISTINCT, SPECIFIC items relevant "
            "to the question (one per line). Skip duplicates (same event mentioned multiple times), "
            "preferences, plans, interests, and anything not directly answering the question.\n"
            "Format each as: • <item name or description> [<value/amount if any>]\n\n"
            "STEP 2 — COMPUTE & JUDGE\n"
            "Count the distinct items / sum the values / do any date math needed.\n"
            "Compare your answer to the expected answer (semantic equivalence is fine).\n\n"
            "Reply in EXACTLY this format:\n"
            "Items:\n"
            "• <item 1>\n"
            "• <item 2>\n"
            "...\n"
            "Answer: <your computed answer>\n"
            "Match: yes OR Match: no"
        )
    else:
        system = (
            "You are an evaluation judge. Given a question and retrieved memory context:\n"
            f"{date_hint}"
            "1. Answer the question directly and concisely based ONLY on the retrieved context. "
            "Do any counting, date math, or reasoning needed.\n"
            "2. Compare your answer to the expected answer.\n"
            "3. Determine if they are semantically equivalent.\n\n"
            "Reply in EXACTLY this format:\n"
            "Answer: <your direct concise answer>\n"
            "Match: yes OR Match: no"
        )

    user = (
        f"Question: {question}\n"
        f"Expected answer: {expected_str}\n\n"
        f"Retrieved memories:\n{context}\n\n"
        f"Answer the question from the memories, then say whether it matches the expected answer."
    )

    try:
        reply = await llm.chat_text(system, user, fast=True)
        hit = "match: yes" in reply.strip().lower()
        return hit, 1.0 if hit else 0.0
    except Exception as e:
        log.warning("LLM judge error, falling back to substring: %s", e)
        return score_substring(expected, retrieved)

# ---------------------------------------------------------------------------
# Query + evaluate
# ---------------------------------------------------------------------------

async def run_questions(
    memory,
    user_id: str,
    questions: list[Question],
    judge: str = "substring",
    progress_fn: ProgressFn | None = None,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []

    for idx, q in enumerate(questions):
        # Parse question_date into a datetime upper bound so retrieval excludes
        # memories that post-date when the question was actually asked.
        not_after: datetime | None = None
        if q.question_date:
            import re as _re
            from datetime import timezone as _tz
            # Dataset format: "2023/02/20 (Mon) 22:35" — strip the day-of-week token
            # then try a few common formats.
            clean = _re.sub(r"\s*\([A-Za-z]+\)\s*", " ", q.question_date).strip()
            for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    not_after = datetime.strptime(clean[:len(fmt) + 2], fmt).replace(tzinfo=_tz.utc)
                    break
                except ValueError:
                    continue

        try:
            found, _report = await memory.recall(
                user_id=user_id, query=q.question, not_after=not_after,
            )
            retrieved = [m.content for m in found]
        except Exception as e:
            log.warning("recall error q=%s: %s", q.question_id, e)
            retrieved = []

        if judge == "llm":
            hit, rr = await score_llm_judge(
                memory.llm, q.question, q.answers, retrieved, question_date=q.question_date,
            )
        elif judge == "combined":
            hit, rr = score_substring(q.answers, retrieved)
            if not hit:
                hit, rr = await score_llm_judge(
                    memory.llm, q.question, q.answers, retrieved, question_date=q.question_date,
                )
        else:
            hit, rr = score_substring(q.answers, retrieved)

        results.append(QuestionResult(
            question_id=q.question_id,
            category=q.category,
            question=q.question,
            expected_answers=q.answers,
            retrieved_contents=retrieved[:5],
            hit=hit,
            reciprocal_rank=rr,
        ))

        if progress_fn:
            progress_fn(idx + 1, len(questions), q.category)

    return results

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[QuestionResult]) -> dict:
    categories: dict[str, list[QuestionResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    def metrics(items: list[QuestionResult]) -> dict:
        n = len(items) or 1
        return {
            "count": len(items),
            "hit_rate": round(sum(1 for r in items if r.hit) / n, 3),
            "mrr": round(sum(r.reciprocal_rank for r in items) / n, 3),
        }

    report = {"overall": metrics(results), "by_category": {}}
    for cat in sorted(categories):
        report["by_category"][cat] = metrics(categories[cat])
    return report


def format_report(report: dict, results: list[QuestionResult], verbose: bool = False) -> str:
    lines: list[str] = []
    overall = report["overall"]
    lines.append("=" * 60)
    lines.append("  LongMemEval Results")
    lines.append("=" * 60)
    lines.append(f"  Questions : {overall['count']}")
    lines.append(f"  Hit Rate  : {overall['hit_rate']:.1%}")
    lines.append(f"  MRR       : {overall['mrr']:.3f}")
    lines.append("-" * 60)

    for cat, m in report["by_category"].items():
        lines.append(f"  {cat:<40s}  n={m['count']:>3d}  hit={m['hit_rate']:.1%}  mrr={m['mrr']:.3f}")

    if verbose:
        lines.append("")
        lines.append("-" * 60)
        lines.append("  Misses:")
        lines.append("-" * 60)
        for r in results:
            if not r.hit:
                lines.append(f"  [{r.category}] {r.question}")
                lines.append(f"    expected: {r.expected_answers}")
                if r.retrieved_contents:
                    lines.append(f"    got:      {r.retrieved_contents[0][:120]}...")
                else:
                    lines.append("    got:      (nothing retrieved)")
                lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
