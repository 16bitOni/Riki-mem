"""LongMemEval benchmark harness for riki-memory.

Handles dataset downloading/parsing, session ingestion, querying,
scoring (substring + LLM judge), and aggregate reporting.

Dataset: https://huggingface.co/datasets/xiaowu0162/LongMemEval
Paper:   "LongMemEval: Benchmarking Chat Assistants on Long-Term
          Interactive Memory" (ICLR 2025)
"""

import asyncio
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
    # This question's own history (its haystack), in dataset order.
    session_ids: list[str] = field(default_factory=list)


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
    all_retrieved_contents: list[str] = field(default_factory=list)
    # QA mode (fixed reader + official judge)
    answer: str = ""
    context_tokens: int = 0  # what the memory layer handed the reader (~4 characters per token)
    read_model_calls: int = 0  # model calls inside the memory layer while answering
    recall_ms: float = 0.0
    abstention: bool = False

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

    cache_dir = env_get(env, "RIKI_LLM_CACHE", "longeval/cache")
    if cache_dir and cache_dir.lower() != "off":
        from cache import CachingProvider

        llm = CachingProvider(
            llm, cache_dir,
            chat_model=env_get(env, "RIKI_CHAT_MODEL", "gpt-4o-mini"),
            fast_model=env_get(env, "RIKI_FAST_MODEL", "gpt-4o-mini"),
        )

    db_url = env_get(env, "RIKI_DB_URL", "postgresql://riki:riki@localhost:5433/riki")

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

        own_sessions: list[str] = []
        for idx, sess_turns in enumerate(haystack):
            sid = str(session_ids[idx]) if idx < len(session_ids) else f"{qid}_s{idx}"
            if sid in sessions:
                own_sessions.append(sid)
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
            own_sessions.append(sid)

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
            session_ids=own_sessions,
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
    """Feed conversation sessions into riki-memory via remember(), one turn at a time."""
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

        if not memory.config.enable_episodes:
            continue  # a summary nobody reads is a paid call per session
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

    window = retrieved if _is_aggregation_judge(question) else retrieved[:30]
    context = "\n".join(f"[{i+1}] {m}" for i, m in enumerate(window))
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
# QA mode: memory context -> fixed reader -> official judge
# ---------------------------------------------------------------------------

# LongMemEval's own reader prompt for a system that returns facts extracted from chats
# (src/generation/run_generation.py, the step-by-step variant), so every memory system
# is read the same way.
READER_PROMPT = (
    "I will give you several facts extracted from history chats between you and a user. "
    "Please answer the question based on the relevant facts. Answer the question step by step: "
    "first extract all the relevant information, and then reason over the information to get the answer."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"
)

# LongMemEval's judge prompts, verbatim from src/evaluation/evaluate_qa.py.
_JUDGE_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
    "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
    "the correct answer or contains all the intermediate steps to get the correct answer, you should "
    "also answer yes. If the response only contains a subset of the information required by the "
    "answer, answer no."
)
JUDGE_PROMPTS = {
    "default": _JUDGE_BASE + " \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "temporal-reasoning": _JUDGE_BASE + " In addition, do not penalize off-by-one errors for the number "
    "of days. If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still "
    "correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response contains some previous information along with an updated answer, the response should be "
    "considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\n"
    "Correct Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized "
    "response, and a response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal information "
    "correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as unanswerable. The model "
    "could say that the information is incomplete, or some other information is given but the asked "
    "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only.",
}
OFFICIAL_JUDGE_MODEL = "zai.glm-5"


def judge_prompt(category: str, question_id: str, question: str, answer: str, response: str) -> str:
    key = "abstention" if "_abs" in question_id else (category if category in JUDGE_PROMPTS else "default")
    return JUDGE_PROMPTS[key].format(question, answer, response)


@dataclass
class QAClients:
    """The fixed reader and judge, the same for every memory system under test."""
    reader: object  # OpenAICompatibleProvider
    reader_model: str
    judge: object
    judge_model: str
    cache: object | None = None  # a CachingProvider, so re-runs replay answers and verdicts

    async def ask(self, provider, model: str, prompt: str, max_tokens: int | None = None) -> str:
        from cache import ask

        if self.cache is not None:
            return await self.cache.ask(provider, model, prompt, max_tokens)
        return await ask(provider, model, prompt, max_tokens)


def build_qa_clients(env: dict[str, str], cache=None) -> QAClients:
    """Reader: READER_MODEL (default RIKI_CHAT_MODEL). Judge: JUDGE_MODEL (default the official
    gpt-4o-2024-08-06). Each can have its own JUDGE_/READER_ BASE_URL and API_KEY."""
    from riki_memory import OpenAICompatibleProvider

    def client(prefix: str):
        return OpenAICompatibleProvider(
            api_key=env_get(env, f"{prefix}_API_KEY") or env_get(env, "RIKI_API_KEY"),
            base_url=env_get(env, f"{prefix}_BASE_URL") or env_get(env, "RIKI_BASE_URL", "https://api.openai.com/v1"),
        )

    return QAClients(
        reader=client("READER"), reader_model=env_get(env, "READER_MODEL") or env_get(env, "RIKI_CHAT_MODEL", "gpt-4o-mini"),
        judge=client("JUDGE"), judge_model=env_get(env, "JUDGE_MODEL") or OFFICIAL_JUDGE_MODEL,
        cache=cache,
    )


async def answer_and_judge(memory, user_id: str, q: "Question", not_after, qa: QAClients) -> dict:
    """context() -> reader -> judge. Costs and latency of the memory layer are measured
    around context() only; the reader and judge are the benchmark's, not the system's."""
    from riki_memory import telemetry

    t0 = time.perf_counter()
    async with telemetry.tracking(user_id, "read") as report:
        context, found = await memory.context(user_id, q.question, at=not_after)
    recall_ms = (time.perf_counter() - t0) * 1000
    answer = await qa.ask(qa.reader, qa.reader_model, READER_PROMPT.format(context, q.question_date, q.question))
    verdict = await qa.ask(
        qa.judge, qa.judge_model,
        judge_prompt(q.category, q.question_id, q.question, " OR ".join(q.answers), answer), max_tokens=10,
    )
    return {
        "hit": "yes" in verdict.lower(), "answer": answer, "found": [m.content for m in found],
        "context_tokens": len(context) // 4, "read_model_calls": report.model_calls, "recall_ms": recall_ms,
    }


# ---------------------------------------------------------------------------
# Query + evaluate
# ---------------------------------------------------------------------------

async def score_question(
    memory, user_id: str, q: Question, judge: str = "substring", qa: QAClients | None = None,
) -> QuestionResult:
    """Ask one question of its own user's memory and score the answer."""
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

    if judge == "qa":
        try:
            out = await answer_and_judge(memory, user_id, q, not_after, qa)
        except Exception as e:
            log.warning("qa error q=%s: %s", q.question_id, e)
            out = {"hit": False, "answer": f"(error: {e})", "found": [], "context_tokens": 0,
                   "read_model_calls": 0, "recall_ms": 0.0}
        return QuestionResult(
            question_id=q.question_id, category=q.category, question=q.question,
            expected_answers=q.answers, retrieved_contents=out["found"][:5], hit=out["hit"],
            reciprocal_rank=1.0 if out["hit"] else 0.0, all_retrieved_contents=out["found"],
            answer=out["answer"], context_tokens=out["context_tokens"],
            read_model_calls=out["read_model_calls"], recall_ms=out["recall_ms"],
            abstention="_abs" in q.question_id,
        )

    try:
        found, _report = await memory.recall(
            user_id=user_id, query=q.question, not_after=not_after,
        )
        all_retrieved = [m.content for m in found]
    except Exception as e:
        log.warning("recall error q=%s: %s", q.question_id, e)
        all_retrieved = []

    if judge == "llm":
        hit, rr = await score_llm_judge(
            memory.llm, q.question, q.answers, all_retrieved, question_date=q.question_date,
        )
    elif judge == "combined":
        hit, rr = score_substring(q.answers, all_retrieved)
        if not hit:
            hit, rr = await score_llm_judge(
                memory.llm, q.question, q.answers, all_retrieved, question_date=q.question_date,
            )
    else:
        hit, rr = score_substring(q.answers, all_retrieved)

    return QuestionResult(
        question_id=q.question_id,
        category=q.category,
        question=q.question,
        expected_answers=q.answers,
        retrieved_contents=all_retrieved[:5],
        hit=hit,
        reciprocal_rank=rr,
        all_retrieved_contents=all_retrieved,
    )


def question_user(prefix: str, q: Question) -> str:
    """Each question gets its own memory: LongMemEval gives every question its own history."""
    return f"{prefix}:{q.question_id}"


async def run_benchmark(
    memory,
    dataset: Dataset,
    judge: str = "substring",
    concurrency: int = 8,
    skip_ingest: bool = False,
    user_prefix: str = "longeval",
    progress_fn: ProgressFn | None = None,
    qa: QAClients | None = None,
) -> tuple[list[QuestionResult], dict]:
    """Ingest each question's own sessions into its own user, then ask it.

    Questions are independent, so up to ``concurrency`` run at once; inside a
    question, turns stay in order (extraction sees what earlier turns saved).
    Results come back in dataset order.
    """
    gate = asyncio.Semaphore(concurrency)
    totals = {"ingested": 0, "errors": 0, "sessions": 0, "episodes": 0}
    done = 0
    t0 = time.time()

    async def one(q: Question) -> QuestionResult:
        nonlocal done
        async with gate:
            user_id = question_user(user_prefix, q)
            if not skip_ingest:
                # oldest first, so "later" mentions update earlier ones
                ordered = sorted(
                    q.session_ids,
                    key=lambda sid: _parse_session_date(dataset.sessions[sid].date) or datetime.min,
                )
                sessions = {sid: dataset.sessions[sid] for sid in ordered}
                stats = await ingest_sessions(memory, user_id, sessions)
                for key in totals:
                    totals[key] += stats.get(key, 0)
            result = await score_question(memory, user_id, q, judge, qa)
        done += 1
        if progress_fn:
            progress_fn(done, len(dataset.questions), q.category)
        return result

    results = await asyncio.gather(*(one(q) for q in dataset.questions))
    totals["elapsed_s"] = round(time.time() - t0, 1)
    return list(results), totals


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[QuestionResult]) -> dict:
    categories: dict[str, list[QuestionResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    def metrics(items: list[QuestionResult]) -> dict:
        n = len(items) or 1
        out = {
            "count": len(items),
            "hit_rate": round(sum(1 for r in items if r.hit) / n, 3),
            "mrr": round(sum(r.reciprocal_rank for r in items) / n, 3),
        }
        if any(r.answer for r in items):  # QA mode: accuracy plus what it cost to get there
            ms = sorted(r.recall_ms for r in items)
            out.update(
                accuracy=out["hit_rate"],
                context_tokens_avg=round(sum(r.context_tokens for r in items) / n),
                read_model_calls_avg=round(sum(r.read_model_calls for r in items) / n, 2),
                recall_ms_p50=round(ms[len(ms) // 2]) if ms else 0,
                recall_ms_p95=round(ms[min(len(ms) - 1, int(len(ms) * 0.95))]) if ms else 0,
            )
        return out

    report = {"overall": metrics(results), "by_category": {}}
    for cat in sorted(categories):
        report["by_category"][cat] = metrics(categories[cat])
    abstain = [r for r in results if r.abstention]
    if abstain and any(r.answer for r in results):
        report["by_category"]["abstention (_abs)"] = metrics(abstain)
    return report


def format_report(report: dict, results: list[QuestionResult], verbose: bool = False) -> str:
    lines: list[str] = []
    overall = report["overall"]
    lines.append("=" * 60)
    lines.append("  LongMemEval Results")
    lines.append("=" * 60)
    lines.append(f"  Questions      : {overall['count']}")
    if "accuracy" in overall:
        lines.append(f"  Accuracy (QA)  : {overall['accuracy']:.1%}   (fixed reader, official judge prompts)")
        lines.append(f"  Context tokens : {overall['context_tokens_avg']} per question")
        lines.append(f"  Model calls    : {overall['read_model_calls_avg']} per question inside memory")
        lines.append(f"  Recall latency : p50 {overall['recall_ms_p50']} ms, p95 {overall['recall_ms_p95']} ms")
    else:
        lines.append(f"  Hit Rate       : {overall['hit_rate']:.1%}")
        lines.append(f"  MRR            : {overall['mrr']:.3f}")
    lines.append("-" * 60)

    for cat, m in report["by_category"].items():
        if "accuracy" in m:
            lines.append(
                f"  {cat:<30s}  n={m['count']:>3d}  acc={m['accuracy']:.1%}"
                f"  ctx={m['context_tokens_avg']:>5d} tok  p50={m['recall_ms_p50']} ms"
            )
        else:
            lines.append(
                f"  {cat:<30s}  n={m['count']:>3d}  hit={m['hit_rate']:.1%}"
                f"  mrr={m['mrr']:.3f}"
            )

    if verbose:
        lines.append("")
        lines.append("-" * 60)
        lines.append("  Misses:")
        lines.append("-" * 60)
        for r in results:
            if not r.hit:
                lines.append(f"  [{r.category}] {r.question}")
                lines.append(f"    expected: {r.expected_answers}")
                if r.answer:
                    lines.append(f"    answered: {r.answer[-160:]}")
                elif r.retrieved_contents:
                    lines.append(f"    got:      {r.retrieved_contents[0][:120]}...")
                else:
                    lines.append("    got:      (nothing retrieved)")
                lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
