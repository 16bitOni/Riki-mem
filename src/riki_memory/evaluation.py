"""A tiny retrieval test set: real questions with the memory text that should come back.

Run it after changing a prompt, a threshold, a weight or the embedding model,
and compare the numbers. The original Riki never got a trustworthy number out
of its eval because pass rates swung wildly between runs — keep your case file
stable and re-run it the same way every time before claiming an improvement.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .recall import search


class EvalCase(BaseModel):
    query: str
    expect: list[str] = Field(min_length=1, description="Text that must appear in a returned memory.")


def load_cases(path: str) -> list[EvalCase]:
    return [EvalCase(**item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]


def score_case(case: EvalCase, returned: list[str]) -> float:
    """1 / rank of the first returned memory containing an expected text; 0 if none does."""
    for rank, content in enumerate(returned, start=1):
        if any(text.lower() in content.lower() for text in case.expect):
            return 1.0 / rank
    return 0.0


async def run_eval(
    db: Database, llm: LLMProvider, config: MemoryConfig, user_id: str, cases: list[EvalCase]
) -> dict:
    """hit_rate = share of questions answered at all; mrr rewards putting the answer near the top."""
    scores, misses = [], []
    for case in cases:
        found, _ = await search(db, llm, config, user_id, case.query)
        value = score_case(case, [m.content for m in found])
        scores.append(value)
        if value == 0:
            misses.append(case.query)
    count = len(cases) or 1
    return {
        "cases": len(cases),
        "hit_rate": round(sum(1 for s in scores if s) / count, 3),
        "mrr": round(sum(scores) / count, 3),
        "misses": misses,
    }
