"""Optional second pass: reorder the shortlist by how well each memory answers the question.

- ``"cross_encoder"``: a reranking model (bge-reranker, Cohere, Jina, ...) scores each
  pair. Tens of milliseconds, a fraction of a cent, the same order every time.
- ``"llm"``: the fast chat model picks the order. One full model call per question.

Off by default. It can only reorder what retrieval already found, and any failure
returns the original order unchanged.
"""

import logging

from pydantic import BaseModel, Field

from .llm import LLMProvider

log = logging.getLogger(__name__)

RERANK_PROMPT = """You rank memories by how useful each is for answering the user's message.
Return the numbers of the useful memories, most useful first. Leave out memories that are irrelevant."""


class Ranking(BaseModel):
    order: list[int] = Field(description="Memory numbers, most useful first.")


def apply_order(memories: list, order: list[int]) -> list:
    """Put the chosen memories first (in the given order); keep the rest after them in their old order.

    Numbers that are out of range or repeated are ignored, so a confused model cannot lose a memory.
    """
    seen: set[int] = set()
    first = []
    for number in order:
        if 1 <= number <= len(memories) and number not in seen:
            seen.add(number)
            first.append(memories[number - 1])
    rest = [m for i, m in enumerate(memories, start=1) if i not in seen]
    return first + rest


async def rerank(llm: LLMProvider, query: str, memories: list) -> list:
    if len(memories) < 2:
        return memories
    listing = "\n".join(f"{i}. {m.content}" for i, m in enumerate(memories, start=1))
    try:
        result = await llm.chat_json(
            system=RERANK_PROMPT,
            user=f"Message: {query}\n\nMemories:\n{listing}",
            schema=Ranking,
            fast=True,
        )
        return apply_order(memories, result.order)
    except Exception:
        log.exception("reranker failed; keeping the original order")
        return memories


async def cross_encode(llm, query: str, memories: list) -> list:
    """Sort by the provider's ``rerank_scores``; stable, so ties keep the fused order."""
    scorer = getattr(llm, "rerank_scores", None)
    if len(memories) < 2 or not callable(scorer):
        if not callable(scorer):
            log.warning('rerank="cross_encoder" but the provider has no rerank_scores(); order unchanged')
        return memories
    try:
        scores = await scorer(query, [m.content for m in memories])
        return [m for _, m in sorted(enumerate(memories), key=lambda pair: -scores[pair[0]])]
    except Exception:
        log.exception("cross-encoder rerank failed; keeping the original order")
        return memories
