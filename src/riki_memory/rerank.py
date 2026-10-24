"""Optional second pass: a cheap LLM reorders the shortlist by how well each memory
answers the question.

Off by default (``MemoryConfig.rerank = "off"``) because it adds one model call
per message. It can only reorder what retrieval already found, and any failure
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
