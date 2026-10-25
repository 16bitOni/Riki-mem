"""Agentic multi-step retrieval.

When ``MemoryConfig.retrieval_mode == "agentic"``, a reflector LLM examines
the initial search results and decides whether follow-up searches are needed.
This helps with queries that require multiple pieces of information — e.g.
"how many days between event X and event Y?" where both events must be
retrieved.

The loop:
1. Run hybrid_search with the original plan.
2. Ask the reflector if the results are sufficient.
3. If not, build a new plan from the reflector's follow-up query and search again.
4. Merge results (union by memory ID, keep highest score).
5. Repeat up to ``config.max_retrieval_steps`` total.
"""

import logging

from pydantic import BaseModel, Field

from .config import MemoryConfig
from .db import Database
from .llm import LLMProvider
from .models import Memory
from .retrieval import fuse_scores, hybrid_search, importance_boost
from .schemas import RetrievalPlan

log = logging.getLogger(__name__)

REFLECTOR_PROMPT = """You are a retrieval quality checker for a personal memory system.

Given a user's question and the memories retrieved so far, decide whether the
retrieved context contains ALL the information needed to answer the question.

Pay special attention to:
- Temporal questions ("how many days between X and Y?", "which happened first?"):
  BOTH events/dates must be present with their specific dates.
- Counting questions ("how many X before Y?"): ALL relevant items must be present,
  not just some of them.
- Comparison questions ("which was more expensive?"): BOTH items being compared
  must be present with the relevant details.

If information is missing, write a focused follow-up search query to find it.
Include specific keywords, entity names, and topic tags to search for.
For aggregation questions, use follow_up_tags (e.g. "wedding", "model-kit") to
find ALL instances of a category."""


class ReflectionResult(BaseModel):
    sufficient: bool = Field(
        default=True,
        description="True if the retrieved memories contain all information "
        "needed to answer the question.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of what is present and what is missing.",
    )
    follow_up_query: str | None = Field(
        default=None,
        description="A search query to find the missing information. "
        "Null when sufficient is True.",
    )
    follow_up_keywords: list[str] = Field(
        default_factory=list,
        description="Specific keywords for the follow-up search.",
    )
    follow_up_entities: list[str] = Field(
        default_factory=list,
        description="Entity names to search for in the follow-up.",
    )
    follow_up_tags: list[str] = Field(
        default_factory=list,
        description="Topic tags to search for (e.g. 'wedding', 'purchase'). "
        "Use for aggregation queries to find all instances.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.follow_up_query and self.sufficient:
            self.sufficient = False


def _merge_memory_lists(
    existing: list[Memory],
    existing_scores: dict[str, float],
    new: list[Memory],
    new_scores: dict[str, float],
) -> tuple[list[Memory], dict[str, float]]:
    """Union two ranked memory lists, keeping the higher score per memory."""
    seen = {m.id: m for m in existing}
    merged_scores = dict(existing_scores)

    for m in new:
        if m.id in seen:
            merged_scores[m.id] = max(merged_scores.get(m.id, 0), new_scores.get(m.id, 0))
        else:
            seen[m.id] = m
            merged_scores[m.id] = new_scores.get(m.id, 0)

    ranked = sorted(seen.values(), key=lambda m: merged_scores.get(m.id, 0), reverse=True)
    return ranked, merged_scores


async def agentic_search(
    db: Database,
    llm: LLMProvider,
    config: MemoryConfig,
    user_id: str,
    plan: RetrievalPlan,
    limit: int | None = None,
    rerank_query: str | None = None,
) -> tuple[list[Memory], dict]:
    """Multi-step retrieval with a reflector loop."""
    limit = limit or config.recall_limit
    max_steps = config.max_retrieval_steps
    agentic_steps = []

    found, report = await hybrid_search(
        db, llm, config, user_id, plan, limit=limit, rerank_query=rerank_query
    )
    all_scores = {
        r["id"]: r["score"] for r in report.get("results", [])
    }
    agentic_steps.append({"step": 1, "planner": "initial", "report": report})

    all_memories = list(found)

    for step in range(2, max_steps + 1):
        if not rerank_query:
            break

        memory_listing = "\n".join(
            f"- {m.content}" for m in all_memories[:20]
        ) or "(nothing retrieved)"

        try:
            reflection = await llm.chat_json(
                system=REFLECTOR_PROMPT,
                user=(
                    f"Question: {rerank_query}\n\n"
                    f"Retrieved memories:\n{memory_listing}"
                ),
                schema=ReflectionResult,
                fast=True,
            )
        except Exception:
            log.exception("reflector failed at step %d; returning current results", step)
            break

        if reflection.sufficient:
            log.debug("reflector says sufficient at step %d: %s", step, reflection.reasoning)
            break

        log.debug(
            "reflector step %d: %s -> follow-up: %s",
            step, reflection.reasoning, reflection.follow_up_query,
        )

        follow_up_plan = RetrievalPlan(
            needs_memory=True,
            keywords=reflection.follow_up_keywords or [],
            semantic_query=reflection.follow_up_query,
            entities=reflection.follow_up_entities or [],
            tags=reflection.follow_up_tags or [],
        )

        new_found, new_report = await hybrid_search(
            db, llm, config, user_id, follow_up_plan,
            limit=limit, rerank_query=reflection.follow_up_query,
        )
        new_scores = {
            r["id"]: r["score"] for r in new_report.get("results", [])
        }

        all_memories, all_scores = _merge_memory_lists(
            all_memories, all_scores, new_found, new_scores,
        )
        agentic_steps.append({
            "step": step,
            "reasoning": reflection.reasoning,
            "follow_up": reflection.follow_up_query,
            "report": new_report,
        })

    if len(agentic_steps) > 1 and config.rerank == "llm" and rerank_query:
        from .rerank import rerank
        try:
            all_memories = await rerank(llm, rerank_query, all_memories[:limit * 2])
        except Exception:
            log.exception("final rerank after agentic merge failed")

    final = all_memories[:limit]

    merged_report = dict(report)
    merged_report["agentic_steps"] = agentic_steps
    merged_report["agentic_total_steps"] = len(agentic_steps)
    merged_report["returned"] = len(final)
    merged_report["results"] = [
        {
            "id": str(m.id),
            "rank": rank,
            "kind": m.kind,
            "importance": m.importance,
            "score": round(all_scores.get(m.id, 0), 5),
            "text": m.content[:160],
        }
        for rank, m in enumerate(final, start=1)
    ]

    return final, merged_report
