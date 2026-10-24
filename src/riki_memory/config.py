"""All tunables for the memory system in one place.

Pass a ``MemoryConfig`` to :class:`riki_memory.RikiMemory`. Every field has the
default that proved itself in production; change them only when you have a
measured reason (see ``evaluation.py`` for the retrieval test harness).
"""

from typing import Literal

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    # -- retrieval -----------------------------------------------------------
    recall_limit: int = Field(15, ge=1, le=30, description="Memories handed to the LLM per request.")
    min_similarity: float = Field(
        0.3, ge=0, le=1, description="Cosine similarity below this is not a semantic match."
    )
    graph_max_hops: int = Field(
        3, ge=1, le=4, description="How far to walk along relations."
    )
    graph_fanout_limit: int = Field(
        200, ge=10, le=2000, description="Most entities one graph walk may visit."
    )
    weight_keyword: float = Field(1.5, ge=0, description="How much each channel counts in rank fusion.")
    weight_semantic: float = Field(0.6, ge=0)
    weight_graph: float = Field(0.7, ge=0)
    weight_time: float = Field(0.5, ge=0)
    importance_weight: float = Field(
        0.05, ge=0, le=0.3, description="Small boost per importance point above 3."
    )
    rerank: Literal["off", "llm"] = Field(
        "llm", description='"llm" re-orders the top candidates with the fast model.'
    )
    rerank_candidates: int = Field(20, ge=5, le=50)

    # -- dedup ---------------------------------------------------------------
    dedup_trigram_threshold: float = Field(0.85, ge=0.5, le=1, description="Spelling-level similarity.")
    dedup_vector_threshold: float = Field(
        0.93, ge=0.5, le=1, description="Meaning-level similarity (embedded memories only)."
    )

    # -- maintenance ---------------------------------------------------------
    reembed_concurrency: int = Field(5, ge=1, le=20)
    consolidation_min_cluster_size: int = Field(3, ge=2, le=10)
    consolidation_similarity_threshold: float = Field(0.8, ge=0.5, le=0.99)
    profile_refresh_after: int = Field(
        10, ge=1, description="Rebuild the user profile after this many new memories."
    )

    # -- standalone behaviour ------------------------------------------------
    enable_entities: bool = Field(
        True,
        description="Extract and link entities/relations, and enable the graph retrieval "
        "channel. Disable to skip the write overhead when you only need fact recall.",
    )
    embed_all_kinds: bool = Field(
        True,
        description="Embed every kind of memory for semantic search. The original Riki "
        "only embedded preferences and episodes, which left facts invisible to the "
        "semantic channel.",
    )
    profile_injection: Literal["always", "when_empty", "never"] = Field(
        "when_empty",
        description="When to inject the user profile into the prompt context. "
        "'when_empty' (default) only injects it when recall found nothing, "
        "saving ~350 tokens on most turns.",
    )
    heuristic_pregate: bool = Field(
        True,
        description="Answer greetings/thanks/farewells without calling the planner LLM at all.",
    )
    planner_mode: Literal["llm", "heuristic", "escalate"] = Field(
        "escalate",
        description="How the retrieval plan is built. 'llm' always calls the planner model; "
        "'heuristic' never does (zero chat tokens — keywords/entities/dates come from regex); "
        "'escalate' tries the heuristic plan first and only calls the LLM planner when "
        "cheap retrieval finds nothing.",
    )
    default_timezone: str = Field("UTC", description="Assumed timezone for users without one.")
