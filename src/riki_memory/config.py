"""All tunables for the memory system in one place.

Pass a ``MemoryConfig`` to :class:`riki_memory.RikiMemory`. Every field has the
default that proved itself in production; change them only when you have a
measured reason (see ``evaluation.py`` for the retrieval test harness).
"""

from typing import Literal, get_args, get_origin

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    # -- retrieval -----------------------------------------------------------
    recall_limit: int = Field(15, ge=1, le=50, description="Memories handed to the LLM per request.")
    aggregation_recall_limit: int = Field(
        30, ge=1, le=100,
        description="Recall limit used for aggregation queries ('how many X', 'total', 'all my'). "
        "Higher than recall_limit to ensure all instances of a category fit in the window.",
    )
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
    weight_semantic: float = Field(0.6, ge=0, description="Weight of the meaning (vector) channel in rank fusion.")
    weight_graph: float = Field(0.7, ge=0, description="Weight of the people/places graph channel in rank fusion.")
    weight_time: float = Field(0.5, ge=0, description="Weight of the date-window channel in rank fusion.")
    weight_tag: float = Field(1.8, ge=0, description="RRF weight for the tag retrieval channel. "
                              "High by default because tag matches are precise and exhaustive.")
    tag_similarity_threshold: float = Field(
        0.82, ge=0.5, le=1,
        description="Cosine similarity threshold for normalizing a proposed tag to an "
        "existing canonical tag at write time, and for expanding query tags to "
        "semantically similar tags at read time. Lower = more aggressive merging.",
    )
    subject_match_threshold: float = Field(
        0.4, ge=0.1, le=1,
        description="Cosine similarity at which a stored instance subject becomes a candidate "
        "for a counting question's noun when no word matches ('instruments' -> 'guitar'). "
        "One fast LLM call then keeps only real members, so this is set low for recall. "
        "Measured: amazon.titan-embed-text-v1 0.40 (96% of members kept, 0.98 precision after "
        "the LLM step); cohere.embed-v4 0.56 (96%, 0.94). Too low only costs a few tokens, too "
        "high loses members, so 0.40 is a safe default for both. Other models score on other "
        "scales: run longeval/calibrate_subjects.py after changing embedding model.",
    )
    importance_weight: float = Field(
        0.05, ge=0, le=0.3, description="Small boost per importance point above 3."
    )
    rerank: Literal["off", "cross_encoder", "llm"] = Field(
        "off",
        description='"cross_encoder" re-orders the top candidates with a reranking model '
        "(RIKI_RERANK_MODEL, served at {base_url}/rerank): fast and cheap. \"llm\" uses the fast "
        "chat model: one extra model call per question, with every candidate's text.",
    )
    rerank_candidates: int = Field(
        20, ge=5, le=50, description="How many top candidates the reranker reorders (when rerank is on)."
    )

    # -- dedup ---------------------------------------------------------------
    dedup_trigram_threshold: float = Field(0.85, ge=0.5, le=1, description="Spelling-level similarity.")
    dedup_vector_threshold: float = Field(
        0.93, ge=0.5, le=1, description="Meaning-level similarity (embedded memories only)."
    )

    # -- maintenance ---------------------------------------------------------
    reembed_concurrency: int = Field(
        5, ge=1, le=20, description="Embedding calls run at once by reembed_stale()."
    )
    consolidation_min_cluster_size: int = Field(
        3, ge=2, le=10, description="consolidate() only merges groups of at least this many similar memories."
    )
    consolidation_similarity_threshold: float = Field(
        0.8, ge=0.5, le=0.99,
        description="How similar (cosine) memories must be for consolidate() to treat them as one group.",
    )
    profile_refresh_after: int = Field(
        10, ge=1, description="Rebuild the user profile after this many new memories."
    )
    empty_write_warning_after: int = Field(
        25, ge=0,
        description="Log a warning after this many writes in a row propose no memories "
        "without any error: the sign that the model's replies are not being understood. 0 = never.",
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

    # -- episodic memory -----------------------------------------------------
    enable_episodes: bool = Field(
        False,
        description="Enable the episodic memory layer. Call close_session() after each "
        "session to generate episode summaries and link them to atomic memories.",
    )
    weight_episode: float = Field(0.6, ge=0, description="RRF weight for the episode retrieval channel.")
    episode_similarity_threshold: float = Field(
        0.35, ge=0, le=1,
        description="Minimum cosine similarity for an episode to match a query.",
    )

    # -- aggregation ---------------------------------------------------------
    enable_sql_aggregation: bool = Field(
        True,
        description="Route 'how many X' / 'total Y' questions to deterministic SQL "
        "against the Instance table instead of RRF top-k text retrieval.",
    )

    # -- agentic retrieval ---------------------------------------------------
    retrieval_mode: Literal["single", "agentic"] = Field(
        "single",
        description="'single' does one-shot hybrid search (original behaviour). "
        "'agentic' runs a reflector after retrieval that may issue follow-up searches.",
    )
    max_retrieval_steps: int = Field(
        3, ge=1, le=5,
        description="Maximum total search steps in agentic mode (including the initial search).",
    )

    # -- extraction ----------------------------------------------------------
    extraction_instructions: str = Field(
        "", max_length=4000,
        description="Your app's own rules for the extractor, added to its prompt. Use it to "
        "teach new kinds of data, e.g. 'For medications, put name, dose and schedule in "
        "attributes.' Structured values the model fills in land in Memory.attributes.",
    )

    store_turns: bool = Field(
        True,
        description="Keep each exchange in the turn log before extracting: a crash means a retry, "
        "not a lost turn, and reextract() can replay history after a prompt or model change. "
        "Off: nothing raw is stored (and remember(wait=False) is unavailable).",
    )

    # -- discovering the options ---------------------------------------------

    @classmethod
    def describe(cls) -> list[dict]:
        """Every option: section, name, type, allowed values or range, default, meaning."""
        import inspect
        import re

        section, sections = "general", {}
        for line in inspect.getsource(cls).splitlines():
            if m := re.match(r"\s*# -- (.+?) -+\s*$", line):
                section = m.group(1)
            elif m := re.match(r"\s{4}(\w+):", line):
                sections[m.group(1)] = section
        rows = []
        for name, info in cls.model_fields.items():
            limits = {}
            for item in info.metadata:
                for bound in ("ge", "le", "max_length"):
                    if getattr(item, bound, None) is not None:
                        limits[bound] = getattr(item, bound)
            if get_origin(info.annotation) is Literal:
                kind, limits["choices"] = "choice", list(get_args(info.annotation))
            else:
                kind = getattr(info.annotation, "__name__", None) or str(info.annotation).replace("typing.", "")
            rows.append({
                "section": sections.get(name, "general"),
                "name": name,
                "type": kind,
                "limits": limits,
                "default": info.default,
                "description": info.description or "",
            })
        return rows

    def changed(self) -> dict:
        """Only the options that differ from their defaults."""
        return {
            name: value for name, value in self.model_dump().items()
            if value != type(self).model_fields[name].default
        }
