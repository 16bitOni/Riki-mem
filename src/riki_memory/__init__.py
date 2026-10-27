"""riki-memory: a standalone long-term memory layer for AI agents.

LLM extraction turns conversations into durable memories; hybrid retrieval
(keyword + semantic + time + entity-graph, fused with weighted RRF) brings the
right ones back. Postgres + pgvector is the store, and you bring your own LLM
via a tiny provider protocol — no framework lock-in.
"""

from .config import MemoryConfig
from .evaluation import EvalCase, load_cases
from .heuristic import make_heuristic_plan
from .llm import LLMProvider, NullProvider, OpenAICompatibleProvider
from .models import Entity, EntityRelation, Episode, EpisodeMemory, Instance, Memory, MemoryEntity, User
from .schemas import EntityMention, Extraction, InstanceItem, MemoryCreate, NewMemory, RelationMention, RetrievalPlan
from .service import RikiMemory
from .similarity import cosine_similarity, trigram_similarity
from .telemetry import WriteReport, WriteStats

__version__ = "0.2.0"

__all__ = [
    "RikiMemory",
    "MemoryConfig",
    "MemoryCreate",
    "NewMemory",
    "Extraction",
    "RetrievalPlan",
    "EntityMention",
    "RelationMention",
    "EvalCase",
    "load_cases",
    "make_heuristic_plan",
    "LLMProvider",
    "NullProvider",
    "OpenAICompatibleProvider",
    "Memory",
    "Entity",
    "EntityRelation",
    "Episode",
    "EpisodeMemory",
    "MemoryEntity",
    "Instance",
    "InstanceItem",
    "User",
    "cosine_similarity",
    "WriteReport",
    "WriteStats",
    "trigram_similarity",
]
