"""riki-memory: a standalone long-term memory layer for AI agents.

LLM extraction turns conversations into durable memories; hybrid retrieval
(keyword + semantic + time + entity-graph, fused with weighted RRF) brings the
right ones back. SQLite is the default store, and you bring your own LLM via a
tiny provider protocol — no framework lock-in.
"""

from .config import MemoryConfig
from .evaluation import EvalCase, load_cases
from .heuristic import make_heuristic_plan
from .keyword import PostgresTSIndex, SQLiteFTSIndex, select_keyword_index
from .llm import LLMProvider, NullProvider, OpenAICompatibleProvider
from .models import Entity, EntityRelation, Episode, EpisodeMemory, Memory, MemoryEntity, User
from .schemas import EntityMention, Extraction, MemoryCreate, NewMemory, RelationMention, RetrievalPlan
from .service import RikiMemory
from .similarity import cosine_similarity, trigram_similarity

__version__ = "0.1.0"

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
    "PostgresTSIndex",
    "SQLiteFTSIndex",
    "select_keyword_index",
    "LLMProvider",
    "NullProvider",
    "OpenAICompatibleProvider",
    "Memory",
    "Entity",
    "EntityRelation",
    "Episode",
    "EpisodeMemory",
    "MemoryEntity",
    "User",
    "cosine_similarity",
    "trigram_similarity",
]
