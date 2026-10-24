"""Shapes for memory.

Reading:  RetrievalPlan   (what the router decides to search for)
Writing:  NewMemory       (what the extraction LLM proposes; forgiving, so one odd value
                           does not lose the whole turn)
          MemoryCreate    (what the database accepts; strict, built from a NewMemory
                           or constructed directly by the host app)
"""

import logging
from datetime import datetime
from typing import Annotated, Literal, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

log = logging.getLogger(__name__)

MemoryKind = Literal["fact", "preference", "event", "episode"]
EntityType = Literal["person", "place", "org", "project", "other"]
MemoryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
Importance = Annotated[int, Field(ge=1, le=5, description="1 = trivia ... 5 = core to who the user is")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Kinds the extraction model tends to invent, mapped to the closest real one.
KIND_ALIASES = {
    "plan": "event",
    "commitment": "event",
    "intention": "event",
    "appointment": "event",
    "goal": "fact",
    "decision": "fact",
    "relationship": "fact",
    "like": "preference",
    "dislike": "preference",
    "opinion": "preference",
    "summary": "episode",
}

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Predicate = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class RetrievalPlan(BaseModel):
    """How to look something up in long-term memory."""

    needs_memory: bool = Field(
        description="False for greetings, small talk, or questions that need nothing personal."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Words or short phrases likely to appear verbatim in the memory (names, places, things).",
    )
    semantic_query: str | None = Field(
        default=None,
        description="A rewritten query for meaning-based search, or null if keywords are enough.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Names of people, places, organisations or projects the question is about.",
    )
    relation_words: list[str] = Field(
        default_factory=list,
        description="Relationship words in the question, such as 'sister', 'husband', 'boss', 'works at'.",
    )
    start: datetime | None = Field(
        default=None, description="Start of the time window the question refers to, if any."
    )
    end: datetime | None = Field(
        default=None, description="End of the time window the question refers to, if any."
    )

    @field_validator("start", "end", mode="before")
    @classmethod
    def _strip_datetime_prefix(cls, v):
        if isinstance(v, str):
            v = v.lstrip("~≈> </.")
            # reject single-char or clearly non-date strings
            if len(v) < 4:
                return None
        return v or None


ENTITY_TYPE_ALIASES = {
    "event": "other",
    "activity": "other",
    "location": "place",
    "company": "org",
    "organisation": "org",
    "organization": "org",
    "repo": "project",
    "repository": "project",
    "team": "org",
}


class EntityMention(BaseModel):
    name: Name
    type: EntityType = "other"

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_entity_type(cls, value):
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        return ENTITY_TYPE_ALIASES.get(key, key if key in get_args(EntityType) else "other")


class RelationMention(BaseModel):
    subject: Name
    predicate: Predicate = Field(description="Short verb phrase, e.g. 'sister of', 'works at'.")
    target: Name


class NewMemory(BaseModel):
    """A memory as proposed by the extraction LLM."""

    content: str = Field(description="One self-contained sentence, understandable without the chat.")
    kind: MemoryKind = Field(
        description="Exactly one of: fact, preference, event, episode. "
        "Future plans and commitments are 'event' (with event_at) or 'fact' if undated."
    )
    importance: int = Field(description="1 = trivia ... 5 = core to who the user is.")
    event_at: datetime | None = Field(
        default=None, description="When it happened or will happen, if it is tied to a date."
    )
    semantic: bool = Field(
        description="True for opinions, feelings, preferences, ideas. False for names, dates, numbers."
    )
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationMention] = Field(default_factory=list)
    replaces: list[str] = Field(
        default_factory=list, description="IDs of known memories this one makes outdated."
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value):
        """Map kinds the model invents ("plan", "goal", ...) onto the real ones instead of failing."""
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        return KIND_ALIASES.get(key, key if key in get_args(MemoryKind) else "fact")


class Extraction(BaseModel):
    memories: list[NewMemory] = Field(default_factory=list)

    @field_validator("memories", mode="before")
    @classmethod
    def _drop_invalid(cls, items):
        """Validate each memory on its own so one malformed item does not discard the rest."""
        if not isinstance(items, list):
            return items
        kept = []
        for item in items:
            try:
                kept.append(item if isinstance(item, NewMemory) else NewMemory.model_validate(item))
            except ValidationError as error:
                log.warning("dropped an invalid memory from the extractor: %s", error)
        return kept


class MemoryCreate(Strict):
    """The only way a memory gets into the database.

    ``user_id`` is an opaque string owned by the host app — pass your own user,
    session, or agent id. The memory system never looks inside it.
    """

    user_id: str = Field(min_length=1, max_length=64)
    content: MemoryText
    kind: MemoryKind = "fact"
    importance: Importance = 3
    event_at: AwareDatetime | None = None  # must carry a timezone
    semantic: bool | None = None  # None = decide by config (embed_all_kinds)
    source_key: Annotated[str, StringConstraints(max_length=512)] | None = None
    entities: list[EntityMention] = Field(default_factory=list, max_length=10)
    relations: list[RelationMention] = Field(default_factory=list, max_length=10)
    replaces: list[str] = Field(default_factory=list, max_length=20)
