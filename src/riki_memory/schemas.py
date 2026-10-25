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

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

from . import telemetry

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
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags to search for (e.g. ['wedding', 'doctor']). For counting or "
        "aggregation questions ('how many X', 'total', 'all my') put the noun being counted first.",
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


InstanceStatus = Literal["occurred", "planned", "desired", "former", "current"]

# Deliberately generic: the domain lives in the free-text ``subject``, so this
# list never needs to grow for a new kind of question.
InstanceKind = Literal["event", "item", "visit", "trip", "media", "person", "activity", "money", "other"]

# Kinds the extractor tends to invent, mapped to the closest real one.
INSTANCE_KIND_ALIASES = {
    "purchase": "item", "possession": "item", "product": "item", "object": "item", "pet": "item",
    "attendance": "event", "occasion": "event", "ceremony": "event",
    "appointment": "visit", "viewing": "visit",
    "travel": "trip", "journey": "trip",
    "book": "media", "film": "media", "movie": "media", "show": "media", "game": "media",
    "relationship": "person", "birth": "person",
    "session": "activity", "workout": "activity", "class": "activity", "hobby": "activity",
    "donation": "money", "expense": "money", "payment": "money", "income": "money",
}

# Units the extractor writes in many spellings, mapped to one canonical form so
# SUM never treats "$" and "USD" (or "hrs" and "hours") as different units.
UNIT_ALIASES = {
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD", "us$": "USD",
    "hour": "hours", "hr": "hours", "hrs": "hours", "h": "hours",
    "minute": "minutes", "min": "minutes", "mins": "minutes",
    "day": "days", "week": "weeks", "month": "months", "year": "years",
    "kilometer": "km", "kilometers": "km", "kms": "km",
    "mile": "miles", "mi": "miles",
    "item": "count", "items": "count", "piece": "count", "pieces": "count",
}


def _slugify(value: str, max_length: int) -> str:
    import re
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return (slug or value.strip().lower())[:max_length]


_IRREGULAR_PLURALS = {
    "people": "person", "children": "child", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
}
_SAME_IN_PLURAL = frozenset({"series", "species", "news", "glasses", "clothes", "jeans", "pants"})


def singular(word: str) -> str:
    """Rough English singular for one lowercase word. Good enough for subject nouns."""
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if len(word) <= 3 or word in _SAME_IN_PLURAL:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses", "zzes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def subject_slug(value: str) -> str:
    """Slug with its last word made singular: "Musical Instruments" -> "musical-instrument"."""
    slug = _slugify(value, 100)
    head, _, last = slug.rpartition("-")
    last = singular(last)
    return f"{head}-{last}" if head else last


class InstanceItem(BaseModel):
    """One countable thing extracted from a memory for deterministic aggregation."""

    kind: InstanceKind = Field(
        description="Exactly one of: event (attended or took part), item (bought, owned, sold or "
        "wanted), visit (a professional or place visited), trip, media (book, film, show, game), "
        "person, activity (a session of something done), money (spent, donated, raised, earned "
        "and not tied to an item), other."
    )
    subject: str = Field(
        description="The everyday noun you would count this by, singular: the X in 'how many X'. "
        "Prefer the broader class a person would ask about ('instrument' for a guitar, "
        "'doctor' for a dermatologist); the specifics go in display_name."
    )
    instance_key: str = Field(
        description="Normalized identity string for dedup (lowercase-hyphenated). "
        "Examples: 'fender-stratocaster', 'rachel-mike-wedding', 'dr-lee-orthopedics'. "
        "The same real-world thing must always get the same key."
    )
    display_name: str = Field(
        description="Human-readable name. Examples: 'Fender Stratocaster', "
        "'Rachel and Mike wedding', 'Dr. Lee (orthopedics)'."
    )
    status: InstanceStatus = Field(
        description="'occurred' for completed events/purchases, 'current' for things "
        "currently owned/active, 'planned' for future intentions, 'desired' for wishes, "
        "'former' for sold/ended/no-longer-true."
    )
    date: datetime | None = Field(
        default=None,
        description="When the instance occurred or was acquired, if known."
    )
    numeric_value: float | None = Field(
        default=None,
        description="Numeric amount ONLY if explicitly stated in the source text "
        "(dollars, hours, count). Never infer or guess."
    )
    unit: str | None = Field(
        default=None,
        description="Unit for numeric_value: 'USD', 'hours', 'days', 'km', 'count', etc."
    )

    # Lengths match the Instance columns: Postgres rejects an over-long value.
    @field_validator("instance_key", mode="before")
    @classmethod
    def _slugify_key(cls, v):
        return _slugify(v, 500) if isinstance(v, str) else v

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value):
        """Map kinds the model invents onto the real ones instead of dropping the instance."""
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        return INSTANCE_KIND_ALIASES.get(key, key if key in get_args(InstanceKind) else "other")

    @field_validator("subject", mode="before")
    @classmethod
    def _slugify_subject(cls, v):
        return subject_slug(v) if isinstance(v, str) else v

    @field_validator("display_name", mode="before")
    @classmethod
    def _cap_display_name(cls, v):
        return v.strip()[:500] if isinstance(v, str) else v

    @field_validator("unit", mode="before")
    @classmethod
    def _canonical_unit(cls, v):
        if not isinstance(v, str) or not v.strip():
            return None
        unit = v.strip()
        return UNIT_ALIASES.get(unit.lower(), unit)[:50]

    @model_validator(mode="after")
    def _default_unit(self):
        """A bare number with no unit is a count ("3 fish"), never a silent mix-in to a dollar SUM."""
        if self.numeric_value is not None and self.unit is None:
            self.unit = "count"
        return self


class NewMemory(BaseModel):
    """A memory as proposed by the extraction LLM."""

    content: str = Field(description="One self-contained sentence, understandable without the chat.")
    kind: MemoryKind = Field(
        description="Exactly one of: fact, preference, event, episode. "
        "Future plans and commitments are 'event' (with event_at) or 'fact' if undated."
    )
    # Defaults for the two judgement fields: some models omit them, and losing the
    # whole memory over a missing score is worse than a neutral guess.
    importance: int = Field(default=3, description="1 = trivia ... 5 = core to who the user is.")
    event_at: datetime | None = Field(
        default=None, description="When it happened or will happen, if it is tied to a date."
    )
    semantic: bool = Field(
        default=False,
        description="True for opinions, feelings, preferences, ideas. False for names, dates, numbers.",
    )
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationMention] = Field(default_factory=list)
    tags: list[str] = Field(
        default_factory=list,
        description="2-5 short lowercase topic/category labels describing what this memory "
        "is about. Examples: wedding, purchase, doctor-visit, hobby, travel, cooking, "
        "fitness, work, family, pet, finance, home-improvement, streaming, gaming.",
    )
    replaces: list[str] = Field(
        default_factory=list, description="IDs of known memories this one makes outdated."
    )
    instances: list[InstanceItem] = Field(
        default_factory=list,
        description="Countable things THIS memory mentions (zero, one or many). "
        "Never list an instance under a memory that does not mention it.",
    )

    @field_validator("instances", mode="before")
    @classmethod
    def _drop_invalid_instances(cls, items):
        """Validate each instance on its own so one malformed item does not discard the memory."""
        if not isinstance(items, list):
            return []
        kept = []
        for item in items:
            try:
                kept.append(item if isinstance(item, InstanceItem) else InstanceItem.model_validate(item))
            except ValidationError as error:
                log.warning("dropped an invalid instance from the extractor: %s", error)
        return kept

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, values):
        if not isinstance(values, list):
            return []
        return [t.strip().lower().replace(" ", "-") for t in values if isinstance(t, str) and t.strip()][:10]

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value):
        """Map kinds the model invents ("plan", "goal", ...) onto the real ones instead of failing."""
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        return KIND_ALIASES.get(key, key if key in get_args(MemoryKind) else "fact")


class Extraction(BaseModel):
    # Required, not defaulted: a reply without "memories" is a malformed answer
    # and must fail loudly, not read as "nothing worth remembering".
    memories: list[NewMemory]

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
                telemetry.record_dropped()
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
    tags: list[str] = Field(default_factory=list, max_length=10)
    replaces: list[str] = Field(default_factory=list, max_length=20)
    instances: list[InstanceItem] = Field(default_factory=list, max_length=20)
