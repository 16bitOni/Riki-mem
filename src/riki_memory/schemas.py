"""Shapes for memory.

Reading:  RetrievalPlan   (what the router decides to search for)
Writing:  NewMemory       (what the extraction LLM proposes; forgiving, so one odd value
                           does not lose the whole turn)
          MemoryCreate    (what the database accepts; strict, built from a NewMemory
                           or constructed directly by the host app)
"""

import logging
from datetime import datetime
from typing import Annotated, ClassVar, Literal, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

from . import telemetry

log = logging.getLogger(__name__)

MemoryKind = Literal["fact", "preference", "event", "episode"]
EntityType = Literal["person", "place", "org", "project", "other"]
# stated: the user said it. inferred: the extractor concluded it (a mood, an implied
# preference). assistant: it came from the assistant's reply (advice the user received).
MemorySource = Literal["stated", "inferred", "assistant"]
# Core memory: short lists shown in every context (see profile.py).
CoreBlock = Literal["identity", "preferences", "constraints", "people", "projects"]
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
        description="Topic tags to search for (e.g. ['interview', 'dentist']). For counting or "
        "aggregation questions ('how many X', 'total', 'all my') put the noun being counted first.",
    )
    start: datetime | None = Field(
        default=None, description="Start of the time window the question refers to, if any."
    )
    end: datetime | None = Field(
        default=None, description="End of the time window the question refers to, if any."
    )
    include_history: bool = Field(
        default=False,
        description="True when the question is about how things were before ('used to', "
        "'before I moved', 'my old job'): facts that are no longer true are searched too.",
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
        "Prefer the broader class a person would ask about ('appliance' for a dishwasher, "
        "'course' for a Python bootcamp); the specifics go in display_name."
    )
    instance_key: str = Field(
        description="Normalized identity string for dedup (lowercase-hyphenated). "
        "Examples: 'bosch-dishwasher', 'priya-farewell-dinner', 'dr-mehta-dentist'. "
        "The same real-world thing must always get the same key."
    )
    display_name: str = Field(
        description="Human-readable name. Examples: 'Bosch dishwasher', "
        "'Priya's farewell dinner', 'Dr. Mehta (dentist)'."
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


AttributeValue = str | int | float | bool | None | list[str | int | float | bool]
MAX_ATTRIBUTES = 30


def clean_attributes(value) -> dict:
    """Keep what fits ``attributes``: up to 30 string keys (60 chars) with scalar or
    list-of-scalar values. Model output that doesn't fit is dropped, not fatal."""
    if not isinstance(value, dict):
        return {}
    kept = {}
    for key, item in list(value.items())[:MAX_ATTRIBUTES]:
        if not isinstance(key, str) or not key.strip() or len(key) > 60:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            kept[key.strip()] = item
        elif isinstance(item, list) and all(isinstance(x, (str, int, float, bool)) for x in item):
            kept[key.strip()] = item[:50]
    return kept


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
    replaces: list[str] = Field(
        default_factory=list, description="IDs of known memories this one makes outdated."
    )
    source: MemorySource = Field(
        default="stated",
        description="'stated' if the user said it; 'inferred' if you concluded it from how they "
        "spoke (moods, implied preferences); 'assistant' if it comes from the assistant's reply.",
    )
    core: CoreBlock | None = Field(
        default=None,
        description="The core-memory block this belongs in, only for the few facts worth having "
        "in every conversation; empty for everything else.",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="Only for a temporary state with a known end ('in Goa until Sunday', "
        "'on leave this week'): when it stops being true. Empty otherwise.",
    )
    instances: list[InstanceItem] = Field(
        default_factory=list,
        description="Countable things THIS memory mentions (zero, one or many). "
        "Never list an instance under a memory that does not mention it.",
    )
    attributes: dict[str, AttributeValue] = Field(
        default_factory=dict,
        description="Structured fields, ONLY when the app's instructions ask for them; otherwise empty.",
    )

    @field_validator("attributes", mode="before")
    @classmethod
    def _keep_valid_attributes(cls, value):
        return clean_attributes(value)

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

    @field_validator("entities", "relations", mode="before")
    @classmethod
    def _at_most_ten(cls, values):
        """A crowded sentence names more than MemoryCreate keeps: keep the first ten, not lose the memory."""
        return values[:10] if isinstance(values, list) else values

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value):
        """Map kinds the model invents ("plan", "goal", ...) onto the real ones instead of failing."""
        if not isinstance(value, str):
            return value
        key = value.strip().lower()
        return KIND_ALIASES.get(key, key if key in get_args(MemoryKind) else "fact")


class Extraction(BaseModel):
    # The prompt carries every rule, so the schema sent to the model leaves its descriptions
    # out: they repeated the prompt and doubled the input tokens of every write.
    compact_schema: ClassVar[bool] = True

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
    # When the memories in `replaces` stopped being true. Set: the world changed, and
    # they are kept as searchable history. None: a correction (no history).
    changed_at: AwareDatetime | None = None
    source: MemorySource = "stated"
    core: CoreBlock | None = None  # extraction routes it to this core-memory block
    valid_to: AwareDatetime | None = None  # a temporary state ends then
    instances: list[InstanceItem] = Field(default_factory=list, max_length=20)
    # Your app's own structured fields: short keys, scalar or list-of-scalar values.
    attributes: dict[str, AttributeValue] = Field(default_factory=dict, max_length=MAX_ATTRIBUTES)
