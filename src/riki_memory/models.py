"""The internal schema. It lives inside this package and is never exposed to the
host application: the host only passes opaque ``user_id`` strings and calls
``save`` / ``search``. Nobody migrates these tables into their own database.

Postgres only. Embeddings are pgvector ``vector`` columns with no fixed
dimension, so vectors from an old and a new embedding model can live side by
side during a re-embed; ``vectors.py`` gives each model its own HNSW index.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .vectors import Embedding


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


NAMESPACE_KINDS = ("user", "agent", "team", "org")


class User(Base):
    """A memory namespace. Every memory belongs to exactly one.

    The host app owns the real user; here a row carries the display name,
    timezone and rolling profile. ``id`` is globally unique (e.g. "acme/alice");
    ``tenant_id`` says which organisation owns it, ``kind`` whose memory it is
    (a person, an agent, or a team/org whose memory is shared). See
    Docs/SCOPE_MODEL.md.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", server_default="default", index=True)
    kind: Mapped[str] = mapped_column(String(20), default="user", server_default="user")
    name: Mapped[str] = mapped_column(String(200), default="")
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    profile: Mapped[str] = mapped_column(Text, default="")  # the core-memory blocks, rendered
    # {label: [{"text": ..., "from": [memory ids]}]}; see profile.py
    blocks = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    profile_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint(
            "kind in (" + ",".join(f"'{k}'" for k in NAMESPACE_KINDS) + ")", name="user_kind_valid"
        ),
    )


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # fact | preference | event | episode
    content: Mapped[str] = mapped_column(Text)  # one self-contained sentence or short paragraph
    content_hash: Mapped[str] = mapped_column(String(64))  # sha256 of normalised content
    importance: Mapped[int] = mapped_column(Integer, default=3)
    event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_key: Mapped[str | None] = mapped_column(String(512))
    # Only filled when meaning-search is useful. `deferred`: never loaded unless asked for.
    embedding = mapped_column(Embedding(), nullable=True, deferred=True)
    # Drift protection: which model made the vector, and whether this memory should have one at all.
    # Search only compares vectors from the current model; `reembed_stale` brings the rest up to date.
    wants_embedding: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    tags = mapped_column(JSON, nullable=True, default=list)
    # Your app's own structured fields ({"type": "medication", "dose": "5 mg"}); filterable.
    attributes = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # When it stopped (or will stop) being true. Set on an active memory for a temporary
    # state ("in Goa until Sunday"); set on a retired one when the world changed ("moved
    # to Bangalore"), which keeps it as history. A reworded duplicate has none.
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(12), default="stated", server_default="stated")
    forgotten_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    superseded_by_id: Mapped[str | None] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("user_id", "content_hash", name="memory_user_content_unique"),
        CheckConstraint("kind in ('fact','preference','event','episode')", name="memory_kind_valid"),
        CheckConstraint("source in ('stated','inferred','assistant')", name="memory_source_valid"),
        CheckConstraint("importance between 1 and 5", name="memory_importance_range"),
        CheckConstraint("length(content) > 0", name="memory_content_not_empty"),
        CheckConstraint(
            "embedding is null or embedding_model is not null", name="memory_vector_names_its_model"
        ),
        Index("memory_user_active_created", "user_id", "is_active", "created_at"),
        Index("memory_user_event", "user_id", "event_at"),
        Index("memory_source_key", "source_key"),
        Index("memory_attributes", "attributes", postgresql_using="gin"),
    )


class Entity(Base):
    """A person, place, org or project mentioned in memories."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(String(20), default="other")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint(
            "type in ('person','place','org','project','other')", name="entity_type_valid"
        ),
        # "riya" and "Riya" are the same person
        Index("entity_user_name_unique", "user_id", text("lower(name)"), unique=True),
    )


class MemoryEntity(Base):
    __tablename__ = "memory_entities"

    memory_id: Mapped[str] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (Index("memory_entity_entity", "entity_id"),)


class EntityRelation(Base):
    """``Riya --sister of--> Subhadip``."""

    __tablename__ = "entity_relations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    predicate: Mapped[str] = mapped_column(String(100))  # "sister of", "works at", "lives in"
    target_id: Mapped[str] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))

    __table_args__ = (
        UniqueConstraint("subject_id", "predicate", "target_id", name="relation_unique"),
        Index("relation_target", "target_id"),
    )


class Episode(Base):
    """A session-level summary linking the atomic memories extracted from one conversation."""

    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_key: Mapped[str] = mapped_column(String(512))
    summary: Mapped[str] = mapped_column(Text)
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding = mapped_column(Embedding(), nullable=True, deferred=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint("user_id", "session_key", name="episode_user_session_unique"),
        Index("episode_user_created", "user_id", "created_at"),
    )


class EpisodeMemory(Base):
    """Junction linking an episode to its constituent atomic memories."""

    __tablename__ = "episode_memories"

    episode_id: Mapped[str] = mapped_column(
        ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True
    )
    memory_id: Mapped[str] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (Index("episode_memory_memory", "memory_id"),)


class Instance(Base):
    """One countable thing: an event attended, an item owned, a visit made.

    Populated during the same extraction LLM call that creates Memory rows
    (zero additional cost).  Enables deterministic SQL COUNT/SUM/AVG for
    aggregation queries instead of LLM-counting over ranked text.

    ``kind``: a small, domain-free closed list (event, item, visit, ...).
    ``subject``: the free-text noun a person counts by ("instrument",
    "wedding", "doctor"), singular and normalised per user, so the schema
    works for things nobody anticipated.
    ``instance_key``: normalized identity (e.g. "bosch-dishwasher"),
    so the same real-world thing mentioned across sessions deduplicates.
    ``status``: separates occurrences from plans/desires/former — the filter
    that makes "how many did I attend" exclude "plans to attend".
    """

    __tablename__ = "instances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    subject: Mapped[str] = mapped_column(String(100))
    instance_key: Mapped[str] = mapped_column(String(500))
    display_name: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20))
    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    numeric_value: Mapped[float | None] = mapped_column(nullable=True)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_memory_id: Mapped[str] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint(
            "status in ('occurred','planned','desired','former','current')",
            name="instance_status_valid",
        ),
        UniqueConstraint(
            "user_id", "kind", "subject", "instance_key", "source_memory_id",
            name="instance_user_kind_subject_key_memory_unique",
        ),
        Index("instance_user_subject", "user_id", "subject"),
    )


TURN_STATUSES = ("pending", "processing", "done", "failed", "redacted")


class Turn(Base):
    """One exchange as it arrived: the raw record memories are extracted from.

    Written before extraction, so a crash means a retry rather than a lost turn,
    failures stay visible (``status='failed'``), and turns can be re-extracted
    after a prompt or model change. ``key`` is the host's ``turn_key`` (or
    ``turn:<id>``) and becomes the ``source_key`` of every memory from it.
    ``known_ids``: the memories the caller showed (``None`` = look them up).
    """

    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(512))
    user_text: Mapped[str] = mapped_column(Text)
    reply_text: Mapped[str] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # when the exchange happened
    known_ids = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("user_id", "key", name="turn_user_key_unique"),
        CheckConstraint(
            "status in (" + ",".join(f"'{s}'" for s in TURN_STATUSES) + ")", name="turn_status_valid"
        ),
        Index("turn_status_created", "status", "created_at"),
        Index("turn_user_at", "user_id", "at"),
    )
