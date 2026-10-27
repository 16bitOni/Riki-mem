"""The turn log: an exchange is written down first, then memories are extracted from it.

- ``enqueue`` records a turn (idempotent on its key) and can claim it for the caller.
- ``claim`` hands pending turns to a worker; ``FOR UPDATE SKIP LOCKED`` lets several
  workers share the queue without taking the same turn twice.
- ``finish`` marks a turn done or failed.

A turn whose worker died mid-extraction (``processing`` for longer than ``STALE``)
is claimed again; a turn that failed ``MAX_ATTEMPTS`` times stays ``failed`` with
its error, for a person to look at.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert

from .db import Database
from .models import Memory, Turn, User, new_id, utcnow

MAX_ATTEMPTS = 3
STALE = timedelta(minutes=10)


def _claimable(now: datetime):
    return or_(
        and_(Turn.status.in_(("pending", "failed")), Turn.attempts < MAX_ATTEMPTS),
        and_(Turn.status == "processing", Turn.claimed_at < now - STALE),
    )


async def enqueue(
    db: Database, user_id: str, user_text: str, reply_text: str, *,
    key: str | None, at: datetime | None, known_ids: list[str] | None, claim: bool,
    default_timezone: str = "UTC",
) -> tuple[Turn, bool]:
    """Record a turn. Returns ``(turn, claimed)``.

    The same ``key`` twice returns the turn already recorded (a retry never
    extracts twice). With ``claim`` the caller processes the turn itself.
    """
    turn_id = new_id()
    now = utcnow()
    at = at or now
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    async with db.session_scope() as session:
        await session.execute(
            insert(User).values(id=user_id, name=user_id, timezone=default_timezone).on_conflict_do_nothing()
        )
        row = await session.scalar(
            insert(Turn)
            .values(
                id=turn_id, user_id=user_id, key=key or f"turn:{turn_id}", user_text=user_text,
                reply_text=reply_text, at=at, known_ids=known_ids,
                status="processing" if claim else "pending", attempts=1 if claim else 0,
                claimed_at=now if claim else None, created_at=now,
            )
            .on_conflict_do_nothing()
            .returning(Turn)
        )
        if row is not None:
            return row, claim
        existing = await session.scalar(select(Turn).where(Turn.user_id == user_id, Turn.key == key))
    if claim:
        taken = await claim_turns(db, 1, turn_id=existing.id)
        if taken:
            return taken[0], True
    return existing, False


async def claim_turns(
    db: Database, limit: int, *, user_id: str | None = None, turn_id: str | None = None
) -> list[Turn]:
    """Mark up to ``limit`` claimable turns as ``processing`` and return them, oldest first."""
    now = utcnow()
    scope = [_claimable(now)]
    if user_id:
        scope.append(Turn.user_id == user_id)
    if turn_id:
        scope.append(Turn.id == turn_id)
    picked = (
        select(Turn.id).where(*scope).order_by(Turn.created_at, Turn.id)
        .limit(limit).with_for_update(skip_locked=True).scalar_subquery()
    )
    async with db.session_scope() as session:
        rows = (await session.scalars(
            update(Turn).where(Turn.id.in_(picked))
            .values(status="processing", attempts=Turn.attempts + 1, claimed_at=now, error=None)
            .returning(Turn)
            .execution_options(synchronize_session=False)
        )).all()
    return sorted(rows, key=lambda t: (t.created_at, t.id))


async def finish(db: Database, turn_id: str, error: str | None = None) -> None:
    async with db.session_scope() as session:
        await session.execute(
            update(Turn).where(Turn.id == turn_id)
            .values(status="failed" if error else "done", error=error, processed_at=utcnow())
        )


async def memories_of(db: Database, turn: Turn) -> list[Memory]:
    """What a finished turn saved (for a retry of an already processed turn)."""
    async with db.session_scope() as session:
        return list((await session.scalars(
            select(Memory).where(Memory.user_id == turn.user_id, Memory.source_key == turn.key)
            .order_by(Memory.created_at, Memory.id)
        )).all())


async def known_memories(db: Database, turn: Turn) -> list[Memory] | None:
    """The memories the caller showed when the turn was recorded (``None``: look them up)."""
    if turn.known_ids is None:
        return None
    async with db.session_scope() as session:
        return list((await session.scalars(
            select(Memory).where(Memory.user_id == turn.user_id, Memory.id.in_(turn.known_ids))
        )).all())


async def counts(db: Database, user_id: str | None = None) -> dict[str, int]:
    """Turns per status, e.g. ``{"done": 120, "pending": 3, "failed": 1}``."""
    scope = [Turn.user_id == user_id] if user_id else []
    async with db.session_scope() as session:
        rows = (await session.execute(
            select(Turn.status, func.count()).where(*scope).group_by(Turn.status)
        )).all()
    return {status: n for status, n in rows}


async def history(db: Database, user_id: str, since: datetime | None = None) -> list[Turn]:
    """Processed turns, oldest first (for re-extraction)."""
    stmt = select(Turn).where(Turn.user_id == user_id, Turn.status == "done")
    if since:
        stmt = stmt.where(Turn.at >= since)
    async with db.session_scope() as session:
        return list((await session.scalars(stmt.order_by(Turn.at, Turn.id))).all())


async def status_of(db: Database, turn_id: str) -> str | None:
    """The status of one turn."""
    async with db.session_scope() as session:
        return await session.scalar(select(Turn.status).where(Turn.id == turn_id))
