"""Memory namespaces. The host app owns the real user record; here a row just
carries the display name, timezone and rolling profile used by extraction and
recall prompts."""

from sqlalchemy import select

from .db import Database
from .models import User


async def ensure_user(
    db: Database, user_id: str, name: str = "", timezone: str = "UTC"
) -> User:
    """Fetch the user row, creating it on first use. ``user_id`` is opaque —
    pass your own user, session, or agent id."""
    async with db.session_scope() as session:
        user = await session.scalar(select(User).where(User.id == user_id))
        if user is None:
            user = User(id=user_id, name=name or user_id, timezone=timezone or "UTC")
            session.add(user)
            await session.flush()
        # Fill in blanks on later calls (name/timezone may arrive late).
        if name and not user.name:
            user.name = name
        if timezone and user.timezone in ("", "UTC") and timezone != "UTC":
            user.timezone = timezone
        return user
