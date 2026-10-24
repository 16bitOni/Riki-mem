"""Small helpers so timezone handling is done the same way everywhere."""

from datetime import datetime
from zoneinfo import ZoneInfo


def now_in(timezone: str) -> datetime:
    return datetime.now(ZoneInfo(timezone))


def localize(value: datetime | None, timezone: str) -> datetime | None:
    """LLMs sometimes return a time without an offset. Treat that as the user's local time."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=ZoneInfo(timezone))
    return value


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def parse_iso(text: str | None, timezone: str) -> datetime | None:
    if not text:
        return None
    return localize(datetime.fromisoformat(text), timezone)
