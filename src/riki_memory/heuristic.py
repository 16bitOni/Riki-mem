"""Zero-token query planning: keywords, entities and dates from the message itself.

The LLM planner is better at one thing — resolving references ("that role",
"the thing I mentioned") using recent chat. Everything else it does (pick
keywords, name entities, spot dates) can be approximated with regex for very
small money: zero chat tokens.

``MemoryConfig.planner_mode`` picks the strategy:

- ``"llm"``:       always call the planner (original behaviour, most tokens)
- ``"heuristic"``: never call the planner (zero chat tokens)
- ``"escalate"``:  try the heuristic plan first; only call the LLM planner
  when cheap retrieval finds nothing (default — token-free in the common case)

Every mode still makes one embedding call for the semantic channel, the same
as the reference systems (one embedding per query, no LLM calls in retrieval).
"""

import re
from datetime import date, datetime, timedelta

from .schemas import RetrievalPlan
from .timeutil import now_in

MAX_TERMS = 10

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can cannot could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself me more most my myself no nor not of off on once only
    or other ought our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why with would you your
    yours yourself yourselves what when where why how which who whom whose tell show give
    get got going know think like just really still also even ever never always often
    """.split()
)

# Capitalised words that are almost never entities (sentence starts, questions).
_ENTITY_SKIP = frozenset(
    "what when where why how which who whom whose tell show give get got going is are "
    "was were do does did can could would should will shall may might must i we you".split()
)

_TEMPORAL_CALC_RE = re.compile(
    r"\bhow (?:many|much|long)\s+(?:days?|months?|years?|hours?|minutes?|weeks?)\s+"
    r"(?:before|after|since|ago|until|between|apart|did it take|does it take|will it take)",
    re.IGNORECASE,
)

_AGGREGATION_RE = re.compile(
    r"\b(how many|how much|total|all (?:the |my )?|every|each|count|sum)\b", re.IGNORECASE
)


def is_aggregation_query(text: str) -> bool:
    """True for counting/summing queries, but NOT temporal math like 'how many days before X'."""
    if not _AGGREGATION_RE.search(text):
        return False
    if _TEMPORAL_CALC_RE.search(text):
        return False
    return True


def _tags(text: str) -> list[str]:
    """Extract likely topic tags from the query — especially for aggregation questions."""
    if not _AGGREGATION_RE.search(text):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for phrase in re.findall(r"""['"]([^'"]{2,40})['"]""", text):
        tag = phrase.strip().lower().replace(" ", "-")
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    nouns = re.findall(
        r"\b(wedding|trip|doctor|visit|purchase|expense|model.?kit|instrument|"
        r"fish|tank|aquarium|game|class|festival|concert|book|movie|recipe|"
        r"bak(?:e|ing)|cook(?:ing)?|bike|guitar|piano|drum|project|pet|"
        r"charity|event|property|house|apartment|service|subscription|"
        r"device|gadget|appliance|car|vehicle)\w*\b",
        text.lower(),
    )
    for noun in nouns:
        tag = noun.replace(" ", "-")
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    for kw in _keywords(text):
        tag = kw.lower().replace(" ", "-")
        if len(tag) >= 3 and tag not in seen and tag not in STOPWORDS:
            seen.add(tag)
            tags.append(tag)
    return tags[:MAX_TERMS]


RELATION_WORDS = (
    "sister", "brother", "mother", "father", "wife", "husband", "partner",
    "girlfriend", "boyfriend", "son", "daughter", "uncle", "aunt", "cousin",
    "boss", "manager", "colleague", "coworker", "mentor", "client", "friend",
    "works at", "worked at", "lives in", "married to", "engaged to",
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTH_RE = "|".join(_MONTHS)
_WEEKDAY_RE = "|".join(_WEEKDAYS)


def _keywords(text: str) -> list[str]:
    """Quoted phrases first (strongest), then significant tokens."""
    out: list[str] = []
    seen: set[str] = set()
    for phrase in re.findall(r"""['"]([^'"]{2,60})['"]""", text):
        phrase = phrase.strip()
        if phrase and phrase.lower() not in seen:
            seen.add(phrase.lower())
            out.append(phrase)
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text.lower()):
        token = token.strip("'-")
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:MAX_TERMS]


def _entities(text: str) -> list[str]:
    """Quoted phrases plus capitalised word sequences ("Riya", "New York")."""
    found: list[str] = []
    seen: set[str] = set()

    def add(name: str):
        # A capitalised sequence can start with a sentence-start word
        # ("Does Riya" -> "Riya"): strip those from the front.
        words = [w for w in name.strip().split() if w.lower() not in _ENTITY_SKIP]
        name = " ".join(words)
        key = name.lower()
        if len(name) >= 2 and key not in seen:
            seen.add(key)
            found.append(name)

    for phrase in re.findall(r'"([^"]{2,60})"', text):
        add(phrase)
    for m in re.finditer(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,3})\b", text):
        add(m.group(1))
    return found[:MAX_TERMS]


def _relation_words(text: str) -> list[str]:
    found = []
    for w in RELATION_WORDS:
        # word boundaries, so "sister's" still matches "sister"
        if re.search(rf"\b{re.escape(w)}\b", text.lower()):
            found.append(w)
    return found[:MAX_TERMS]


# ------------------------------------------------------------------ dates


def _day_window(day: date, tz) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    return start, start + timedelta(days=1) - timedelta(microseconds=1)


def _week_window(day: date, tz) -> tuple[datetime, datetime]:
    monday = day - timedelta(days=day.weekday())
    start = datetime(monday.year, monday.month, monday.day, tzinfo=tz)
    return start, start + timedelta(days=7) - timedelta(microseconds=1)


def _month_window(year: int, month: int, tz) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=tz)
    nxt = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=tz)
    return start, nxt - timedelta(microseconds=1)


def _last_weekday(today: date, wd: int) -> date:
    return today - timedelta(days=(today.weekday() - wd) % 7 or 7)


def _next_weekday(today: date, wd: int) -> date:
    return today + timedelta(days=(wd - today.weekday()) % 7 or 7)


def _this_weekday(today: date, wd: int) -> date:
    return today - timedelta(days=today.weekday()) + timedelta(days=wd)


def _parse_dates(text: str, now: datetime) -> tuple[datetime | None, datetime | None]:
    """Find the first time window the message refers to, or (None, None)."""
    tz = now.tzinfo
    today = now.date()
    low = text.lower()

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", low)
    if m:
        try:
            return _day_window(date(int(m[1]), int(m[2]), int(m[3])), tz)
        except ValueError:
            pass

    m = re.search(r"\b(\d{1,3})\s+(day|week|month|year)s?\s+ago\b", low)
    if m:
        n, unit = int(m[1]), m[2]
        if unit == "day":
            return _day_window(today - timedelta(days=n), tz)
        if unit == "week":
            return _week_window(today - timedelta(weeks=n), tz)
        if unit == "month":
            month = today.month - n
            year = today.year + (month - 1) // 12
            return _month_window(year, (month - 1) % 12 + 1, tz)
        year = today.year - n
        return datetime(year, 1, 1, tzinfo=tz), datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=tz)

    if re.search(r"\byesterday\b", low):
        return _day_window(today - timedelta(days=1), tz)
    if re.search(r"\btoday\b", low):
        return _day_window(today, tz)
    if re.search(r"\btomorrow\b", low):
        return _day_window(today + timedelta(days=1), tz)

    m = re.search(rf"\b(last|this|next)\s+({_WEEKDAY_RE})\b", low)
    if m:
        which, wd = m[1], _WEEKDAYS[m[2]]
        day = {"last": _last_weekday, "this": _this_weekday, "next": _next_weekday}[which](today, wd)
        return _day_window(day, tz)

    m = re.search(r"\b(last|this|next)\s+(week|month|year)\b", low)
    if m:
        which, unit = m[1], m[2]
        if unit == "week":
            ref = {"last": -1, "this": 0, "next": 1}[which]
            return _week_window(today + timedelta(weeks=ref), tz)
        if unit == "month":
            ref = {"last": -1, "this": 0, "next": 1}[which]
            month = today.month + ref
            year = today.year + (month - 1) // 12
            return _month_window(year, (month - 1) % 12 + 1, tz)
        year = today.year + {"last": -1, "this": 0, "next": 1}[which]
        return datetime(year, 1, 1, tzinfo=tz), datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=tz)

    m = re.search(rf"\bin\s+({_MONTH_RE})\b", low)
    if m:
        month = _MONTHS[m[1]]
        year = now.year if month <= now.month else now.year - 1
        return _month_window(year, month, tz)

    return None, None


# ------------------------------------------------------------------ plan


def make_heuristic_plan(
    message: str, timezone: str, *, pregated: bool = False, now: datetime | None = None
) -> RetrievalPlan:
    """Build a RetrievalPlan with zero LLM calls.

    ``pregated``: the caller already decided this message needs nothing (the
    greeting pre-gate). ``now`` is injectable for tests.
    """
    if pregated:
        return RetrievalPlan(needs_memory=False)
    ref = now or now_in(timezone)
    start, end = _parse_dates(message, ref)
    return RetrievalPlan(
        needs_memory=True,
        keywords=_keywords(message),
        # No rewrite without an LLM; the raw message is the semantic query.
        semantic_query=message,
        entities=_entities(message),
        relation_words=_relation_words(message),
        tags=_tags(message),
        start=start,
        end=end,
    )
