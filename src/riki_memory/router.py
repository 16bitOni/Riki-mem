"""Decide which path a message takes through recall — before any LLM call.

One function, one answer, recorded in every recall report as ``report["route"]``:

- ``chitchat``    — greetings/thanks/farewells; nothing personal is looked up;
- ``temporal``    — date arithmetic between remembered events ("how many days
  before X did I Y"); answered by computing a date difference in code;
- ``exploratory`` — open-ended ("what do you know about me"); a broader search;
- ``aggregate``   — counting/summing ("how many X", "total Y"); answered from
  the Instance table with a deterministic SQL query;
- ``lookup``      — everything else; hybrid retrieval.

Zero tokens: regex only. A route that cannot produce an answer (no matching
subject, an anchor with no date) falls back to ``lookup`` inside recall, and
the report says why.
"""

import re
from typing import Literal

from .heuristic import is_aggregation_query, is_temporal_arithmetic

Route = Literal["chitchat", "temporal", "exploratory", "aggregate", "lookup"]

# Messages that obviously need nothing personal.
PREGATE_RE = re.compile(
    r"^\s*(?:(?:hi+|hey+|hello+|yo|sup|good\s?(?:morning|afternoon|evening|night|day)|"
    r"thanks?|thank\s?you|thx|bye+|goodbye|see\s?you|ok|okay|k|cool|nice|great|"
    r"lol|haha)\b[\s,!?.…]*)+$",
    re.IGNORECASE,
)

_EXPLORATORY_RE = re.compile(
    r"\b(what (?:do|did) you (?:know|remember)|tell me (?:all |everything )?about (?:me|myself)\b|"
    r"tell me everything|everything (?:you know|about)|anything about me|"
    r"what have (?:we|i) (?:talked|discussed|chatted|said)|"
    r"summar(?:y|ize|ise) (?:of )?(?:me|my|what|everything))",
    re.IGNORECASE,
)


def route_query(text: str, pregate: bool = True) -> Route:
    text = text or ""
    if pregate and PREGATE_RE.match(text):
        return "chitchat"
    if is_temporal_arithmetic(text):
        return "temporal"
    if _EXPLORATORY_RE.search(text):
        return "exploratory"
    if is_aggregation_query(text):
        return "aggregate"
    return "lookup"
