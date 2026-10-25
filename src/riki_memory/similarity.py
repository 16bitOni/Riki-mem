"""In-Python similarity, for comparing a handful of already-loaded rows.

Search runs in Postgres (``keyword.py``, ``vectors.py``). These two are for the
small comparisons around a write: near-duplicate checks against the few
memories in the same date slot, tag/subject matching, consolidation.

:func:`trigram_similarity` gives the same scores as pg_trgm's ``similarity()``
(same word splitting and padding), so the ``dedup_trigram_threshold`` default
(0.85), tuned against pg_trgm, carries over unchanged.
"""

import re
from math import sqrt

# pg_trgm: anything that is not a letter or digit separates words, so
# "Paris." and "Paris!" are the same word.
_WORD_RE = re.compile(r"[^\W_]+")


def trigrams(text: str) -> set[str]:
    """Lower-cased trigram set, pg_trgm style: two spaces before each word, one after."""
    grams: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        padded = f"  {word} "
        grams.update(padded[i : i + 3] for i in range(len(padded) - 2))
    return grams


def trigram_similarity(a: str, b: str) -> float:
    """0 (nothing alike) … 1 (identical), Jaccard over trigram sets."""
    ga, gb = trigrams(a), trigrams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity, 0 … 1 for normalised embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
