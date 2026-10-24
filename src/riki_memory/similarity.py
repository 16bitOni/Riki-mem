"""Portable similarity functions.

Replaces the two Postgres-only operators the original used:

- ``pg_trgm`` ``%`` / ``similarity()`` → :func:`trigram_similarity` (Jaccard over
  trigram multisets, same padding/normalisation as pg_trgm)
- ``pgvector`` ``cosine_distance`` → :func:`cosine_similarity` (pure Python;
  install the ``numpy`` extra if you outgrow it)

The ``dedup_trigram_threshold`` default (0.85) was tuned against pg_trgm;
Jaccard behaves the same monotonically, so retune only if dedup feels
too eager or too lax on your data.
"""

from collections import Counter
from math import sqrt


def trigrams(text: str) -> Counter:
    """Lower-cased, space-padded trigram multiset, pg_trgm style."""
    counts: Counter = Counter()
    for word in text.lower().split():
        padded = f"  {word}  "
        for i in range(len(padded) - 2):
            counts[padded[i : i + 3]] += 1
    return counts


def trigram_similarity(a: str, b: str) -> float:
    """0 (nothing alike) … 1 (identical), Jaccard over trigram multisets."""
    ca, cb = trigrams(a), trigrams(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    shared = sum((ca & cb).values())
    total = sum(ca.values()) + sum(cb.values()) - shared
    return shared / total if total else 1.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity, 0 … 1 for normalised embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
