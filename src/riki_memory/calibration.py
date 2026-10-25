"""Pick ``subject_match_threshold`` for a given embedding model from labelled pairs.

When no word matches, a counting question's noun ("instruments") reaches
stored subjects by meaning: embeddings above the threshold become candidates
and one fast LLM call keeps the real members ("guitar", not "guitar-lesson").
The pairs here are those hard cases: a class and its members should match, a
neighbour from the same domain ("wedding" vs "funeral") should not.

Embeddings alone can't separate the two (dog-cat scores like pet-dog), so the
threshold is set for recall and precision comes from the LLM step. Cosine
scales differ between embedding models: run ``python
longeval/calibrate_subjects.py`` once per model and set the result in config.
"""

from dataclasses import dataclass

from .schemas import subject_slug
from .similarity import cosine_similarity

# (question noun, stored subject, should the question count it?)
SUBJECT_PAIRS: list[tuple[str, str, bool]] = [
    # a class and its members
    ("instrument", "guitar", True),
    ("instrument", "piano", True),
    ("instrument", "violin", True),
    ("instrument", "ukulele", True),
    ("pet", "dog", True),
    ("pet", "cat", True),
    ("pet", "hamster", True),
    ("vehicle", "car", True),
    ("vehicle", "motorcycle", True),
    ("vehicle", "truck", True),
    ("fruit", "apple", True),
    ("fruit", "mango", True),
    ("sport", "tennis", True),
    ("sport", "football", True),
    ("sport", "basketball", True),
    ("country", "france", True),
    ("country", "japan", True),
    ("book", "novel", True),
    ("movie", "film", True),
    ("show", "series", True),
    ("doctor", "physician", True),
    ("doctor", "pediatrician", True),
    ("shoe", "sneaker", True),
    ("shoe", "boot", True),
    ("furniture", "sofa", True),
    ("furniture", "chair", True),
    ("plant", "succulent", True),
    ("plant", "fern", True),
    ("game", "board game", True),
    ("game", "video game", True),
    ("course", "class", True),
    ("workout", "run", True),
    ("workout", "yoga session", True),
    ("trip", "vacation", True),
    ("trip", "journey", True),
    ("gadget", "smartphone", True),
    ("gadget", "laptop", True),
    ("drink", "coffee", True),
    ("drink", "tea", True),
    ("relative", "cousin", True),
    ("relative", "aunt", True),
    ("event", "concert", True),
    ("event", "festival", True),
    ("appliance", "refrigerator", True),
    ("appliance", "microwave", True),
    ("clothing", "jacket", True),
    ("clothing", "shirt", True),
    ("bird", "parrot", True),
    ("restaurant", "cafe", True),
    ("gift", "present", True),
    # neighbours from the same domain that must stay apart
    ("wedding", "funeral", False),
    ("wedding", "birthday party", False),
    ("concert", "movie", False),
    ("book", "movie", False),
    ("book", "podcast", False),
    ("dog", "cat", False),
    ("pet", "vet", False),
    ("pet", "pet food", False),
    ("car", "bike", False),
    ("car", "car wash", False),
    ("car", "parking ticket", False),
    ("guitar", "guitar lesson", False),
    ("instrument", "concert", False),
    ("instrument", "song", False),
    ("doctor", "pharmacy", False),
    ("doctor", "medicine", False),
    ("dentist", "dermatologist", False),
    ("restaurant", "recipe", False),
    ("restaurant", "grocery", False),
    ("country", "city", False),
    ("country", "language", False),
    ("city", "hotel", False),
    ("trip", "flight", False),
    ("trip", "suitcase", False),
    ("shoe", "sock", False),
    ("shoe", "hat", False),
    ("plant", "garden tool", False),
    ("plant", "flower pot", False),
    ("fruit", "vegetable", False),
    ("fruit", "juice", False),
    ("sport", "gym membership", False),
    ("tennis", "golf", False),
    ("game", "toy", False),
    ("course", "teacher", False),
    ("course", "exam", False),
    ("workout", "protein shake", False),
    ("laptop", "phone", False),
    ("gadget", "charger", False),
    ("coffee", "tea", False),
    ("drink", "snack", False),
    ("cousin", "friend", False),
    ("relative", "coworker", False),
    ("relative", "neighbor", False),
    ("event", "ticket", False),
    ("festival", "museum", False),
    ("appliance", "furniture", False),
    ("jacket", "shirt", False),
    ("clothing", "jewelry", False),
    ("bird", "fish", False),
    ("gift", "donation", False),
]


@dataclass
class ThresholdScore:
    threshold: float
    precision: float
    recall: float
    f_half: float  # F0.5 weighs precision: a wrong subject corrupts a count


@dataclass
class Calibration:
    model: str
    best: ThresholdScore
    scores: list[ThresholdScore]
    similarities: list[tuple[str, str, bool, float]]

    def missed(self) -> list[tuple[str, str, bool, float]]:
        """True members the chosen threshold never shows the LLM."""
        t = self.best.threshold
        return [s for s in self.similarities if s[2] and s[3] < t]


def score_threshold(similarities: list[tuple[str, str, bool, float]], threshold: float) -> ThresholdScore:
    tp = sum(1 for *_, label, sim in similarities if label and sim >= threshold)
    fp = sum(1 for *_, label, sim in similarities if not label and sim >= threshold)
    fn = sum(1 for *_, label, sim in similarities if label and sim < threshold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    denominator = 0.25 * precision + recall
    f_half = 1.25 * precision * recall / denominator if denominator else 0.0
    return ThresholdScore(threshold, precision, recall, f_half)


async def calibrate_subject_threshold(
    llm, pairs: list[tuple[str, str, bool]] = SUBJECT_PAIRS, step: float = 0.01,
    min_recall: float = 0.95,
) -> Calibration:
    """Embed every pair the way recall does (singular slugs) and sweep the threshold.

    The threshold only picks candidates for the LLM to confirm, so ``best`` is
    the highest one that still keeps ``min_recall`` of the true members: higher
    means fewer labels sent to the model, lower only costs tokens.
    """
    vectors: dict[str, list[float]] = {}
    for term in {subject_slug(t) for pair in pairs for t in pair[:2]}:
        vectors[term] = await llm.embed(term)

    similarities = [
        (q, s, label, cosine_similarity(vectors[subject_slug(q)], vectors[subject_slug(s)]))
        for q, s, label in pairs
    ]
    # the config allows 0.1 … 1
    candidates = [round(0.1 + i * step, 4) for i in range(int(0.9 / step) + 1)]
    scores = [score_threshold(similarities, t) for t in candidates]
    keeps_members = [s for s in scores if s.recall >= min_recall] or scores[:1]
    best = max(keeps_members, key=lambda s: s.threshold)
    return Calibration(llm.embedding_model_id, best, scores, similarities)


async def confirmed_scores(llm, calibration: Calibration) -> ThresholdScore:
    """End-to-end precision/recall: embedding candidates at the chosen threshold,
    then the LLM confirm step recall really runs. One fast LLM call per question noun."""
    from .aggregation import confirm_members

    by_term: dict[str, list[tuple[str, bool, float]]] = {}
    for q, s, label, sim in calibration.similarities:
        by_term.setdefault(q, []).append((s, label, sim))

    outcome = []
    for term, rows in by_term.items():
        candidates = [subject_slug(s) for s, _, sim in rows if sim >= calibration.best.threshold]
        kept = set(await confirm_members(llm, term, candidates)) if candidates else set()
        outcome += [(term, s, label, 1.0 if subject_slug(s) in kept else 0.0) for s, label, _ in rows]
    return score_threshold(outcome, 0.5)
