"""Phase 3 read paths: status families, sums, subject matching, temporal anchors,
the computed block in context(), and result dedup. No LLM call anywhere."""

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import update

from riki_memory import InstanceItem, MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.aggregation import (
    AggregationQuery,
    execute_aggregation,
    infer_status,
    parse_aggregation_query,
    question_unit,
)
from riki_memory.models import Memory
from riki_memory.recall import _dedup
from riki_memory.schemas import RetrievalPlan
from riki_memory.temporal import date_anchor, difference
from riki_memory.store import save_memory


@pytest.fixture
async def mem(db_url, provider):
    m = RikiMemory(
        database_url=db_url,
        llm=provider,
        config=MemoryConfig(planner_mode="heuristic", rerank="off"),
    )
    await m.init()
    yield m
    await m.close()


def _at(month: int, day: int = 1) -> datetime:
    return datetime(2025, month, day, tzinfo=timezone.utc)


async def _save(mem, provider, content, *instances, event_at=None, kind="event"):
    return await save_memory(mem.db, provider, mem.config, MemoryCreate(
        user_id="u1", content=content, kind=kind, importance=3, event_at=event_at,
        instances=list(instances),
    ))


def _inst(subject, key, status="occurred", kind="item", **extra):
    return InstanceItem(kind=kind, subject=subject, instance_key=key,
                        display_name=key.replace("-", " ").title(), status=status, **extra)


async def _run(mem, op, terms, **query):
    query.setdefault("status_filter", ["occurred", "current"])
    return await execute_aggregation(
        mem.db, "u1", AggregationQuery(operation=op, terms=terms, **query), llm=mem.llm,
    )


# ---------- R4: status phrase families ----------


@pytest.mark.parametrize("question,statuses,current", [
    ("How many instruments do I currently own?", ["current", "occurred"], True),
    ("How many guitars have I bought?", ["occurred", "current"], False),
    ("How many cars did I sell?", ["former"], False),
    ("How many bikes do I no longer have?", ["former"], False),
    ("How many trips am I planning to take?", ["planned"], False),
    ("How many countries do I want to visit?", ["desired"], False),
])
def test_status_families(question, statuses, current):
    assert infer_status(question) == (statuses, current)


# ---------- R5: sums by rows, one unit, operation chosen by unit ----------


async def test_same_key_repairs_on_different_days_both_add_up(mem, provider):
    for day, amount in [(1, 40), (20, 40)]:
        await _save(mem, provider, f"Paid ${amount} for a bike repair on March {day}.",
                    _inst("bike-repair", "bike-repair", kind="money", numeric_value=amount, unit="USD",
                          date=_at(3, day)), event_at=_at(3, day))
    result = await _run(mem, "SUM", ["bike-repair"])
    assert result.numeric_total == 80


async def test_repeat_mention_of_one_amount_counts_once(mem, provider):
    repair = _inst("bike-repair", "chain-fix", kind="money", numeric_value=40, unit="USD", date=_at(3, 2))
    await _save(mem, provider, "Paid $40 to fix the bike chain.", repair, event_at=_at(3, 2))
    await _save(mem, provider, "The chain fix cost me forty dollars.", repair, event_at=_at(3, 2), kind="fact")
    result = await _run(mem, "SUM", ["bike-repair"])
    assert result.numeric_total == 40
    assert [why for _, why in result.excluded] == ["repeat mention, counted once"]


async def test_dollars_and_hours_never_mix(mem, provider):
    await _save(mem, provider, "Spent $120 on the road trip.",
                _inst("road-trip", "goa-trip", kind="trip", numeric_value=120, unit="USD"))
    await _save(mem, provider, "Drove 6 hours on the road trip.",
                _inst("road-trip", "goa-drive", kind="trip", numeric_value=6, unit="hours"))
    result = await _run(mem, "SUM", ["road-trip"], unit="hours")
    assert (result.numeric_total, result.unit) == (6, "hours")
    assert [why for _, why in result.excluded] == ["in USD, not hours"]


async def test_how_many_fish_sums_the_stated_counts(mem, provider):
    await _save(mem, provider, "I have 6 neon tetras.", _inst("fish", "neon-tetra", "current", numeric_value=6))
    await _save(mem, provider, "I have 4 guppies.", _inst("fish", "guppy", "current", numeric_value=4))
    result = await _run(mem, "COUNT_DISTINCT", ["fish"])
    assert (result.operation, result.value) == ("SUM", "10")


def test_question_unit_turns_count_into_sum():
    plan = RetrievalPlan(needs_memory=True, tags=["driving"])
    query = parse_aggregation_query("How many hours did I spend driving?", plan)
    assert (query.operation, query.unit) == ("SUM", "hours")
    assert question_unit("How much money did I raise?") == "USD"
    assert question_unit("How many weddings did I attend?") is None


# ---------- R2: meaning-level subject match ----------


class Confirms:
    """Fake embeddings, plus a canned answer to the subject-confirm question."""

    def __init__(self, provider, members):
        self._provider = provider
        self.members = members
        self.asked: list[str] = []
        self.embedding_model_id = provider.embedding_model_id

    async def embed(self, text):
        return await self._provider.embed(text)

    async def chat_json(self, system, user, schema, *, fast=True):
        if self.members is None:
            raise RuntimeError("model unavailable")
        self.asked.append(user)
        return schema(members=self.members)


async def _by_meaning(mem, llm, terms):
    return await execute_aggregation(
        mem.db, "u1",
        AggregationQuery(operation="COUNT_DISTINCT", terms=terms, status_filter=["occurred", "current"]),
        llm=llm, subject_threshold=0.5,
    )


async def _pets(mem, provider):
    # Fake embeddings: "dogs" = "doggo" (1.0); "paris-dog-walker" is half dog, half paris (0.71).
    await _save(mem, provider, "Adopted Bruno.", _inst("doggo", "bruno", "current"))
    await _save(mem, provider, "Hired a dog walker in Paris.", _inst("paris-dog-walker", "sam", "current", kind="person"))
    await _save(mem, provider, "Went to Priya's wedding.", _inst("wedding", "priya", kind="event"))


async def test_subject_matched_by_meaning_when_no_word_matches(mem, provider):
    await _pets(mem, provider)
    llm = Confirms(provider, ["doggo"])
    result = await _by_meaning(mem, llm, ["dogs"])
    assert (result.subjects, result.count) == (["doggo"], 1)
    # only the embedding candidates were shown, not the unrelated wedding
    assert "wedding" not in llm.asked[0] and "paris-dog-walker" in llm.asked[0]


async def test_meaning_match_only_keeps_what_the_model_confirms(mem, provider):
    await _pets(mem, provider)
    # a made-up label the model returns is ignored too
    result = await _by_meaning(mem, Confirms(provider, ["dog-walker"]), ["dogs"])
    assert (result.subjects, result.value) == ([], "0")


async def test_meaning_match_fails_closed_without_the_model(mem, provider):
    await _pets(mem, provider)
    result = await _by_meaning(mem, Confirms(provider, None), ["dogs"])
    assert result.subjects == []


# ---------- R6 + R8: computed block, never empty when something was counted ----------


async def _instruments(mem, provider):
    await _save(mem, provider, "Bought a Fender Stratocaster.", _inst("instrument", "fender"), event_at=_at(1))
    await _save(mem, provider, "Bought a Korg piano.", _inst("instrument", "korg"), event_at=_at(2))
    await _save(mem, provider, "Bought a Pearl drum set.", _inst("instrument", "pearl-drums"), event_at=_at(3))
    await _save(mem, provider, "Sold the Pearl drum set.", _inst("instrument", "pearl-drums", "former"),
                event_at=_at(6))
    await _save(mem, provider, "Wants a Kala ukulele someday.", _inst("instrument", "kala-ukulele", "desired"),
                kind="preference")


async def test_context_shows_the_count_with_inclusions_and_exclusions(mem, provider):
    await _instruments(mem, provider)
    text, memories = await mem.context("u1", "How many instruments do I currently own?")

    assert "## Computed from memory" in text
    assert "COUNT of instrument" in text and "= 2" in text
    counted, left_out = text.split("Left out:")
    assert "Fender" in counted and "Korg" in counted and "Pearl" not in counted
    assert "Pearl Drums | former" in left_out and "Kala Ukulele | desired" in left_out
    assert text.index("## Computed") < text.index("## Memories that may be relevant")
    assert memories


async def test_recall_report_carries_route_and_computed(mem, provider):
    await _instruments(mem, provider)
    found, report = await mem.recall("u1", "How many instruments have I bought?")
    assert report["route"] == "aggregate"
    assert report["computed"]["value"] == "3"  # bought counts the drums too, sold later or not
    assert found


async def test_counted_but_unretrievable_evidence_falls_back(mem, provider):
    m = await _save(mem, provider, "I own a Yamaha guitar.", _inst("instrument", "yamaha", "current"),
                    kind="fact")
    async with mem.db.session_scope() as session:  # simulate a lost evidence row
        await session.execute(update(Memory).where(Memory.id == m.id).values(is_active=False))
    found, report = await mem.recall("u1", "How many instruments do I own?")
    assert report["sql_count"] == 1
    assert report["evidence_missing"] == [m.id]


async def test_unknown_subject_falls_back_to_ranked_search(mem, provider):
    await _save(mem, provider, "Bought a Fender Stratocaster.", _inst("instrument", "fender"))
    found, report = await mem.recall("u1", "How many boats do I own?")
    assert report["aggregation"] == "no_matching_subject"
    assert "computed" not in report


# ---------- R7: temporal anchors ----------


async def _workshop_and_meeting(mem, provider):
    await _save(mem, provider, "Attended the data science workshop.", event_at=_at(3, 10))
    await _save(mem, provider, "Had the quarterly team meeting.", event_at=_at(3, 17))


async def test_temporal_difference_end_to_end(mem, provider):
    await _workshop_and_meeting(mem, provider)
    found, report = await mem.recall(
        "u1", "How many days after the data science workshop did I have the team meeting?"
    )
    assert report["route"] == "temporal"
    assert report["computed_days"] == 7
    assert report["computed"]["anchors"][0]["date"] == "2025-03-10"


async def test_temporal_ago_uses_the_question_date(mem, provider):
    await _workshop_and_meeting(mem, provider)
    found, report = await mem.recall(
        "u1", "How many days ago did I attend the data science workshop?", not_after=_at(3, 20),
    )
    assert report["computed_days"] == 10


async def test_undated_anchor_falls_back_instead_of_guessing(mem, provider):
    await _workshop_and_meeting(mem, provider)
    found, report = await mem.recall(
        "u1", "How many days after the data science workshop did I go skydiving?"
    )
    assert report["route"] == "temporal"
    assert report["temporal"] == "anchor_unresolved"
    assert "computed" not in report


async def test_anchor_date_skips_a_decoy_about_something_else(mem, provider):
    decoy = await _save(mem, provider, "Bought snacks for the workshop.", event_at=_at(3, 1))
    real = await _save(mem, provider, "Attended the data science workshop.", event_at=_at(3, 10))
    when, source = await date_anchor(mem.db, "data science workshop", [decoy, real])
    assert source.id == real.id


@pytest.mark.parametrize("a,b,unit,expected", [
    (date(2025, 3, 10), date(2025, 3, 24), "weeks", 2.0),
    (date(2025, 1, 15), date(2025, 4, 15), "months", 3),
    (date(2025, 1, 31), date(2025, 2, 28), "months", 0),
    (date(2023, 6, 1), date(2025, 7, 1), "years", 2),
    (date(2025, 6, 20), date(2025, 6, 1), "days", 19),
])
def test_difference_units(a, b, unit, expected):
    assert difference(a, b, unit)[1] == expected


# ---------- R9: dedup ----------


async def test_dedup_keeps_distinct_same_day_and_drops_repeats(mem, provider):
    kit_a = await _save(mem, provider, "Bought a B-29 model kit.", _inst("model-kit", "b-29"), event_at=_at(6, 10))
    kit_b = await _save(mem, provider, "Bought a Camaro model kit.", _inst("model-kit", "camaro"),
                        event_at=_at(6, 10))
    again = await _save(mem, provider, "Really happy with the B-29 kit.", _inst("model-kit", "b-29"),
                        event_at=_at(6, 10), kind="preference")
    note = await _save(mem, provider, "My older sister Riya lives in Paris with her husband.", kind="fact")
    # Different kind, so save-time dedup keeps both; recall-time dedup should not.
    note_again = await _save(mem, provider, "My older sister Riya lives in Paris with her husband now.",
                             kind="preference")

    kept = await _dedup(mem.db, mem.config, [kit_a, kit_b, again, note, note_again])
    assert [m.id for m in kept] == [kit_a.id, kit_b.id, note.id]
