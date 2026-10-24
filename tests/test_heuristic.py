"""Zero-token planning: keywords, entities, dates, and the escalate mode."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from riki_memory import MemoryConfig, RikiMemory, make_heuristic_plan
from riki_memory.schemas import RetrievalPlan
from conftest import FakeProvider


def plan_at(message: str, at: datetime) -> RetrievalPlan:
    return make_heuristic_plan(message, "UTC", now=at)


def test_keywords_drop_stopwords():
    plan = make_heuristic_plan("where does my sister work these days", "UTC")
    assert "sister" in plan.keywords
    assert "work" in plan.keywords
    for stop in ("where", "does", "my", "these"):
        assert stop not in plan.keywords


def test_quoted_phrases_are_keywords():
    plan = make_heuristic_plan('remind me about "quarterly taxes" sometime', "UTC")
    assert "quarterly taxes" in plan.keywords


def test_entities_find_capitalised_names():
    plan = make_heuristic_plan("Does Riya still work at Zerodha?", "UTC")
    assert "Riya" in plan.entities
    assert "Zerodha" in plan.entities
    # sentence-start question words are not entities
    assert "Does" not in plan.entities


def test_relation_words_matched():
    plan = make_heuristic_plan("where does my sister's husband work", "UTC")
    assert "sister" in plan.relation_words
    assert "husband" in plan.relation_words


def test_semantic_query_is_raw_message():
    message = "what did we decide about the launch"
    plan = make_heuristic_plan(message, "UTC")
    assert plan.semantic_query == message


def test_pregated_plan_needs_nothing():
    plan = make_heuristic_plan("hi there", "UTC", pregated=True)
    assert plan.needs_memory is False


def test_dates_yesterday():
    at = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo("UTC"))
    plan = plan_at("what did I do yesterday", at)
    assert plan.start.date().isoformat() == "2026-10-22"
    assert plan.end.date().isoformat() == "2026-10-22"


def test_dates_last_week():
    # Friday 2026-10-23 -> last week = Mon 2026-10-12 .. Sun 2026-10-18
    at = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo("UTC"))
    plan = plan_at("summarise last week for me", at)
    assert plan.start.date().isoformat() == "2026-10-12"
    assert plan.end.date().isoformat() == "2026-10-18"


def test_dates_in_month():
    at = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo("UTC"))
    plan = plan_at("what happened in March", at)
    assert (plan.start.year, plan.start.month) == (2026, 3)
    assert (plan.end.year, plan.end.month) == (2026, 3)


def test_dates_next_friday():
    # Friday 2026-10-23 -> next Friday = 2026-10-30
    at = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo("UTC"))
    plan = plan_at("dentist next Friday", at)
    assert plan.start.date().isoformat() == "2026-10-30"


def test_dates_iso_date():
    at = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo("UTC"))
    plan = plan_at("notes from 2026-03-10", at)
    assert plan.start.date().isoformat() == "2026-03-10"


def test_dates_none_when_absent():
    plan = make_heuristic_plan("tell me a joke", "UTC")
    assert plan.start is None and plan.end is None


class StubPlanner(FakeProvider):
    """Counts planner calls; answers with a canned plan."""

    def __init__(self, plan: RetrievalPlan):
        super().__init__()
        self.canned = plan
        self.plans = 0

    async def chat_json(self, system: str, user: str, schema, *, fast: bool = True):
        self.plans += 1
        return self.canned


@pytest.fixture
async def escalating(tmp_path):
    provider = StubPlanner(RetrievalPlan(needs_memory=False))
    mem = RikiMemory(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/esc.db",
        llm=provider,
        config=MemoryConfig(planner_mode="escalate"),
    )
    await mem.init()
    yield mem, provider
    await mem.close()


async def test_escalate_hits_without_llm(escalating):
    from riki_memory.schemas import MemoryCreate

    mem, provider = escalating
    await mem.save(MemoryCreate(user_id="u1", content="Riya works at Zerodha in Bangalore."))
    found, report = await mem.recall("u1", "where does Riya work")
    assert len(found) == 1
    assert report["planner"] == "heuristic"
    assert provider.plans == 0  # zero chat tokens spent


async def test_escalate_falls_back_to_llm_on_miss(escalating):
    mem, provider = escalating
    found, report = await mem.recall("u1", "tell me about quantum tunneling")
    assert found == []
    assert report["planner"] == "llm"
    assert provider.plans == 1  # exactly one planner call, only because heuristic whiffed


async def test_heuristic_mode_never_calls_llm(tmp_path):
    from riki_memory.llm import NullProvider  # raises on any model call

    mem = RikiMemory(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/heur.db",
        llm=NullProvider(),
        config=MemoryConfig(planner_mode="heuristic"),
    )
    await mem.init()
    try:
        found, report = await mem.recall("u1", "anything at all here")
        assert report["planner"] == "heuristic"
    finally:
        await mem.close()
