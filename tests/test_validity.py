"""Validity over time: a changed fact stays findable as history, a temporary state
expires, and counting respects the asked-about window."""

from datetime import datetime, timezone

import pytest

from riki_memory import InstanceItem, MemoryConfig, MemoryCreate, RikiMemory


def _at(month: int, day: int = 1, year: int = 2026) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@pytest.fixture
async def mem(db_url, provider):
    m = RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig(planner_mode="heuristic"))
    await m.init()
    yield m
    await m.close()


async def _moved(mem):
    pune = await mem.save(MemoryCreate(user_id="u1", content="Sub lives in Pune.", kind="fact"))
    blr = await mem.save(MemoryCreate(
        user_id="u1", content="Sub lives in Bangalore.", kind="fact",
        replaces=[pune.id], changed_at=_at(9, 15),
    ))
    return pune, blr


async def test_a_changed_fact_is_history_not_current(mem):
    pune, blr = await _moved(mem)
    old = await mem.get("u1", pune.id)
    assert old.is_active is False and old.valid_to == _at(9, 15)

    found, _ = await mem.recall("u1", "Which city does Sub live in, Pune or Bangalore?")
    assert [m.id for m in found] == [blr.id]


async def test_history_questions_see_the_old_fact_marked_as_past(mem):
    pune, blr = await _moved(mem)
    text, found = await mem.context("u1", "Where did I live before I moved to Bangalore? Pune?")
    assert {m.id for m in found} == {pune.id, blr.id}
    assert "Sub lives in Pune. (NO LONGER TRUE since 2026-09-15)" in text


async def test_forgotten_duplicate_and_corrected_memories_are_never_history(mem):
    forgotten = await mem.save(MemoryCreate(user_id="u1", content="Sub used to live in Pune near the river.",
                                            kind="fact", valid_to=_at(1, 1)))
    await mem.forget("u1", forgotten.id)
    typo = await mem.save(MemoryCreate(user_id="u1", content="Sub used to live in Pnue.", kind="fact"))
    await mem.edit("u1", typo.id, content="Sub used to live in Pune.")  # a correction, not a change
    found, _ = await mem.recall("u1", "Where did I use to live? Pune?")
    assert [m.content for m in found] == ["Sub used to live in Pune."]


async def test_a_temporary_state_shows_its_end_and_expires(mem):
    goa = await mem.save(MemoryCreate(user_id="u1", content="Sub is on holiday in Goa.", kind="fact",
                                      valid_to=_at(11, 1)))
    text, _ = await mem.context("u1", "Am I in Goa?")
    assert "Sub is on holiday in Goa. (until 2026-11-01)" in text

    assert await mem.expire(as_of=_at(10, 31)) == 0
    assert await mem.expire(as_of=_at(11, 2)) == 1
    assert (await mem.get("u1", goa.id)).is_active is False
    found, _ = await mem.recall("u1", "Am I in Goa?")
    assert found == []
    found, _ = await mem.recall("u1", "Was I in Goa in the past? Goa holiday")
    assert [m.id for m in found] == [goa.id]


def _concert(key: str, day: datetime) -> InstanceItem:
    return InstanceItem(kind="event", subject="concert", instance_key=key, display_name=key,
                        status="occurred", date=day)


async def test_counting_stops_at_the_end_of_the_asked_window(mem):
    for key, day in [("coldplay", _at(3, 5)), ("ar-rahman", _at(3, 20)), ("arijit", _at(4, 2))]:
        await mem.save(MemoryCreate(user_id="u1", content=f"Sub went to the {key} concert.", kind="event",
                                    event_at=day, instances=[_concert(key, day)]))
    _, report = await mem.recall("u1", "How many concerts did I go to in March 2026?")
    assert report["sql_result"] == "2"


async def test_a_state_question_about_a_month_counts_what_was_owned_then(mem):
    def plant(key, status, day):
        return MemoryCreate(user_id="u1", content=f"Sub's {key} plant is {status}.", kind="event", event_at=day,
                            instances=[InstanceItem(kind="item", subject="plant", instance_key=key,
                                                    display_name=key, status=status, date=day)])

    await mem.save(plant("money", "current", _at(7, 10)))  # bought in July: still there in August
    await mem.save(plant("tulsi", "current", _at(7, 10)))
    await mem.save(plant("areca", "current", _at(8, 20)))
    await mem.save(plant("tulsi", "former", _at(9, 5)))  # died after August
    _, report = await mem.recall("u1", "How many plants did I have in August 2026?")
    assert report["sql_result"] == "3"


async def test_a_hinglish_count_is_answered_from_the_records(mem):
    for key in ("money-plant", "snake-plant"):
        await mem.save(MemoryCreate(user_id="u1", content=f"Sub bought a {key}.", kind="event", instances=[
            InstanceItem(kind="item", subject="plant", instance_key=key, display_name=key, status="current")]))
    _, report = await mem.recall("u1", "mere paas kitne plants hai?")
    assert report["route"] == "aggregate" and report["sql_result"] == "2"


async def test_a_scheduled_interview_whose_date_passed_counts_as_done(mem):
    def interview(key, status, day):
        return MemoryCreate(user_id="u1", content=f"Sub's {key} interview ({status}).", kind="event", event_at=day,
                            instances=[InstanceItem(kind="event", subject="interview", instance_key=key,
                                                    display_name=key.title(), status=status, date=day)])

    await mem.save(interview("swiggy", "occurred", _at(8, 18)))
    await mem.save(interview("zerodha", "planned", _at(8, 24)))   # "on Monday", never confirmed
    await mem.save(interview("freshworks", "planned", _at(8, 27)))
    await mem.save(interview("google", "planned", _at(12, 1)))    # still in the future
    text, _ = await mem.context("u1", "How many interviews have I had?", at=_at(10, 27))
    assert "= 3" in text and "Zerodha | planned, date passed" in text


async def test_counting_also_shows_memories_the_records_missed(mem):
    def bought(content, key, value, subject):
        return MemoryCreate(user_id="u1", content=content, kind="event", event_at=_at(8, 1), instances=[
            InstanceItem(kind="item", subject=subject, instance_key=key, display_name=key,
                         status="occurred", numeric_value=value, unit="USD")])

    await mem.save(bought("Sub spent $450 on paint for the living room makeover.", "paint", 450, "renovation"))
    await mem.save(bought("Sub bought a sofa for the living room makeover for $1,200.", "sofa", 1200, "sofa"))
    text, _ = await mem.context("u1", "How much have I spent on the renovation for the living room makeover?")
    assert "$1,200" in text  # not in the SUM's records, but in front of the reader all the same
