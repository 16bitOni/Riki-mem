"""Tests for the SQL aggregation module."""

import pytest
from datetime import datetime, timezone

from riki_memory import InstanceItem, MemoryCreate
from riki_memory.aggregation import (
    AggregationQuery,
    execute_aggregation,
    parse_aggregation_operation,
    parse_aggregation_query,
)
from riki_memory.heuristic import make_heuristic_plan
from riki_memory.store import save_memory


# ---------- query parsing ----------


def test_parse_count():
    assert parse_aggregation_operation("How many weddings have I attended?") == "COUNT_DISTINCT"


def test_parse_sum():
    assert parse_aggregation_operation("How much total have I spent on bike expenses?") == "SUM"


def test_parse_avg():
    assert parse_aggregation_operation("What is the average age of my family?") == "AVG"


def test_parse_list():
    assert parse_aggregation_operation("List all my instruments") == "LIST"


def test_parse_aggregation_query_with_tags():
    plan = make_heuristic_plan("How many weddings have I attended?", "UTC")
    q = parse_aggregation_query("How many weddings have I attended?", plan)
    assert q is not None
    assert q.operation == "COUNT_DISTINCT"
    assert "wedding" in q.category


def test_parse_aggregation_query_non_agg():
    """Non-aggregation questions return None."""
    plan = make_heuristic_plan("What is my sister's name?", "UTC")
    q = parse_aggregation_query("What is my sister's name?", plan)
    assert q is None


# ---------- SQL execution ----------


async def test_count_distinct_instances(memory, provider):
    """COUNT_DISTINCT returns the number of unique instance_keys."""
    for i, name in enumerate(["Rachel-Mike", "Emily-Sarah", "Tom-Lisa"]):
        data = MemoryCreate(
            user_id="u1",
            content=f"Attended {name} wedding.",
            kind="event",
            importance=3,
            instances=[InstanceItem(
                category="wedding-attended",
                instance_key=f"{name.lower()}-wedding",
                display_name=f"{name} wedding",
                status="occurred",
                date=datetime(2023, i + 1, 15, tzinfo=timezone.utc),
            )],
        )
        await save_memory(memory.db, provider, memory.config, data)

    query = AggregationQuery(
        operation="COUNT_DISTINCT",
        category="wedding-attended",
        status_filter=["occurred"],
    )
    result = await execute_aggregation(memory.db, "u1", query)
    assert result.count == 3
    assert result.value == "3"


async def test_sum_instances(memory, provider):
    """SUM returns the total numeric_value across unique instances."""
    expenses = [("helmet", 85.0), ("jersey", 50.0), ("repair", 50.0)]
    for name, amount in expenses:
        data = MemoryCreate(
            user_id="u1",
            content=f"Bought {name} for ${amount}.",
            kind="event",
            importance=3,
            instances=[InstanceItem(
                category="bike-expense",
                instance_key=f"bike-{name}",
                display_name=name,
                status="occurred",
                numeric_value=amount,
                unit="USD",
            )],
        )
        await save_memory(memory.db, provider, memory.config, data)

    query = AggregationQuery(
        operation="SUM",
        category="bike-expense",
        status_filter=["occurred"],
    )
    result = await execute_aggregation(memory.db, "u1", query)
    assert result.numeric_total == 185.0


async def test_status_filter_excludes_planned(memory, provider):
    """Planned instances are excluded from occurred-only queries."""
    data1 = MemoryCreate(
        user_id="u1",
        content="Attended Holiday Market in November.",
        kind="event",
        importance=3,
        instances=[InstanceItem(
            category="movie-festival",
            instance_key="holiday-market-2023",
            display_name="Holiday Market 2023",
            status="occurred",
        )],
    )
    data2 = MemoryCreate(
        user_id="u1",
        content="Planning to attend Sundance next year.",
        kind="event",
        importance=3,
        instances=[InstanceItem(
            category="movie-festival",
            instance_key="sundance-2024",
            display_name="Sundance 2024",
            status="planned",
        )],
    )
    await save_memory(memory.db, provider, memory.config, data1)
    await save_memory(memory.db, provider, memory.config, data2)

    query = AggregationQuery(
        operation="COUNT_DISTINCT",
        category="movie-festival",
        status_filter=["occurred"],
    )
    result = await execute_aggregation(memory.db, "u1", query)
    assert result.count == 1


async def test_empty_instances_returns_zero(memory, provider):
    """No matching instances returns count 0."""
    query = AggregationQuery(
        operation="COUNT_DISTINCT",
        category="wedding-attended",
        status_filter=["occurred"],
    )
    result = await execute_aggregation(memory.db, "u1", query)
    assert result.count == 0
    assert result.value == "0"
