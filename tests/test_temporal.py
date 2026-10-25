"""Tests for temporal anchor extraction and date arithmetic."""

import pytest
from datetime import datetime, timezone

from riki_memory.heuristic import (
    extract_temporal_anchors,
    is_temporal_arithmetic,
)


# ---------- temporal arithmetic detection ----------


def test_is_temporal_arithmetic_days_before():
    assert is_temporal_arithmetic("How many days before the wedding did I buy the suit?")


def test_is_temporal_arithmetic_days_between():
    assert is_temporal_arithmetic("How many days between my first and second doctor visit?")


def test_is_temporal_arithmetic_months_since():
    assert is_temporal_arithmetic("How many months since I started the new job?")


def test_is_temporal_arithmetic_negative_counting():
    assert not is_temporal_arithmetic("How many weddings have I attended?")


def test_is_temporal_arithmetic_negative_listing():
    assert not is_temporal_arithmetic("List all my instruments.")


# ---------- anchor extraction ----------


def test_extract_between_anchors():
    anchors = extract_temporal_anchors(
        "How many days between the Rachel Mike wedding and the Emily Sarah wedding?"
    )
    assert len(anchors) == 2
    assert "Rachel Mike wedding" in anchors[0]
    assert "Emily Sarah wedding" in anchors[1]


def test_extract_before_anchor():
    anchors = extract_temporal_anchors(
        "How many days before the Brookside offer did I visit the first property?"
    )
    assert len(anchors) >= 1
    assert "Brookside offer" in anchors[0]


def test_extract_since_anchor():
    anchors = extract_temporal_anchors(
        "How many months since the knee surgery did I start running again?"
    )
    assert len(anchors) >= 1
    assert "knee surgery" in anchors[0]


def test_extract_no_anchors_from_counting():
    anchors = extract_temporal_anchors("How many instruments do I own?")
    assert len(anchors) == 0


def test_extract_time_on_weekday():
    anchors = extract_temporal_anchors(
        "What time on Tuesdays do I usually have my meeting?"
    )
    assert len(anchors) >= 1


# ---------- date difference computation (pure Python, no DB) ----------


def test_date_difference_days():
    """Verify that two datetimes produce the correct day delta."""
    d1 = datetime(2023, 3, 10, tzinfo=timezone.utc)
    d2 = datetime(2023, 3, 17, tzinfo=timezone.utc)
    diff = abs((d2 - d1).days)
    assert diff == 7


def test_date_difference_across_months():
    d1 = datetime(2023, 1, 15, tzinfo=timezone.utc)
    d2 = datetime(2023, 4, 15, tzinfo=timezone.utc)
    diff = abs((d2 - d1).days)
    assert diff == 90


def test_date_difference_order_independent():
    d1 = datetime(2023, 6, 1, tzinfo=timezone.utc)
    d2 = datetime(2023, 6, 20, tzinfo=timezone.utc)
    assert abs((d1 - d2).days) == abs((d2 - d1).days) == 19
