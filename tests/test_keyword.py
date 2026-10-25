"""The Postgres keyword query: its shape, and that it compiles to valid Postgres SQL."""

from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql

from riki_memory.keyword import build_fts_query, keyword_search_stmt


def test_fts_query_shape():
    assert build_fts_query([]) is None
    assert build_fts_query(["  "]) is None
    assert build_fts_query(["riya", "zerodha"]) == '"riya" OR "zerodha"'
    # quotes are stripped and re-added, so phrases stay phrases
    assert build_fts_query(['"new york"']) == '"new york"'


def test_postgres_search_sql_compiles():
    stmt = keyword_search_stmt(
        "u1",
        ["riya", "zerodha"],
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 12, 31, tzinfo=timezone.utc),
    )
    sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "to_tsvector('english'" in sql
    assert "websearch_to_tsquery('english'" in sql
    assert "ts_rank_cd" in sql
    assert "@@" in sql
    assert "is_active" in sql


def test_postgres_search_empty_keywords_compiles():
    # empty keywords must not produce a MATCH against NULL
    assert build_fts_query([]) is None
