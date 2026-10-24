"""Keyword backend selection and the Postgres query shape.

No live Postgres is available in CI, so the Postgres side is verified by
compiling the generated SQL against the Postgres dialect (catches bad
function names, bad operators, bad quoting). The SQLite side is covered by
test_store.py's keyword tests, which now run through the same dispatcher.
"""

from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql

from riki_memory import PostgresTSIndex, SQLiteFTSIndex, select_keyword_index
from riki_memory.keyword import build_fts_query


def test_backend_selection_by_url():
    assert isinstance(select_keyword_index("sqlite+aiosqlite:///m.db"), SQLiteFTSIndex)
    assert isinstance(select_keyword_index("sqlite+aiosqlite:///:memory:"), SQLiteFTSIndex)
    assert isinstance(select_keyword_index("postgresql+asyncpg://u:p@host/db"), PostgresTSIndex)
    assert isinstance(select_keyword_index("postgresql://u:p@host/db"), PostgresTSIndex)


def test_backend_names():
    assert SQLiteFTSIndex().name == "sqlite-fts5"
    assert PostgresTSIndex().name == "postgres-tsvector"


def test_fts_query_shape():
    assert build_fts_query([]) is None
    assert build_fts_query(["  "]) is None
    assert build_fts_query(["riya", "zerodha"]) == '"riya" OR "zerodha"'
    # quotes are stripped and re-added, so phrases stay phrases
    assert build_fts_query(['"new york"']) == '"new york"'


def test_postgres_search_sql_compiles():
    stmt = PostgresTSIndex()._search_stmt(
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
    index = PostgresTSIndex()
    assert build_fts_query([]) is None
