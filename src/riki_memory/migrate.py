"""Schema migrations for riki-memory's tables (Alembic, shipped in the package).

``RikiMemory.init()`` calls :func:`upgrade`, so a database is always brought
to the schema this version of the library expects, without losing data.

- Two processes starting at once are serialised by a Postgres advisory lock.
- A database created before migrations existed (by ``create_all``) is adopted:
  if riki's tables are already there and no version is recorded, it is
  stamped at the baseline revision and upgraded from there.

Changing a model? Write a migration too (``tests/test_migrations.py`` fails
until you do)::

    python -m riki_memory.migrate revision "add memory scope" postgresql://...
"""

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

MIGRATIONS = Path(__file__).parent / "migrations"
VERSION_TABLE = "riki_schema_version"
BASELINE = "0001_baseline"
# Any 64-bit number unique to riki-memory: the advisory lock that serialises upgrades.
LOCK_ID = 0x52494B49_4D454D  # "RIKIMEM"


def include_object(obj, name, type_, reflected, compare_to) -> bool:
    """What migrations manage: riki's own tables, minus indexes built outside the models."""
    from .models import Base

    if type_ == "table":
        return name in Base.metadata.tables
    if type_ == "index" and name and (
        name.startswith(("memories_hnsw_", "episodes_hnsw_")) or name == "memory_fts_idx"
    ):
        # HNSW indexes are built at runtime per embedding model (vectors.py); the
        # full-text index is an expression index created by the baseline migration.
        return False
    return True


def _config(connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.attributes["connection"] = connection
    return config


def upgrade(connection) -> None:
    """Bring riki's tables to the latest schema. Takes a sync connection (``run_sync``)."""
    connection.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": LOCK_ID})
    inspector = inspect(connection)
    if not inspector.has_table(VERSION_TABLE) and inspector.has_table("memories"):
        command.stamp(_config(connection), BASELINE)  # made by create_all before migrations existed
    command.upgrade(_config(connection), "head")


def _scripts():
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    return ScriptDirectory.from_config(config)


def head() -> str:
    """The newest revision this version of the library ships."""
    return _scripts().get_current_head()


def revisions() -> set[str]:
    """Every revision this version of the library knows."""
    return {script.revision for script in _scripts().walk_revisions()}


def current(connection) -> str | None:
    """The revision this database is at (``None`` before the first upgrade)."""
    if not inspect(connection).has_table(VERSION_TABLE):
        return None
    return connection.execute(text(f"SELECT version_num FROM {VERSION_TABLE}")).scalar()


async def revision(message: str, database_url: str) -> None:
    """Autogenerate a migration by comparing the models with a database brought to head.

    Use a scratch database: it is upgraded first.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    from .db import async_url

    engine = create_async_engine(async_url(database_url))
    async with engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(upgrade)
        await connection.run_sync(
            lambda sync: command.revision(_config(sync), message=message, autogenerate=True)
        )
    await engine.dispose()


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) != 4 or sys.argv[1] != "revision":
        sys.exit('usage: python -m riki_memory.migrate revision "<message>" <postgres url>')
    asyncio.run(revision(sys.argv[2], sys.argv[3]))
