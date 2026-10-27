"""Alembic environment for riki-memory's own tables.

Run by :func:`riki_memory.migrate.upgrade` on the library's connection; there is
no alembic.ini. The version table has its own name so it never collides with
a host application that also uses Alembic in the same database, and only
riki-memory's tables are ever compared or touched (see ``migrate.include_object``).
"""

from alembic import context
from sqlalchemy import text

from riki_memory.migrate import VERSION_TABLE, include_object
from riki_memory.models import Base

connection = context.config.attributes["connection"]
context.configure(
    connection=connection,
    # riki's own schema, never whichever version table search_path happens to find first
    version_table_schema=connection.execute(text("SELECT current_schema()")).scalar(),
    target_metadata=Base.metadata,
    version_table=VERSION_TABLE,
    include_object=include_object,
    compare_type=True,
)
with context.begin_transaction():
    context.run_migrations()
