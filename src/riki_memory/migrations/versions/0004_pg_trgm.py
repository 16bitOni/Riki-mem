"""Enable pg_trgm: near-duplicate checks run in SQL instead of loading memories into Python.

Revision ID: 0004_pg_trgm
Revises: 0003_memory_attributes
Create Date: 2026-10-27
"""

from alembic import op

revision = "0004_pg_trgm"
down_revision = "0003_memory_attributes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def downgrade() -> None:
    pass  # other code in the database may use pg_trgm; never drop it
