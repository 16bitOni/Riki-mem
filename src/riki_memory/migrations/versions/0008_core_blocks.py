"""Core memory blocks on each namespace (identity, preferences, constraints, people, projects).

Revision ID: 0008_core_blocks
Revises: 0007_memory_source
Create Date: 2026-10-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008_core_blocks"
down_revision = "0007_memory_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("blocks", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )


def downgrade() -> None:
    op.drop_column("users", "blocks")
