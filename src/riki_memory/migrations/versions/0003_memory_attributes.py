"""Memories get attributes: the app's own structured fields, filterable.

Revision ID: 0003_memory_attributes
Revises: 0002_namespaces
Create Date: 2026-10-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_memory_attributes"
down_revision = "0002_namespaces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("memory_attributes", "memories", ["attributes"], unique=False, postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("memory_attributes", table_name="memories")
    op.drop_column("memories", "attributes")
