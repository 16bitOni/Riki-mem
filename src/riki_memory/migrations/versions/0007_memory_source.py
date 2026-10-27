"""Memories record where they came from: stated by the user, inferred, or from the assistant.

Revision ID: 0007_memory_source
Revises: 0006_validity
Create Date: 2026-10-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0007_memory_source"
down_revision = "0006_validity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("source", sa.String(length=12), nullable=False, server_default="stated"))
    op.create_check_constraint("memory_source_valid", "memories", "source in ('stated','inferred','assistant')")


def downgrade() -> None:
    op.drop_constraint("memory_source_valid", "memories", type_="check")
    op.drop_column("memories", "source")
