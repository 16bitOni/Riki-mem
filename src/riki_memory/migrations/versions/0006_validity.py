"""Memories get a validity end and a forgotten marker: changed facts stay searchable
as history, temporary states expire, and forgotten memories are never history.

Revision ID: 0006_validity
Revises: 0005_turns
Create Date: 2026-10-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_validity"
down_revision = "0005_turns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True))
    op.add_column("memories", sa.Column("forgotten_at", sa.DateTime(timezone=True), nullable=True))
    # Before this revision, inactive memories that nothing replaced were forgotten.
    op.execute("UPDATE memories SET forgotten_at = updated_at WHERE NOT is_active AND superseded_by_id IS NULL")


def downgrade() -> None:
    op.drop_column("memories", "forgotten_at")
    op.drop_column("memories", "valid_to")
