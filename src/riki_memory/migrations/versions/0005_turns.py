"""Turn log: the raw exchanges memories are extracted from (durable writes, re-extraction).

Revision ID: 0005_turns
Revises: 0004_pg_trgm
Create Date: 2026-10-27 04:32:58.485543
"""

import sqlalchemy as sa
from alembic import op


revision = '0005_turns'
down_revision = '0004_pg_trgm'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('turns',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('key', sa.String(length=512), nullable=False),
    sa.Column('user_text', sa.Text(), nullable=False),
    sa.Column('reply_text', sa.Text(), nullable=False),
    sa.Column('at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('known_ids', sa.JSON(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status in ('pending','processing','done','failed','redacted')", name='turn_status_valid'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'key', name='turn_user_key_unique')
    )
    op.create_index('turn_status_created', 'turns', ['status', 'created_at'], unique=False)
    op.create_index('turn_user_at', 'turns', ['user_id', 'at'], unique=False)


def downgrade() -> None:
    op.drop_index('turn_user_at', table_name='turns')
    op.drop_index('turn_status_created', table_name='turns')
    op.drop_table('turns')
