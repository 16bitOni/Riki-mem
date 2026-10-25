"""Baseline: the schema as it stood when migrations were introduced.

Databases created earlier by ``create_all`` are stamped at this revision
(see ``riki_memory.migrate.upgrade``) rather than re-created.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-25 21:00:09.078922
"""

import sqlalchemy as sa
from alembic import op

import riki_memory.vectors


revision = '0001_baseline'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")  # the embedding columns need the type
    op.create_table('users',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('timezone', sa.String(length=64), nullable=False),
    sa.Column('profile', sa.Text(), nullable=False),
    sa.Column('profile_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('entities',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('type', sa.String(length=20), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("type in ('person','place','org','project','other')", name='entity_type_valid'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('entity_user_name_unique', 'entities', ['user_id', sa.literal_column('lower(name)')], unique=True)
    op.create_index(op.f('ix_entities_user_id'), 'entities', ['user_id'], unique=False)
    op.create_table('episodes',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('session_key', sa.String(length=512), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('date', sa.DateTime(timezone=True), nullable=True),
    sa.Column('embedding', riki_memory.vectors.Embedding(), nullable=True),
    sa.Column('embedding_model', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'session_key', name='episode_user_session_unique')
    )
    op.create_index('episode_user_created', 'episodes', ['user_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_episodes_user_id'), 'episodes', ['user_id'], unique=False)
    op.create_table('memories',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('importance', sa.Integer(), nullable=False),
    sa.Column('event_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('source_key', sa.String(length=512), nullable=True),
    sa.Column('embedding', riki_memory.vectors.Embedding(), nullable=True),
    sa.Column('wants_embedding', sa.Boolean(), nullable=False),
    sa.Column('embedding_model', sa.String(length=100), nullable=True),
    sa.Column('embedded_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('tags', sa.JSON(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('superseded_by_id', sa.String(length=36), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("kind in ('fact','preference','event','episode')", name='memory_kind_valid'),
    sa.CheckConstraint('embedding is null or embedding_model is not null', name='memory_vector_names_its_model'),
    sa.CheckConstraint('importance between 1 and 5', name='memory_importance_range'),
    sa.CheckConstraint('length(content) > 0', name='memory_content_not_empty'),
    sa.ForeignKeyConstraint(['superseded_by_id'], ['memories.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'content_hash', name='memory_user_content_unique')
    )
    op.create_index(op.f('ix_memories_user_id'), 'memories', ['user_id'], unique=False)
    op.create_index('memory_source_key', 'memories', ['source_key'], unique=False)
    op.create_index('memory_user_active_created', 'memories', ['user_id', 'is_active', 'created_at'], unique=False)
    op.create_index('memory_user_event', 'memories', ['user_id', 'event_at'], unique=False)
    op.create_table('entity_relations',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('subject_id', sa.String(length=36), nullable=False),
    sa.Column('predicate', sa.String(length=100), nullable=False),
    sa.Column('target_id', sa.String(length=36), nullable=False),
    sa.ForeignKeyConstraint(['subject_id'], ['entities.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['target_id'], ['entities.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('subject_id', 'predicate', 'target_id', name='relation_unique')
    )
    op.create_index(op.f('ix_entity_relations_user_id'), 'entity_relations', ['user_id'], unique=False)
    op.create_index('relation_target', 'entity_relations', ['target_id'], unique=False)
    op.create_table('episode_memories',
    sa.Column('episode_id', sa.String(length=36), nullable=False),
    sa.Column('memory_id', sa.String(length=36), nullable=False),
    sa.ForeignKeyConstraint(['episode_id'], ['episodes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['memory_id'], ['memories.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('episode_id', 'memory_id')
    )
    op.create_index('episode_memory_memory', 'episode_memories', ['memory_id'], unique=False)
    op.create_table('instances',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('subject', sa.String(length=100), nullable=False),
    sa.Column('instance_key', sa.String(length=500), nullable=False),
    sa.Column('display_name', sa.String(length=500), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('date', sa.DateTime(timezone=True), nullable=True),
    sa.Column('numeric_value', sa.Double(), nullable=True),
    sa.Column('unit', sa.String(length=50), nullable=True),
    sa.Column('source_memory_id', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("status in ('occurred','planned','desired','former','current')", name='instance_status_valid'),
    sa.ForeignKeyConstraint(['source_memory_id'], ['memories.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'kind', 'subject', 'instance_key', 'source_memory_id', name='instance_user_kind_subject_key_memory_unique')
    )
    op.create_index('instance_user_subject', 'instances', ['user_id', 'subject'], unique=False)
    op.create_index(op.f('ix_instances_source_memory_id'), 'instances', ['source_memory_id'], unique=False)
    op.create_index(op.f('ix_instances_user_id'), 'instances', ['user_id'], unique=False)
    op.create_table('memory_entities',
    sa.Column('memory_id', sa.String(length=36), nullable=False),
    sa.Column('entity_id', sa.String(length=36), nullable=False),
    sa.ForeignKeyConstraint(['entity_id'], ['entities.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['memory_id'], ['memories.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('memory_id', 'entity_id')
    )
    op.create_index('memory_entity_entity', 'memory_entities', ['entity_id'], unique=False)
    # Keyword search (keyword.py); an expression index, so the models can't describe it.
    op.execute(
        "CREATE INDEX IF NOT EXISTS memory_fts_idx ON memories USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS memory_fts_idx")
    op.drop_index('memory_entity_entity', table_name='memory_entities')
    op.drop_table('memory_entities')
    op.drop_index(op.f('ix_instances_user_id'), table_name='instances')
    op.drop_index(op.f('ix_instances_source_memory_id'), table_name='instances')
    op.drop_index('instance_user_subject', table_name='instances')
    op.drop_table('instances')
    op.drop_index('episode_memory_memory', table_name='episode_memories')
    op.drop_table('episode_memories')
    op.drop_index('relation_target', table_name='entity_relations')
    op.drop_index(op.f('ix_entity_relations_user_id'), table_name='entity_relations')
    op.drop_table('entity_relations')
    op.drop_index('memory_user_event', table_name='memories')
    op.drop_index('memory_user_active_created', table_name='memories')
    op.drop_index('memory_source_key', table_name='memories')
    op.drop_index(op.f('ix_memories_user_id'), table_name='memories')
    op.drop_table('memories')
    op.drop_index(op.f('ix_episodes_user_id'), table_name='episodes')
    op.drop_index('episode_user_created', table_name='episodes')
    op.drop_table('episodes')
    op.drop_index(op.f('ix_entities_user_id'), table_name='entities')
    op.drop_index('entity_user_name_unique', table_name='entities')
    op.drop_table('entities')
    op.drop_table('users')
