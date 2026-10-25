"""Namespaces get an owner and a kind: tenant_id and kind on users.

Groundwork for company use (see Docs/SCOPE_MODEL.md). Existing rows become
tenant "default", kind "user", which is what they always were.

Revision ID: 0002_namespaces
Revises: 0001_baseline
Create Date: 2026-10-25
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_namespaces"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("tenant_id", sa.String(length=64), nullable=False, server_default="default"))
    op.add_column("users", sa.Column("kind", sa.String(length=20), nullable=False, server_default="user"))
    op.create_index(op.f("ix_users_tenant_id"), "users", ["tenant_id"], unique=False)
    op.create_check_constraint("user_kind_valid", "users", "kind in ('user','agent','team','org')")


def downgrade() -> None:
    op.drop_constraint("user_kind_valid", "users", type_="check")
    op.drop_index(op.f("ix_users_tenant_id"), table_name="users")
    op.drop_column("users", "kind")
    op.drop_column("users", "tenant_id")
