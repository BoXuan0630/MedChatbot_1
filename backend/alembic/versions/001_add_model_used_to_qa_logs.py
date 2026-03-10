"""Add model_used column to qa_logs table.

Revision ID: 001
Revises:
Create Date: 2026-03-07
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [c["name"] for c in inspector.get_columns("qa_logs")]
    if "model_used" not in columns:
        op.add_column("qa_logs", sa.Column("model_used", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("qa_logs", "model_used")
