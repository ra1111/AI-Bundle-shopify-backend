"""Create embedding_cache table.

Revision ID: 20241107_000002
Revises: 20241107_000001
Create Date: 2024-11-07 00:05:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20241107_000002"
down_revision = "20241107_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("embedding_cache"):
        op.create_table(
            "embedding_cache",
            sa.Column("key", sa.String(), primary_key=True),
            sa.Column("payload", sa.Text(), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
        )


def downgrade() -> None:
    op.drop_table("embedding_cache")
