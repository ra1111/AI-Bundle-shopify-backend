"""Create generation_progress table.

Revision ID: 20241107_000001
Revises: 
Create Date: 2024-11-07 00:00:01.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20241107_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("generation_progress"):
        op.create_table(
            "generation_progress",
            sa.Column("upload_id", sa.String(), primary_key=True),
            sa.Column("shop_domain", sa.Text(), nullable=False),
            sa.Column("step", sa.Text(), nullable=False),
            sa.Column(
                "progress",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("message", sa.Text(), nullable=True),
            sa.Column(
                "metadata",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.CheckConstraint(
                "progress BETWEEN 0 AND 100",
                name="ck_generation_progress_range",
            ),
            sa.CheckConstraint(
                "status IN ('in_progress','completed','failed')",
                name="ck_generation_progress_status",
            ),
        )


def downgrade() -> None:
    op.drop_table("generation_progress")
