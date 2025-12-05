"""Add partial unique indexes for order_lines upserts.

Revision ID: 20251203_000003
Revises: d019977d685c
Create Date: 2025-12-03 11:00:00
"""

from alembic import op
import sqlalchemy as sa
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "20251203_000003"
down_revision: Union[str, Sequence[str], None] = "d019977d685c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Ensure partial unique indexes exist for order_lines upserts."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    def _index_exists(name: str) -> bool:
        try:
            return any(ix.get("name") == name for ix in inspector.get_indexes("order_lines"))
        except Exception:
            return False

    if not _index_exists("uq_order_lines_with_line_item_id"):
        op.create_index(
            "uq_order_lines_with_line_item_id",
            "order_lines",
            ["order_id", "line_item_id"],
            unique=True,
            postgresql_where=sa.text("line_item_id IS NOT NULL"),
        )

    if not _index_exists("uq_order_lines_without_line_item_id"):
        op.create_index(
            "uq_order_lines_without_line_item_id",
            "order_lines",
            ["order_id", "sku"],
            unique=True,
            postgresql_where=sa.text("line_item_id IS NULL"),
        )


def downgrade() -> None:
    """Drop partial unique indexes if they exist."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    def _index_exists(name: str) -> bool:
        try:
            return any(ix.get("name") == name for ix in inspector.get_indexes("order_lines"))
        except Exception:
            return False

    if _index_exists("uq_order_lines_without_line_item_id"):
        op.drop_index("uq_order_lines_without_line_item_id", table_name="order_lines")

    if _index_exists("uq_order_lines_with_line_item_id"):
        op.drop_index("uq_order_lines_with_line_item_id", table_name="order_lines")
