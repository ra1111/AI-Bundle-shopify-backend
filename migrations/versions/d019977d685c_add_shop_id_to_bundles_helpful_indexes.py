"""Add shop_id to bundles + helpful indexes

Revision ID: d019977d685c
Revises: 20241107_000002
Create Date: 2025-11-04 21:40:33.405647

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd019977d685c'
down_revision: Union[str, Sequence[str], None] = '20241107_000002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
