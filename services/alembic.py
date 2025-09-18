# versions/20250917_make_nullable_and_adjust_numeric.py
from alembic import op
import sqlalchemy as sa

revision = "20250917_nullable_numeric"
down_revision = None

def upgrade():
    with op.batch_alter_table("csv_uploads") as b:
        b.alter_column("run_id", existing_type=sa.String(), nullable=True)
        b.alter_column("shop_id", existing_type=sa.String(), nullable=True)
        b.alter_column("csv_type", existing_type=sa.String(), nullable=True)
        b.alter_column("error_message", existing_type=sa.Text(), nullable=True)

    # numeric widenings (safe, up only)
    for tbl, col, p, s in [
        ("order_lines", "cost", 12, 3),
        ("products", "weight_kg", 8, 2),
        ("bundle_recommendations", "confidence", 12, 6),
        ("bundle_recommendations", "predicted_lift", 12, 6),
        ("bundle_recommendations", "ranking_score", 12, 6),
        ("bundle_recommendations", "support", 12, 6),
        ("bundle_recommendations", "lift", 12, 6),
    ]:
        op.alter_column(tbl, col, type_=sa.Numeric(p, s), existing_type=sa.Numeric())

def downgrade():
    pass