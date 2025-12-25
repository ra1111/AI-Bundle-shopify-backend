#!/usr/bin/env python3
"""
Cleanup script to reset a shop's data for fresh sync testing.
Deletes all related data for a specific shop.

Usage:
  python scripts/cleanup_shop_data.py <shop_id>
  python scripts/cleanup_shop_data.py <shop_id> --dry-run

Example:
  python scripts/cleanup_shop_data.py rahular1.myshopify.com
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Ensure DATABASE_URL is set
RAW_DB_URL = os.getenv("DATABASE_URL")
if RAW_DB_URL and RAW_DB_URL.startswith("postgresql://") and "+asyncpg" not in RAW_DB_URL:
    os.environ["DATABASE_URL"] = RAW_DB_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

from sqlalchemy import delete, select, func, text, update
from database import (
    AsyncSessionLocal, BundleRecommendation, CsvUpload, GenerationProgress,
    Order, OrderLine, Variant, InventoryLevel, CatalogSnapshot, AssociationRule,
    ShopSyncStatus
)


async def cleanup_shop(shop_id: str, dry_run: bool = False):
    """Delete all data for a specific shop."""

    # Normalize shop_id (remove https://, trailing slashes)
    normalized = shop_id.replace("https://", "").replace("http://", "").rstrip("/")

    print(f"\n🔍 Looking for data matching shop_id: {normalized}")

    async with AsyncSessionLocal() as db:
        # Find matching uploads
        stmt = select(CsvUpload).where(CsvUpload.shop_id == normalized)
        result = await db.execute(stmt)
        uploads = result.scalars().all()

        if not uploads:
            print(f"\n❌ No uploads found for shop: {shop_id}")
            return

        upload_ids = [u.id for u in uploads]
        run_ids = list(set(u.run_id for u in uploads if u.run_id))

        print(f"\n📊 Found data to clean:")
        print(f"   - {len(uploads)} CSV uploads")
        print(f"   - {len(run_ids)} unique run IDs")

        # Count records in each table
        tables_to_clean = [
            ("bundle_recommendations", BundleRecommendation, "csv_upload_id"),
            ("generation_progress", GenerationProgress, "upload_id"),
            ("association_rules", AssociationRule, "csv_upload_id"),
            ("catalog_snapshot", CatalogSnapshot, "csv_upload_id"),
            ("inventory_levels", InventoryLevel, "csv_upload_id"),
            ("variants", Variant, "csv_upload_id"),
            ("order_lines", OrderLine, "csv_upload_id"),
            ("orders", Order, "csv_upload_id"),
        ]

        counts = {}
        for table_name, model, fk_column in tables_to_clean:
            fk_attr = getattr(model, fk_column)
            count_stmt = select(func.count()).select_from(model).where(fk_attr.in_(upload_ids))
            result = await db.execute(count_stmt)
            counts[table_name] = result.scalar() or 0
            if counts[table_name] > 0:
                print(f"   - {counts[table_name]} {table_name}")

        if dry_run:
            print(f"\n🔸 DRY RUN - No changes made")
            return

        # Confirm deletion
        print(f"\n⚠️  This will permanently delete all this data!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("❌ Aborted")
            return

        # Delete in order (respect foreign keys - children before parents)
        print("\n🗑️  Deleting...")

        for table_name, model, fk_column in tables_to_clean:
            if counts.get(table_name, 0) == 0:
                continue
            fk_attr = getattr(model, fk_column)
            del_stmt = delete(model).where(fk_attr.in_(upload_ids))
            result = await db.execute(del_stmt)
            print(f"   ✓ Deleted {result.rowcount} from {table_name}")

        # Finally delete CSV uploads
        del_uploads = delete(CsvUpload).where(CsvUpload.id.in_(upload_ids))
        result = await db.execute(del_uploads)
        print(f"   ✓ Deleted {result.rowcount} csv_uploads")

        # Reset shop_sync_status to allow quick-start mode on next sync
        # This is CRITICAL - without this, the system thinks bundles already exist
        # and will skip quick-start generation (which uses LLM to generate bundles)
        reset_sync_status = (
            update(ShopSyncStatus)
            .where(ShopSyncStatus.shop_id == normalized)
            .values(
                initial_sync_completed=False,
                last_sync_started_at=None,
                last_sync_completed_at=None,
            )
        )
        result = await db.execute(reset_sync_status)
        if result.rowcount > 0:
            print(f"   ✓ Reset shop_sync_status (will enable quick-start mode)")
        else:
            print(f"   ℹ No shop_sync_status record to reset")

        await db.commit()
        print(f"\n✅ Cleanup complete! Shop {shop_id} is ready for fresh sync.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    shop_id = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    asyncio.run(cleanup_shop(shop_id, dry_run))
