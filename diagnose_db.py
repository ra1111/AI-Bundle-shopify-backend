#!/usr/bin/env python3
"""
Diagnostic script to check CockroachDB connection and bundle data
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def diagnose():
    # Import after dotenv loads
    from database import engine, BundleRecommendation
    from sqlalchemy import select, func, text
    from sqlalchemy.ext.asyncio import AsyncSession

    print("=" * 60)
    print("CockroachDB Diagnostics")
    print("=" * 60)
    print()

    # Test connection
    print("1. Testing database connection...")
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to: {version[:100]}...")
            print()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Check bundle_recommendations table
    print("2. Checking bundle_recommendations table...")
    async with AsyncSession(engine) as session:
        try:
            # Count total bundles
            result = await session.execute(select(func.count(BundleRecommendation.id)))
            total_count = result.scalar()
            print(f"   Total bundles: {total_count}")

            # Count by type
            result = await session.execute(
                select(
                    BundleRecommendation.bundle_type,
                    func.count(BundleRecommendation.id)
                ).group_by(BundleRecommendation.bundle_type)
            )
            type_counts = result.all()
            print(f"   By type:")
            for bundle_type, count in type_counts:
                print(f"     - {bundle_type}: {count}")

            # Count by approval status
            result = await session.execute(
                select(
                    BundleRecommendation.is_approved,
                    func.count(BundleRecommendation.id)
                ).group_by(BundleRecommendation.is_approved)
            )
            approval_counts = result.all()
            print(f"   By approval:")
            for is_approved, count in approval_counts:
                status = "approved" if is_approved else "not approved"
                print(f"     - {status}: {count}")

            # Count by shop
            result = await session.execute(
                select(
                    BundleRecommendation.shop_id,
                    func.count(BundleRecommendation.id)
                ).group_by(BundleRecommendation.shop_id)
            )
            shop_counts = result.all()
            print(f"   By shop:")
            for shop_id, count in shop_counts:
                print(f"     - {shop_id}: {count}")

            print()

            # Sample bundles
            print("3. Sample bundle data:")
            result = await session.execute(
                select(BundleRecommendation)
                .where(BundleRecommendation.is_approved == True)
                .limit(3)
            )
            bundles = result.scalars().all()

            for i, bundle in enumerate(bundles, 1):
                print(f"\n   Bundle {i}:")
                print(f"     ID: {bundle.id}")
                print(f"     Shop: {bundle.shop_id}")
                print(f"     Type: {bundle.bundle_type}")
                print(f"     Objective: {bundle.objective}")
                print(f"     Approved: {bundle.is_approved}")
                print(f"     Confidence: {bundle.confidence}")
                print(f"     Products: {list(bundle.products.keys()) if bundle.products else 'None'}")

            print()
            print("=" * 60)
            print("Diagnosis complete!")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Query failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose())
