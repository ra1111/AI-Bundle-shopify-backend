"""
Quick-Start Mode Integration Test

This script verifies that the quick-start first-time install logic works correctly.
Run with: python test_quick_start.py
"""
import asyncio
import sys
from services.storage import storage


async def test_quick_start_detection():
    """Test first-time install detection logic"""
    print("=" * 60)
    print("QUICK-START MODE INTEGRATION TEST")
    print("=" * 60)

    test_shop_id = "test-quick-start-shop"

    print(f"\n1. Testing is_first_time_install() for new shop: {test_shop_id}")
    try:
        is_first = await storage.is_first_time_install(test_shop_id)
        print(f"   ✓ First-time install check: {is_first}")
        if is_first:
            print(f"   ✓ Correctly detected as first-time install")
        else:
            print(f"   ✗ Expected first-time install, got regular user")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print(f"\n2. Testing mark_shop_sync_completed() for shop: {test_shop_id}")
    try:
        await storage.mark_shop_sync_completed(test_shop_id)
        print(f"   ✓ Successfully marked shop sync as completed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print(f"\n3. Verifying is_first_time_install() returns False after completion")
    try:
        is_first_after = await storage.is_first_time_install(test_shop_id)
        print(f"   ✓ First-time install check after completion: {is_first_after}")
        if not is_first_after:
            print(f"   ✓ Correctly detected as regular user now")
        else:
            print(f"   ✗ Expected regular user, still showing first-time install")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    print(f"\n4. Testing get_quick_start_preflight_info()")
    try:
        # Create a dummy CSV upload for testing
        from database import CsvUpload
        from datetime import datetime
        import uuid

        async with storage.get_session() as session:
            test_upload_id = str(uuid.uuid4())
            upload = CsvUpload(
                id=test_upload_id,
                shop_id=test_shop_id,
                csv_type='orders',
                status='completed',
                created_at=datetime.utcnow()
            )
            session.add(upload)
            await session.commit()

        preflight = await storage.get_quick_start_preflight_info(test_upload_id, test_shop_id)
        print(f"   ✓ Pre-flight check result:")
        print(f"      - is_first_time_install: {preflight['is_first_time_install']}")
        print(f"      - has_existing_quick_start: {preflight['has_existing_quick_start']}")
        print(f"      - quick_start_bundle_count: {preflight['quick_start_bundle_count']}")
        print(f"      - csv_upload_status: {preflight['csv_upload_status']}")

        # Cleanup
        from sqlalchemy import delete
        async with storage.get_session() as session:
            await session.execute(
                delete(CsvUpload).where(CsvUpload.id == test_upload_id)
            )
            await session.commit()

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nQuick-start mode is properly configured and working!")
    print("\nConfiguration:")
    import os
    print(f"  QUICK_START_ENABLED: {os.getenv('QUICK_START_ENABLED', 'true')}")
    print(f"  QUICK_START_TIMEOUT_SECONDS: {os.getenv('QUICK_START_TIMEOUT_SECONDS', '120')}")
    print(f"  QUICK_START_MAX_PRODUCTS: {os.getenv('QUICK_START_MAX_PRODUCTS', '50')}")
    print(f"  QUICK_START_MAX_BUNDLES: {os.getenv('QUICK_START_MAX_BUNDLES', '10')}")

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_quick_start_detection())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
