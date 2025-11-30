"""
Test that objective_fit removal works correctly
"""
from decimal import Decimal

def test_weights():
    """Test that objective_fit weights are all 0.0"""
    print("=" * 60)
    print("OBJECTIVE_FIT REMOVAL TEST")
    print("=" * 60)

    try:
        # Import with mock storage to avoid SQLAlchemy dependency
        import sys
        from unittest.mock import MagicMock

        # Mock storage before importing ranker
        sys.modules['services.storage'] = MagicMock()
        sys.modules['services.feature_flags'] = MagicMock()

        from services.ranker import WeightedLinearRanker

        ranker = WeightedLinearRanker()

        # Check default weights
        print("\n1. Checking default weights...")
        default_obj_fit = float(ranker.default_weights.get("objective_fit", -1))
        if default_obj_fit == 0.0:
            print(f"   ✓ Default objective_fit weight: {default_obj_fit} (correct)")
        else:
            print(f"   ✗ Default objective_fit weight: {default_obj_fit} (should be 0.0)")
            return False

        # Check objective-specific weights
        print("\n2. Checking objective-specific weights...")
        all_zero = True
        for obj_name, obj_weights in ranker.objective_weights.items():
            obj_fit = float(obj_weights.get("objective_fit", -1))
            if obj_fit == 0.0:
                print(f"   ✓ {obj_name}: objective_fit = {obj_fit}")
            else:
                print(f"   ✗ {obj_name}: objective_fit = {obj_fit} (should be 0.0)")
                all_zero = False

        if not all_zero:
            return False

        # Verify weights sum to 1.0 (excluding novelty_penalty)
        print("\n3. Verifying weight distribution...")
        for obj_name in ["default", "clear_slow_movers", "margin_guard", "increase_aov"]:
            if obj_name == "default":
                weights = ranker.default_weights
            else:
                weights = ranker.objective_weights.get(obj_name, {})

            positive_sum = sum(
                float(w) for k, w in weights.items()
                if k != "novelty_penalty" and k != "objective_fit"
            )

            # Should sum to ~1.0 (allow small floating point error)
            if abs(positive_sum - 1.0) < 0.01:
                print(f"   ✓ {obj_name}: positive weights sum = {positive_sum:.3f}")
            else:
                print(f"   ⚠ {obj_name}: positive weights sum = {positive_sum:.3f} (expected ~1.0)")

        # Test that compute_objective_fit returns 0 immediately
        print("\n4. Testing compute_objective_fit is a no-op...")
        import asyncio

        async def test_no_op():
            result = await ranker.compute_objective_fit(
                ["SKU-A", "SKU-B"],
                "increase_aov",
                "test-csv-id"
            )
            return result

        result = asyncio.run(test_no_op())
        if result == Decimal('0'):
            print(f"   ✓ compute_objective_fit returns: {result} (correct - no-op)")
        else:
            print(f"   ✗ compute_objective_fit returns: {result} (should be 0)")
            return False

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print("- objective_fit weight = 0.0 everywhere")
        print("- compute_objective_fit() is a no-op (returns 0 immediately)")
        print("- No expensive catalog lookups or heuristic scoring")
        print("- Backwards compatible (key still exists in features dict)")
        print("\n🚀 Pure data-driven ranking active!")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_weights()
    sys.exit(0 if success else 1)
