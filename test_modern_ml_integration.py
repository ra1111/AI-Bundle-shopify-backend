"""
Modern ML Integration Test
Tests co-visitation, bandit pricing, and feature flags
"""
import sys
from decimal import Decimal

def test_imports():
    """Test that all modern ML components can be imported"""
    print("=" * 60)
    print("MODERN ML INTEGRATION TEST")
    print("=" * 60)

    print("\n1. Testing imports...")
    try:
        from services.ml.pseudo_item2vec import (
            build_covis_vectors,
            cosine_similarity,
            compute_bundle_coherence,
            enhance_candidate_with_covisitation,
            CoVisVector
        )
        print("   ✓ pseudo_item2vec module imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import pseudo_item2vec: {e}")
        return False

    try:
        from services.pricing import PricingEngine
        print("   ✓ PricingEngine imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import PricingEngine: {e}")
        return False

    try:
        from services.ranker import WeightedLinearRanker
        print("   ✓ WeightedLinearRanker imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import WeightedLinearRanker: {e}")
        return False

    try:
        from services.feature_flags import feature_flags
        print("   ✓ feature_flags imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import feature_flags: {e}")
        return False

    return True


def test_covisitation():
    """Test co-visitation graph building"""
    print("\n2. Testing co-visitation graph...")
    try:
        from services.ml.pseudo_item2vec import build_covis_vectors, cosine_similarity

        # Mock order lines
        class MockOrderLine:
            def __init__(self, order_id, sku):
                self.order_id = order_id
                self.sku = sku

        order_lines = [
            MockOrderLine("order1", "SKU-A"),
            MockOrderLine("order1", "SKU-B"),
            MockOrderLine("order2", "SKU-A"),
            MockOrderLine("order2", "SKU-B"),
            MockOrderLine("order3", "SKU-A"),
            MockOrderLine("order3", "SKU-C"),
        ]

        # Build co-visitation vectors
        vectors = build_covis_vectors(order_lines, min_co_visits=1, max_neighbors=50)

        if len(vectors) == 0:
            print(f"   ✗ No vectors built from order lines")
            return False

        print(f"   ✓ Built {len(vectors)} co-visitation vectors")

        # Test similarity
        if "SKU-A" in vectors and "SKU-B" in vectors:
            sim = cosine_similarity(vectors["SKU-A"], vectors["SKU-B"])
            print(f"   ✓ Similarity(SKU-A, SKU-B) = {sim:.3f}")
            if sim > 0:
                print(f"   ✓ Co-visitation similarity working correctly")
            else:
                print(f"   ✗ Expected positive similarity, got {sim}")
                return False

        return True

    except Exception as e:
        print(f"   ✗ Error testing co-visitation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bandit_pricing():
    """Test bandit pricing functionality"""
    print("\n3. Testing bandit pricing...")
    try:
        from services.pricing import PricingEngine

        pricing = PricingEngine()

        # Test bandit discount selection
        candidate_discounts = [5, 10, 15, 20, 25]
        features = {
            "covis_similarity": 0.75,
            "avg_price": 50.0,
            "bundle_type": "FBT",
            "objective": "increase_aov"
        }

        selected_discount = pricing.bandit_discount_selection(
            candidate_discounts, features, epsilon=0.1
        )

        if selected_discount in candidate_discounts:
            print(f"   ✓ Bandit selected discount: {selected_discount}%")
        else:
            print(f"   ✗ Invalid discount selected: {selected_discount}")
            return False

        # Test full bandit pricing
        product_prices = {"SKU-A": Decimal("25.00"), "SKU-B": Decimal("30.00")}
        result = pricing.multi_armed_bandit_pricing(
            bundle_products=["SKU-A", "SKU-B"],
            product_prices=product_prices,
            features=features,
            objective="increase_aov"
        )

        if "bundle_price" in result and "discount_pct" in result:
            print(f"   ✓ Bandit pricing result: ${result['bundle_price']} ({result['discount_pct']}% off)")
            print(f"   ✓ Methodology: {result.get('methodology', 'unknown')}")
        else:
            print(f"   ✗ Invalid pricing result: {result}")
            return False

        return True

    except Exception as e:
        print(f"   ✗ Error testing bandit pricing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_flags():
    """Test feature flag functionality"""
    print("\n4. Testing feature flags...")
    try:
        from services.feature_flags import feature_flags

        # Test pareto optimization flag
        pareto_enabled = feature_flags.get_flag("advanced.pareto_optimization", True)
        print(f"   ✓ Pareto optimization flag: {pareto_enabled}")

        if pareto_enabled == False:
            print(f"   ✓ Pareto optimization disabled by default (modern/fast mode)")
        else:
            print(f"   ⚠ Pareto optimization enabled (slower but more thorough)")

        # Test other relevant flags
        covis_flag = feature_flags.get_flag("bundling.v2_pipeline", True)
        print(f"   ✓ V2 pipeline enabled: {covis_flag}")

        return True

    except Exception as e:
        print(f"   ✗ Error testing feature flags: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_candidate_enrichment():
    """Test candidate enrichment with co-visitation features"""
    print("\n5. Testing candidate enrichment...")
    try:
        from services.ml.pseudo_item2vec import build_covis_vectors, enhance_candidate_with_covisitation

        # Mock order lines
        class MockOrderLine:
            def __init__(self, order_id, sku):
                self.order_id = order_id
                self.sku = sku

        order_lines = [
            MockOrderLine("order1", "SKU-A"),
            MockOrderLine("order1", "SKU-B"),
            MockOrderLine("order2", "SKU-A"),
            MockOrderLine("order2", "SKU-B"),
        ]

        vectors = build_covis_vectors(order_lines)

        # Mock candidate
        candidate = {
            "products": [{"sku": "SKU-A"}, {"sku": "SKU-B"}],
            "features": {}
        }

        # Enrich candidate
        enhanced = enhance_candidate_with_covisitation(candidate, vectors)

        if "covis_similarity" in enhanced.get("features", {}):
            sim = enhanced["features"]["covis_similarity"]
            print(f"   ✓ Candidate enriched with covis_similarity: {sim:.3f}")

            if "covis_min_similarity" in enhanced["features"] and "covis_max_similarity" in enhanced["features"]:
                print(f"   ✓ Min/Max similarity features present")

            return True
        else:
            print(f"   ✗ Candidate not enriched properly: {enhanced}")
            return False

    except Exception as e:
        print(f"   ✗ Error testing candidate enrichment: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    results = []

    results.append(("Import Test", test_imports()))

    if results[0][1]:  # Only run other tests if imports succeed
        results.append(("Co-visitation Test", test_covisitation()))
        results.append(("Bandit Pricing Test", test_bandit_pricing()))
        results.append(("Feature Flags Test", test_feature_flags()))
        results.append(("Candidate Enrichment Test", test_candidate_enrichment()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Modern ML integration is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
