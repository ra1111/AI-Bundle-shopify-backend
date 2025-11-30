#!/usr/bin/env python3
"""
Debug script to understand why quick-start generated 0 bundles
"""

# Simulate the SKU pairs we know exist
sku_pairs = {
    ('sku-main-a', 'sku-main-b'): 8,
    ('sku-main-a', 'sku-main-c'): 5,
    ('sku-main-b', 'sku-main-d'): 4,
    ('sku-main-c', 'sku-main-d'): 3,
}

# Simulate catalog (assuming it was loaded correctly from CSV)
class MockCatalogItem:
    def __init__(self, sku, price):
        self.sku = sku
        self.price = price
        self.product_title = f"Product {sku}"
        self.variant_id = f"variant-{sku}"
        self.product_id = f"product-{sku}"

catalog = {
    'sku-main-a': MockCatalogItem('sku-main-a', 30.0),
    'sku-main-b': MockCatalogItem('sku-main-b', 60.0),
    'sku-main-c': MockCatalogItem('sku-main-c', 20.0),
    'sku-main-d': MockCatalogItem('sku-main-d', 15.0),
}

# Simulate co-visitation vectors (might be empty or missing)
covis_vectors = {}  # Empty - this is likely the issue!

print("=" * 60)
print("DEBUGGING QUICK-START BUNDLE GENERATION")
print("=" * 60)

print(f"\nSKU Pairs found: {len(sku_pairs)}")
for pair, count in sku_pairs.items():
    print(f"  {pair}: {count} times")

print(f"\nCatalog entries: {len(catalog)}")
for sku in catalog:
    print(f"  {sku}: ${catalog[sku].price}")

print(f"\nCo-visitation vectors: {len(covis_vectors)}")

print("\n" + "=" * 60)
print("SIMULATING FILTERING LOGIC")
print("=" * 60)

scored_pairs = []
for (sku1, sku2), count in sku_pairs.items():
    print(f"\nProcessing pair: {sku1} + {sku2} (count={count})")

    # Get co-visitation similarity
    covis_sim = 0.0
    if covis_vectors and sku1 in covis_vectors and sku2 in covis_vectors:
        print(f"  ✓ Co-vis similarity calculated")
    else:
        print(f"  ✗ Co-vis similarity = 0.0 (vectors not found)")

    # Check filter condition
    if covis_sim < 0.1 and count < 2:
        print(f"  ✗ FILTERED OUT: covis_sim={covis_sim} < 0.1 AND count={count} < 2")
        continue
    else:
        print(f"  ✓ PASSED FILTER: covis_sim={covis_sim} >= 0.1 OR count={count} >= 2")

    # Check catalog lookup
    p1 = catalog.get(sku1)
    p2 = catalog.get(sku2)
    if not p1 or not p2:
        print(f"  ✗ FILTERED OUT: Catalog lookup failed")
        print(f"     p1={p1}, p2={p2}")
        continue
    else:
        print(f"  ✓ Catalog lookup succeeded")

    # Check prices
    price1 = float(getattr(p1, 'price', 0) or 0)
    price2 = float(getattr(p2, 'price', 0) or 0)
    if price1 <= 0 or price2 <= 0:
        print(f"  ✗ FILTERED OUT: Invalid prices (price1={price1}, price2={price2})")
        continue
    else:
        print(f"  ✓ Price check passed (price1=${price1}, price2=${price2})")

    print(f"  ✅ PAIR ACCEPTED - Would create bundle")
    scored_pairs.append((sku1, sku2, count))

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Pairs that passed all filters: {len(scored_pairs)}")
for sku1, sku2, count in scored_pairs:
    print(f"  {sku1} + {sku2}")

if len(scored_pairs) == 0:
    print("\n⚠️  NO BUNDLES WOULD BE CREATED!")
    print("\nMost likely reason:")
    print("  - Co-visitation vectors are empty/missing")
    print("  - All pairs have covis_sim < 0.1 AND count < 2")
    print("  - But wait... all our counts are >= 2!")
    print("  - So the filter SHOULD pass...")
    print("\n🔍 Need to check actual logs for the real reason!")
