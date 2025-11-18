# Quick-Start Mode: Actual Logic Explained

## Overview

Quick-start mode uses **simple co-occurrence counting** - no ML, no embeddings, no complex optimization. It's designed to be **fast and lightweight** (< 2 minutes).

---

## The Algorithm (Step by Step)

### **Phase 1: Data Loading & Product Selection** (10-40% progress)

**File**: `services/bundle_generator.py:1315-1408`

#### Step 1.1: Load all order lines
```python
order_lines = await storage.get_order_lines(csv_upload_id)
# Example: 1,234 order lines from merchant's CSV
```

#### Step 1.2: Count sales by SKU (quantity-based)
```python
sku_sales = Counter()
for line in order_lines:
    sku = line.sku
    quantity = line.quantity
    sku_sales[sku] += quantity

# Example result:
# SKU-001: 145 units sold
# SKU-002: 98 units sold
# SKU-003: 87 units sold
# ...
```

#### Step 1.3: Select top N products
```python
top_skus = [sku for sku, _ in sku_sales.most_common(max_products)]  # Default: 50
# Example: Top 50 best-selling SKUs

logger.info(f"Selected top {len(top_skus)} products from {len(unique_skus)} total")
```

#### Step 1.4: Filter order lines to top products only
```python
filtered_lines = [
    line for line in order_lines
    if line.sku in top_skus
]
# Reduces dataset size for faster processing
```

**Early Exit**: If < 10 order lines or < 2 unique SKUs → return 0 bundles

---

### **Phase 2: Simple Scoring** (40-70% progress)

**File**: `services/bundle_generator.py:1418-1445`

#### Load catalog data
```python
catalog = await storage.get_catalog_snapshots_map(csv_upload_id)
# Gets product details: price, title, flags (is_slow_mover, is_high_margin)
```

#### Simple flag-based scoring
```python
product_scores = {}
for sku in top_skus:
    snapshot = catalog.get(sku)

    score = 0.5  # Base score

    # Boost slow movers (helps clear inventory)
    if snapshot.is_slow_mover:
        score += 0.3

    # Boost high-margin products (increases profit)
    if snapshot.is_high_margin:
        score += 0.2

    product_scores[sku] = score

# Example:
# SKU-001: 0.5 (normal product)
# SKU-002: 0.8 (slow mover)
# SKU-003: 1.0 (slow mover + high margin)
```

**No ML here** - just simple IF statements based on inventory flags!

---

### **Phase 3: Co-Occurrence Counting** (70-90% progress)

**File**: `services/bundle_generator.py:1455-1602`

This is the **core bundling logic** - pure frequency counting!

#### Step 3.1: Group SKUs by order
```python
order_groups = defaultdict(list)
for line in filtered_lines:
    order_id = line.order_id
    sku = line.sku
    order_groups[order_id].append(sku)

# Example:
# Order #1001: [SKU-001, SKU-005, SKU-012]
# Order #1002: [SKU-001, SKU-005]
# Order #1003: [SKU-002, SKU-007]
# Order #1004: [SKU-001, SKU-005, SKU-008]
```

#### Step 3.2: Count which SKU pairs appear together
```python
sku_pairs = defaultdict(int)

for order_id, skus in order_groups.items():
    unique_skus_in_order = list(set(skus))  # Remove duplicates within order

    # Count all pairs in this order
    for i, sku1 in enumerate(unique_skus_in_order):
        for sku2 in unique_skus_in_order[i+1:]:
            pair = tuple(sorted([sku1, sku2]))  # (SKU-001, SKU-005)
            sku_pairs[pair] += 1

# Example result:
# (SKU-001, SKU-005): 87 times  ← Bought together in 87 orders
# (SKU-001, SKU-012): 23 times
# (SKU-002, SKU-007): 19 times
# (SKU-003, SKU-009): 12 times
# ...
```

**This is simple co-occurrence counting** - no Apriori, no FP-Growth, no association rules!

#### Step 3.3: Sort pairs by frequency
```python
sorted_pairs = sorted(sku_pairs.items(), key=lambda x: x[1], reverse=True)

# Example (sorted):
# [(SKU-001, SKU-005), 87]  ← Most frequently bought together
# [(SKU-001, SKU-012), 23]
# [(SKU-002, SKU-007), 19]
# ...
```

**Early Exit**: If 0 co-purchase pairs (all single-item orders) → return 0 bundles

#### Step 3.4: Create bundles from top pairs
```python
recommendations = []
for (sku1, sku2), count in sorted_pairs[:max_bundles * 3]:  # Try top 30 pairs to get 10 bundles

    # Get product details from catalog
    product1 = catalog.get(sku1)
    product2 = catalog.get(sku2)

    if not product1 or not product2:
        continue  # Skip if missing catalog data

    if len(recommendations) >= max_bundles:  # Stop at 10 bundles
        break

    # Simple pricing: Fixed 10% discount
    price1 = product1.price
    price2 = product2.price
    total_price = price1 + price2
    bundle_price = total_price * 0.9  # 10% off

    # Create bundle record
    bundle = {
        "bundle_type": "FBT",  # Frequently Bought Together
        "objective": "increase_aov",
        "product_skus": [sku1, sku2],
        "bundle_price": float(bundle_price),
        "original_price": float(total_price),
        "discount_type": "PERCENTAGE",
        "discount_value": 10.0,
        "confidence": min(0.95, 0.5 + (count / 100)),  # More co-purchases = higher confidence
        "ranking_score": product_scores[sku1] + product_scores[sku2],
        "reasoning": f"Often purchased together ({count} times)",
        "discount_reference": f"__quick_start_{csv_upload_id}__",  # Marker for cleanup
    }

    recommendations.append(bundle)

# Result: ~10 FBT bundles
```

---

### **Phase 4: Save to Database** (90-100% progress)

**File**: `services/bundle_generator.py:1620-1627`

```python
await storage.create_bundle_recommendations(recommendations)
logger.info(f"Saved {len(recommendations)} preview bundles")
```

---

## What Quick-Start Does **NOT** Use

❌ **No OpenAI embeddings** (no API calls)
❌ **No ML candidate generation** (no CandidateGenerator)
❌ **No multi-objective optimization** (no OptimizationEngine)
❌ **No Bayesian pricing** (fixed 10% discount)
❌ **No AI copy generation** (no GPT-4 for titles/descriptions)
❌ **No ranking model** (simple sum of product scores)
❌ **No Apriori/FPGrowth** (just raw frequency counting)
❌ **No deduplication service** (small dataset, not needed)
❌ **No explainability engine** (hardcoded reasoning)

---

## Comparison: Quick-Start vs Full V2 Pipeline

| Feature | Quick-Start | Full V2 Pipeline |
|---------|-------------|------------------|
| **Products analyzed** | Top 50 by sales | All products (1000s) |
| **Co-occurrence logic** | Simple pair counting | Apriori + FPGrowth algorithms |
| **Bundle types** | FBT only | FBT, Volume, BOGO, Mix&Match, etc. |
| **Objectives** | 2 (increase_aov, clear_slow_movers) | 4-8 (all objectives) |
| **Scoring** | Flag-based (slow_mover, high_margin) | Multi-factor weighted model |
| **ML/AI** | None | OpenAI embeddings + hybrid scoring |
| **Pricing** | Fixed 10% discount | Bayesian discount optimization |
| **Copy generation** | None (hardcoded) | GPT-4 for titles/descriptions |
| **Optimization** | None | Multi-objective constraint solving |
| **Ranking** | Sum of product scores | Weighted linear model |
| **Bundle count** | ~10 bundles | 50+ bundles |
| **Duration** | 30s - 2 min | 5 - 20 min |

---

## Example: Step-by-Step with Real Data

### Input CSV (simplified):
```
order_id, sku, quantity
1001, SHIRT-A, 1
1001, PANTS-B, 1
1002, SHIRT-A, 2
1002, HAT-C, 1
1003, PANTS-B, 1
1003, HAT-C, 1
1004, SHIRT-A, 1
1004, PANTS-B, 1
1004, SHOES-D, 1
```

### Phase 1: Count SKU sales
```
SHIRT-A: 4 units  ← Top seller
PANTS-B: 3 units
HAT-C: 2 units
SHOES-D: 1 unit
```

### Phase 2: Score products (from catalog)
```
SHIRT-A: 0.5 (normal)
PANTS-B: 0.8 (slow mover)
HAT-C: 0.5 (normal)
SHOES-D: 1.0 (slow mover + high margin)
```

### Phase 3: Count co-occurrences
```
Order #1001: [SHIRT-A, PANTS-B] → (SHIRT-A, PANTS-B) +1
Order #1002: [SHIRT-A, HAT-C] → (SHIRT-A, HAT-C) +1
Order #1003: [PANTS-B, HAT-C] → (PANTS-B, HAT-C) +1
Order #1004: [SHIRT-A, PANTS-B, SHOES-D]
  → (SHIRT-A, PANTS-B) +1
  → (SHIRT-A, SHOES-D) +1
  → (PANTS-B, SHOES-D) +1

Co-occurrence counts:
(SHIRT-A, PANTS-B): 2  ← Most frequent
(SHIRT-A, HAT-C): 1
(SHIRT-A, SHOES-D): 1
(PANTS-B, HAT-C): 1
(PANTS-B, SHOES-D): 1
```

### Phase 4: Create bundles (sorted by frequency)
```
Bundle #1: SHIRT-A + PANTS-B
  - Co-purchased: 2 times
  - Original price: $30 + $50 = $80
  - Bundle price: $72 (10% off)
  - Confidence: 0.5 + (2/100) = 0.52
  - Ranking score: 0.5 + 0.8 = 1.3
  - Reasoning: "Often purchased together (2 times)"

Bundle #2: SHIRT-A + HAT-C
  - Co-purchased: 1 time
  - Original price: $30 + $20 = $50
  - Bundle price: $45 (10% off)
  - Confidence: 0.51
  - Ranking score: 0.5 + 0.5 = 1.0
  - Reasoning: "Often purchased together (1 times)"

... up to 10 bundles total
```

---

## Why This Is Fast

1. **Top 50 products only** - Reduces dataset from thousands to manageable size
2. **Simple counting** - No complex ML algorithms
3. **No API calls** - No OpenAI, no external services
4. **Fixed pricing** - No Bayesian calculations
5. **One bundle type** - Only FBT (no Volume, BOGO, etc.)
6. **Early exits** - Fails fast if insufficient data

**Target: < 2 minutes** for most datasets

---

## Limitations

1. **Only 2-product bundles** (no 3+ item bundles)
2. **Only FBT type** (no Volume discounts, no BOGO)
3. **Fixed 10% discount** (not optimized per bundle)
4. **No semantic understanding** (purely frequency-based)
5. **Top 50 products bias** (misses long-tail opportunities)
6. **No copy generation** (generic descriptions)

But that's okay! **This is just a preview.** The full V2 pipeline runs in the background and replaces these with comprehensive bundles.

---

## Key Takeaway

**Quick-start logic = "Which products are bought together most often?"**

That's it! No fancy ML, just:
1. Count how many units each product sold
2. Pick top 50 products
3. Count how many times each pair appears in orders together
4. Create bundles from top pairs
5. Apply 10% discount
6. Done!

It's basically a simplified version of Amazon's "Frequently Bought Together" - pure co-occurrence statistics.
