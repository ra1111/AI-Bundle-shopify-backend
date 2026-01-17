# Bundle Data Structure Analysis

## Executive Summary

This document analyzes the discrepancies between how bundles are stored in the database versus what the frontend expects. The analysis covers all shop sizes (very small, small, medium, large) and all bundle types (FBT, VOLUME, BOGO).

---

## Current Issues Found

### 1. Products Array Structure

| Issue | Expected | Actual (Broken) |
|-------|----------|-----------------|
| VOLUME bundles | `[{product_gid, name, price, ...}]` | `[{"item":"0"}]` |
| FBT bundles | `[{product_gid, name, price, ...}]` | `[{"sku":"43374503821470"}]` |
| Missing product_gid | `gid://shopify/Product/123` | Missing entirely |
| Missing name | Product display name | Missing or "Unknown" |
| Missing price | Numeric price | 0 or missing |

### 2. Confidence & Ranking Scores

| Field | Expected | Actual |
|-------|----------|--------|
| confidence | 0.0-1.0 | Always 0.000000 |
| ranking_score | 0.0-1.0 | Always 0.000000 |

### 3. AI Copy

| Field | Expected | Actual |
|-------|----------|--------|
| title | Custom per bundle | Generic "VOLUME Bundle" |
| description | Personalized | "AI-generated VOLUME bundle recommendation" |

---

## Root Cause Analysis

### Issue 1: Products Not Enriched from Catalog

**Location**: [services/bundle_generator.py:3650](services/bundle_generator.py#L3650)

The ML pipeline generates candidates with products as simple strings (SKUs or variant_ids):

```python
# What ML pipeline produces:
recommendation["products"] = ["variant_123", "variant_456"]

# What frontend needs:
recommendation["products"] = [
    {"product_gid": "gid://shopify/Product/789", "name": "Product", "price": 99.99, ...},
    ...
]
```

**Why it happens**:
1. Association rules store products as SKU/variant_id strings
2. FallbackLadder creates `FallbackCandidate` with `products: List[str]`
3. The `_enrich_bundle_with_type_structure` function is supposed to enrich but:
   - Only runs if `isinstance(products[0], str)`
   - Catalog lookup may fail if keyed differently
   - For VOLUME/BOGO, structure gets overwritten after enrichment

### Issue 2: Confidence/Ranking Scores Are Zero

**Location**: [services/bundle_generator.py:1339-1354](services/bundle_generator.py#L1339)

In the fallback injection path, scores come from association rules:

```python
fallback_rec = {
    "confidence": float(getattr(rule, "confidence", 0.0) or 0.0),  # May be 0
    "lift": float(getattr(rule, "lift", 1.0) or 1.0),
    "ranking_score": fallback_rec["confidence"] * max(fallback_rec["lift"], 1.0),  # 0 * 1 = 0
}
```

**Why it happens**:
1. Small shops have no association rules → confidence defaults to 0
2. FallbackLadder candidates don't set confidence properly
3. No fallback scoring for shops without ML data

### Issue 3: VOLUME Products Structure Broken

**Location**: [services/bundle_generator.py:891-905](services/bundle_generator.py#L891)

VOLUME bundles overwrite the products array:

```python
# Line 891-897 - After enrichment, this overwrites:
if isinstance(products, list) and products:
    first_product = products[0] if isinstance(products[0], dict) else {"sku": products[0]}
    recommendation["products"] = {
        "items": [first_product],  # first_product may just be {"sku": "xxx"}
        "volume_tiers": volume_tiers,
    }
```

**Why it happens**:
1. The check `isinstance(products[0], dict)` passes for partial dicts like `{"sku": "xxx"}`
2. The enrichment at line 871 may have failed or produced partial data
3. The `first_product` preserves the broken structure

---

## Shop Size Tier Mapping

| Tier | Order Count | ML Strategy | Bundle Generation |
|------|-------------|-------------|-------------------|
| Very Small | <50 | FallbackLadder (Tiers 7-5) | Cold-start, popularity, heuristics |
| Small | 50-200 | FallbackLadder (Tiers 5-3) | Heuristics, similarity, smoothed co-occurrence |
| Medium | 200-500 | Standard ML + relaxed thresholds | Association rules + LLM embeddings |
| Large | 500+ | Strict ML pipeline | High-confidence association rules |

### Quick-Start vs Full Pipeline

| Path | When Used | Data Quality |
|------|-----------|--------------|
| Quick-Start | First-time shops, <10 order lines | ✅ Correct (uses catalog directly) |
| Full Pipeline | Existing shops with data | ❌ Broken (enrichment failing) |

---

## Data Flow Diagram

```
                    ┌─────────────────────────┐
                    │    Order Data Input     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Candidate Generation   │
                    │  - FPGrowth             │
                    │  - LLM Embeddings       │
                    │  - FallbackLadder       │
                    └────────────┬────────────┘
                                 │
                    products = ["variant_123", "variant_456"]  ← ISSUE: Just strings
                                 │
                    ┌────────────▼────────────┐
                    │  _enrich_bundle_with_   │
                    │  type_structure()       │
                    └────────────┬────────────┘
                                 │
                    Supposed to enrich but fails:
                    - Catalog lookup fails
                    - VOLUME/BOGO overwrites after enrichment
                    - Partial data preserved
                                 │
                    ┌────────────▼────────────┐
                    │   Store to Database     │
                    └────────────┬────────────┘
                                 │
                    products = [{"item":"0"}] or [{"sku":"xxx"}]  ← BROKEN
                                 │
                    ┌────────────▼────────────┐
                    │     Frontend Fetch      │
                    └────────────┬────────────┘
                                 │
                    ❌ No product_gid → Can't fetch from Shopify
                    ❌ No name → Shows "Unknown Product"
                    ❌ No price → Shows ₹0.00
```

---

## Correct Data Structures

### FBT Bundle (Correct)

```json
{
  "id": "uuid",
  "bundle_type": "FBT",
  "objective": "increase_aov",
  "products": {
    "items": [
      {
        "product_gid": "gid://shopify/Product/7982541234",
        "variant_gid": "gid://shopify/ProductVariant/43374503821470",
        "variant_id": "43374503821470",
        "name": "The Complete Snowboard",
        "title": "The Complete Snowboard",
        "price": 699.95,
        "sku": "SNOW-001",
        "image_url": "https://cdn.shopify.com/..."
      },
      {
        "product_gid": "gid://shopify/Product/7982541235",
        "variant_gid": "gid://shopify/ProductVariant/43374503821471",
        "variant_id": "43374503821471",
        "name": "Snowboard Bindings",
        "title": "Snowboard Bindings",
        "price": 299.95,
        "sku": "BIND-001",
        "image_url": "https://cdn.shopify.com/..."
      }
    ],
    "trigger_product": { ... },
    "addon_products": [ ... ]
  },
  "pricing": {
    "original_total": 999.90,
    "bundle_price": 899.91,
    "discount_amount": 99.99,
    "discount_pct": "10%",
    "discount_percentage": 10.0,
    "discount_type": "percentage"
  },
  "ai_copy": {
    "title": "Complete Snowboard Setup",
    "description": "Get both the snowboard and bindings together and save 10%!",
    "tagline": "Save 10% when bought together",
    "cta_text": "Add Bundle to Cart"
  },
  "confidence": 0.85,
  "predicted_lift": 1.25,
  "ranking_score": 0.75,
  "support": 0.05,
  "lift": 2.3
}
```

### VOLUME Bundle (Correct)

```json
{
  "id": "uuid",
  "bundle_type": "VOLUME",
  "objective": "increase_aov",
  "products": {
    "items": [
      {
        "product_gid": "gid://shopify/Product/7982541234",
        "variant_gid": "gid://shopify/ProductVariant/43374503821470",
        "variant_id": "43374503821470",
        "name": "T-Shirt",
        "title": "T-Shirt",
        "price": 29.99,
        "sku": "TSHIRT-001",
        "image_url": "https://cdn.shopify.com/..."
      }
    ],
    "volume_tiers": [
      {"min_qty": 1, "discount_type": "NONE", "discount_value": 0, "label": null},
      {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5, "label": "Starter Pack"},
      {"min_qty": 3, "discount_type": "PERCENTAGE", "discount_value": 10, "label": "Popular"},
      {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15, "label": "Best Value"}
    ]
  },
  "pricing": {
    "original_total": 29.99,
    "bundle_price": 29.99,
    "discount_amount": 0,
    "discount_type": "tiered",
    "volume_tiers": [ ... ]
  },
  "volume_tiers": [ ... ],
  "ai_copy": {
    "title": "Buy More, Save More - T-Shirt",
    "description": "The more T-Shirts you buy, the more you save! Up to 15% off."
  },
  "confidence": 0.72,
  "ranking_score": 0.65
}
```

### BOGO Bundle (Correct)

```json
{
  "id": "uuid",
  "bundle_type": "BOGO",
  "objective": "clear_slow_movers",
  "products": {
    "items": [
      {
        "product_gid": "gid://shopify/Product/7982541234",
        "variant_gid": "gid://shopify/ProductVariant/43374503821470",
        "variant_id": "43374503821470",
        "name": "Socks",
        "title": "Socks",
        "price": 9.99,
        "sku": "SOCK-001"
      }
    ],
    "qualifiers": [
      {"quantity": 2, "product_gid": "...", "name": "Socks", "price": 9.99, ...}
    ],
    "rewards": [
      {"quantity": 1, "discount_type": "free", "discount_percent": 100, ...}
    ]
  },
  "pricing": {
    "buy_total": 19.98,
    "get_value": 9.99,
    "final_price": 19.98,
    "savings": 9.99,
    "discount_type": "bogo",
    "bogo_config": {
      "buy_qty": 2,
      "get_qty": 1,
      "discount_type": "free",
      "discount_percent": 100,
      "same_product": true
    }
  },
  "bogo_config": { ... },
  "qualifiers": [ ... ],
  "rewards": [ ... ],
  "ai_copy": {
    "title": "Buy 2, Get 1 FREE",
    "description": "Stock up on Socks and save!"
  },
  "confidence": 0.68,
  "ranking_score": 0.70
}
```

---

## Fix Strategy

### Immediate Fixes

1. **Fix `_enrich_bundle_with_type_structure`** ([bundle_generator.py:837](services/bundle_generator.py#L837))
   - Move enrichment AFTER type-specific structure building
   - Ensure catalog lookup works for both SKU and variant_id keys
   - Add validation that enrichment succeeded

2. **Fix confidence/ranking defaults**
   - Set minimum confidence = 0.3 for all fallback bundles
   - Calculate ranking_score from multiple signals

3. **Use the new schemas**
   ```python
   from schemas import normalize_fbt_bundle, validate_bundle

   # Before saving:
   normalized = normalize_fbt_bundle(recommendation, catalog)
   is_valid, errors = validate_bundle(normalized)
   if not is_valid:
       logger.warning(f"Invalid bundle: {errors}")
   ```

### Long-term Fixes

1. Add validation in `create_bundle_recommendations()` before DB save
2. Add integration tests that verify schema compliance
3. Add a migration to fix existing broken bundles

---

## Using the Schema Module

```python
from schemas import (
    normalize_fbt_bundle,
    normalize_volume_bundle,
    normalize_bogo_bundle,
    validate_bundle,
    enrich_products_from_catalog,
)

# Normalize a bundle based on type
if bundle_type == "FBT":
    normalized = normalize_fbt_bundle(raw_bundle, catalog)
elif bundle_type == "VOLUME":
    normalized = normalize_volume_bundle(raw_bundle, catalog)
elif bundle_type == "BOGO":
    normalized = normalize_bogo_bundle(raw_bundle, catalog)

# Validate before saving
is_valid, errors = validate_bundle(normalized)
if not is_valid:
    logger.error(f"Bundle validation failed: {errors}")
    # Handle error...

# Save to database
await storage.create_bundle_recommendations([normalized], csv_upload_id)
```

---

## Files Changed

| File | Purpose |
|------|---------|
| `schemas/__init__.py` | Package exports |
| `schemas/bundle_schemas.py` | Standardized schemas, dataclasses, validation |
| `schemas/BUNDLE_DATA_ANALYSIS.md` | This analysis document |
