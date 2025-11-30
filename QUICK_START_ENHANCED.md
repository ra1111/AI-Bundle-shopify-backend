# Enhanced Quick-Start: Multi-Type Bundle Generation

## Overview

The quick-start mode has been enhanced to generate **three bundle types** instead of just FBT:
- **FBT (Frequently Bought Together)**: 3-5 bundles
- **BOGO (Buy One Get One)**: 2-3 bundles
- **Volume**: 1-2 bundles

Total: ~10 bundles distributed across types to give merchants a diverse preview.

---

## Changes Made

### 1. **Added Three Helper Functions** (`services/bundle_generator.py:1272-1528`)

#### `_build_quick_start_fbt_bundles()`
- **Purpose**: Create 2-product bundles from co-occurrence patterns
- **Logic**: Count which SKU pairs appear together in orders, pick top pairs
- **Discount**: Fixed 10% off
- **Criteria**: Based on transaction frequency

#### `_build_quick_start_bogo_bundles()`
- **Purpose**: Clear slow-moving inventory
- **Logic**: Find products marked as `is_slow_mover` with excess stock (>5 units)
- **Discount**: Buy 2, Get 1 free (~33% effective discount on 3 units)
- **Criteria**:
  - `is_slow_mover == True`
  - `available_total > 5`
  - Sorted by inventory level (highest first)

#### `_build_quick_start_volume_bundles()`
- **Purpose**: Encourage bulk purchases of popular items
- **Logic**: Find high-stock, popular SKUs
- **Discount**: Tiered (5% @ qty 2, 10% @ qty 3, 15% @ qty 5)
- **Criteria**:
  - `available_total > 20` (sufficient stock)
  - `units_sold >= 3` (proven demand)
  - Sorted by sales volume

---

### 2. **Updated `generate_quick_start_bundles()`** (`services/bundle_generator.py:1719-1762`)

**Old Behavior** (Phase 3):
- Only generated FBT bundles
- Manual co-occurrence counting inline
- All 10 bundles were FBT

**New Behavior** (Phase 3):
```python
# Decide distribution
max_fbt_bundles = max(3, min(5, max_bundles))      # 3-5 bundles
remaining = max(0, max_bundles - max_fbt_bundles)
max_bogo_bundles = min(3, remaining)                # 2-3 bundles
remaining -= max_bogo_bundles
max_volume_bundles = min(2, remaining)              # 1-2 bundles

# Build each type
fbt_bundles = _build_quick_start_fbt_bundles(...)
bogo_bundles = _build_quick_start_bogo_bundles(...) if max_bogo_bundles > 0 else []
volume_bundles = _build_quick_start_volume_bundles(...) if max_volume_bundles > 0 else []

# Combine
recommendations = fbt_bundles + bogo_bundles + volume_bundles
```

---

### 3. **Updated Metrics** (`services/bundle_generator.py:1794-1821`)

**Added**:
```python
"bundle_type_counts": {
    "fbt": len(fbt_bundles),
    "bogo": len(bogo_bundles),
    "volume": len(volume_bundles),
}
```

**Removed** (no longer applicable):
- `unique_orders`
- `copurchase_pairs_found`
- `catalog_misses`

---

### 4. **Enhanced Logging** (`services/bundle_generator.py:1823-1831`)

**Old**:
```
========== QUICK-START COMPLETE ==========
  Bundles: 10
  Duration: 45.2s
  Products: 48
```

**New**:
```
========== QUICK-START COMPLETE ==========
  Total Bundles: 10
  - FBT: 5
  - BOGO: 3
  - Volume: 2
  Duration: 45.2s
  Products Analyzed: 48
```

---

## Bundle Data Structure

Each helper function returns bundles with this structure:

### FBT Bundle:
```python
{
    "id": uuid,
    "csv_upload_id": str,
    "bundle_type": "FBT",
    "objective": "increase_aov",
    "products": [
        {"sku": "SKU-001", "name": "...", "price": 29.99, ...},
        {"sku": "SKU-002", "name": "...", "price": 39.99, ...}
    ],
    "pricing": {
        "original_total": 69.98,
        "bundle_price": 62.98,
        "discount_amount": 7.00,
        "discount_pct": "10.0%"
    },
    "confidence": Decimal("0.85"),
    "ranking_score": Decimal("1.3"),
    "discount_reference": "__quick_start_{upload_id}__",
    "is_approved": True,
    ...
}
```

### BOGO Bundle:
```python
{
    "id": uuid,
    "bundle_type": "BOGO",
    "objective": "clear_slow_movers",
    "products": [
        {
            "sku": "SKU-SLOW",
            "name": "...",
            "price": 20.00,
            "min_quantity": 2,      # Buy 2
            "reward_quantity": 1    # Get 1 free
        }
    ],
    "pricing": {
        "original_total": 60.00,      # 3 units × $20
        "bundle_price": 40.00,        # Pay for 2 only
        "discount_amount": 20.00,
        "discount_pct": "33.3%",
        "bogo_config": {
            "buy_qty": 2,
            "get_qty": 1,
            "discount_percent": 33.3
        }
    },
    ...
}
```

### Volume Bundle:
```python
{
    "id": uuid,
    "bundle_type": "VOLUME",
    "objective": "increase_aov",
    "products": [
        {"sku": "SKU-POPULAR", "name": "...", "price": 25.00}
    ],
    "pricing": {
        "original_total": 25.00,
        "bundle_price": 25.00,
        "discount_amount": 0,
        "discount_pct": "0%",
        "volume_tiers": [
            {"min_qty": 1, "discount_type": "NONE",       "discount_value": 0},
            {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5},
            {"min_qty": 3, "discount_type": "PERCENTAGE", "discount_value": 10},
            {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15}
        ]
    },
    ...
}
```

---

## Example Output

For `max_bundles=10` (default):

```
Quick-start Phase 3: Bundle targets - FBT=5, BOGO=3, VOLUME=2

Quick-start Phase 3 complete:
  Generated 10 total bundles (FBT=5, BOGO=2, VOLUME=1)

Note: Actual counts may be lower if insufficient data
  - BOGO requires slow_mover products
  - Volume requires high-stock items
```

---

## Benefits

1. **Diverse Bundle Types**: Merchants see different strategies immediately
2. **Better Inventory Management**: BOGO helps clear slow movers
3. **Higher AOV**: Volume bundles encourage bulk purchases
4. **Real-World Feel**: Looks like full V2 output, not just a preview
5. **Same Speed**: Still completes in < 2 minutes

---

## Data Flow

```
CSV Upload
    ↓
Phase 1: Load top 50 products by sales
    ↓
Phase 2: Score products (slow_mover, high_margin flags)
    ↓
Phase 3: Generate bundles
    ├─ FBT: Co-occurrence counting (5 bundles)
    ├─ BOGO: Slow movers with excess stock (3 bundles)
    └─ Volume: Popular + high stock (2 bundles)
    ↓
Phase 4: Save to bundle_recommendations table
    ↓
Result: 10 diverse bundles with __quick_start_* marker
```

---

## Backward Compatibility

✅ **No breaking changes**
- All bundles still use `discount_reference = "__quick_start_{upload_id}__"`
- Still persisted to `bundle_recommendations` table
- Still marked as `is_approved = True`
- Full V2 pipeline still replaces them in background

---

## Testing

### Manual Test:
```sql
-- After quick-start runs, check bundle types
SELECT
    bundle_type,
    objective,
    COUNT(*) as count
FROM bundle_recommendations
WHERE discount_reference LIKE '__quick_start_%'
GROUP BY bundle_type, objective;

-- Expected result:
-- bundle_type | objective          | count
-- FBT         | increase_aov       | 3-5
-- BOGO        | clear_slow_movers  | 2-3
-- VOLUME      | increase_aov       | 1-2
```

### Check Logs:
```
[upload_123] Quick-start Phase 3: Bundle targets - FBT=5, BOGO=3, VOLUME=2
[upload_123] Quick-start Phase 3 complete: Generated 10 total bundles (FBT=5, BOGO=3, VOLUME=2)
[upload_123] ========== QUICK-START COMPLETE ==========
  Total Bundles: 10
  - FBT: 5
  - BOGO: 3
  - Volume: 2
  Duration: 47.23s
  Products Analyzed: 48
```

---

## Next Steps

The frontend/translator layer needs to map these bundle types to the unified engine format:

1. **FBT** → Already handled (existing logic)
2. **BOGO** → Map `bogo_config` to BXGY bundle_def
3. **Volume** → Map `volume_tiers` to tiered discount function

This will be covered in the next implementation phase.
