# Extensive SKU-Focused Logging Added

## Summary

Added comprehensive SKU tracing throughout the entire bundle generation pipeline to diagnose why SKUs in orders don't match SKUs in catalog, causing 0 bundles.

---

## 🎯 What We're Tracking

### The SKU Journey:
1. **CSV Upload** → Extract raw SKU values from CSV rows
2. **Database Storage** → Store SKUs in order_lines and catalog_snapshots
3. **Quick-Start Phase 1** → Load SKUs from database (order_lines)
4. **Quick-Start Phase 2** → Load SKUs from database (catalog_snapshots)
5. **SKU Cross-Check** → Compare order SKUs vs catalog SKUs
6. **FBT Generation** → Attempt catalog lookup for each SKU pair
7. **Failure Reporting** → Log exact SKUs that fail lookup

---

## 📊 New Logging Added

### 1. CSV Upload - Orders Processing
**File**: `services/csv_processor.py:717-400`

**What's Logged:**
```python
# At extraction time (line 717)
logger.debug(f"[{upload_id}] 🔍 SKU EXTRACTION: variant_id='{variant_id}', sku='{sku}', order_id={order_id}")

# After processing all rows (line 380-400)
logger.info(f"[{upload_id}] 📊 ORDER LINES SKU DISTRIBUTION:")
logger.info(f"[{upload_id}]   Total unique SKUs: {len(sku_counts)}")
logger.info(f"[{upload_id}]   Total unique variant_ids: {len(variant_id_counts)}")
logger.info(f"[{upload_id}]   Top SKUs by frequency:")
for sku, count in sorted(sku_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    logger.info(f"[{upload_id}]     SKU='{sku}': {count} line items")

# Warnings
if none_sku_count > 0:
    logger.warning(f"[{upload_id}] ⚠️  {none_sku_count} order lines have NULL/empty SKUs!")
```

**Why This Helps:**
- ✅ See exact SKU values from CSV
- ✅ Identify NULL/empty SKUs
- ✅ Track SKU frequency distribution
- ✅ Detect placeholder SKUs (no-sku-*, ACC-*)

---

### 2. CSV Upload - Catalog Processing
**File**: `services/csv_processor.py:579-597`

**What's Logged:**
```python
logger.info(f"[{upload_id}] 📋 CATALOG CSV PROCESSING - COMPLETED")
logger.info(f"[{upload_id}]   Total catalog entries: {len(snaps)}")
logger.info(f"[{upload_id}]   Entries with valid SKUs: {len(catalog_skus)}")
logger.info(f"[{upload_id}]   Entries with NULL/empty SKUs: {none_sku_count}")
logger.info(f"[{upload_id}]   Unique SKUs in catalog: {len(set(catalog_skus))}")
logger.info(f"[{upload_id}]   All catalog SKUs: {sorted(list(set(catalog_skus)))}")

logger.info(f"[{upload_id}] 💰 CATALOG PRICES (showing all):")
for sku, price, variant_id in prices:
    logger.info(f"[{upload_id}]     SKU='{sku}' | price=${price} | variant_id={variant_id}")
```

**Why This Helps:**
- ✅ See ALL catalog SKUs (not just top 10)
- ✅ Map SKU → price → variant_id
- ✅ Identify entries without SKUs
- ✅ Spot SKU format patterns

---

### 3. Quick-Start Phase 1 - Order Analysis (Database)
**File**: `services/bundle_generator.py:1334-1367`

**What's Logged:**
```python
logger.info(f"[{csv_upload_id}] 📊 QUICK-START PHASE 1 - SKU ANALYSIS:")
logger.info(f"[{csv_upload_id}]   Total order lines: {len(order_lines)}")
logger.info(f"[{csv_upload_id}]   Order lines with SKU: {len(order_lines) - none_sku_count}")
logger.info(f"[{csv_upload_id}]   Order lines with NULL/empty SKU: {none_sku_count}")
logger.info(f"[{csv_upload_id}]   Unique SKUs found: {len(unique_skus_loaded)}")
logger.info(f"[{csv_upload_id}]   Unique variant_ids found: {len(unique_variant_ids_loaded)}")
logger.info(f"[{csv_upload_id}]   All SKUs from DB: {sorted(unique_skus_loaded)}")

# SKU type breakdown
if no_sku_prefix:
    logger.warning(f"[{csv_upload_id}] ⚠️  {len(no_sku_prefix)} SKUs start with 'no-sku-' prefix: {no_sku_prefix}")
if acc_prefix:
    logger.info(f"[{csv_upload_id}]   {len(acc_prefix)} SKUs start with 'ACC-' prefix: {acc_prefix}")
if other_skus:
    logger.info(f"[{csv_upload_id}]   {len(other_skus)} SKUs with other formats: {other_skus}")
```

**Why This Helps:**
- ✅ Verify SKUs made it from CSV → database
- ✅ Identify 'no-sku-*' placeholder patterns (data quality issue)
- ✅ Track SKU transformations
- ✅ See exact SKU values loaded from DB

---

### 4. Quick-Start Phase 2 - Catalog Analysis (Database)
**File**: `services/bundle_generator.py:1496-1546`

**What's Logged:**
```python
logger.info(f"[{csv_upload_id}] 📋 QUICK-START PHASE 2 - CATALOG ANALYSIS:")
logger.info(f"[{csv_upload_id}]   SKUs in catalog: {len(catalog_skus)}")
logger.info(f"[{csv_upload_id}]   All catalog SKUs: {sorted(catalog_skus)}")

# SKU type breakdown
logger.info(f"[{csv_upload_id}]   Catalog SKUs with 'no-sku-' prefix: {len(no_sku_catalog)}")
logger.info(f"[{csv_upload_id}]   Catalog SKUs with 'ACC-' prefix: {len(acc_catalog)}")
logger.info(f"[{csv_upload_id}]   Catalog SKUs with other formats: {len(other_catalog)}")

# Prices
logger.info(f"[{csv_upload_id}] 💰 CATALOG PRICES (all entries):")
for sku in sorted(skus_with_price):
    price = float(getattr(catalog[sku], 'price', 0))
    variant_id = getattr(catalog[sku], 'variant_id', 'N/A')
    logger.info(f"[{csv_upload_id}]     SKU='{sku}' | price=${price} | variant_id={variant_id}")

# CRITICAL: Cross-check
logger.info(f"[{csv_upload_id}] 🔍 SKU CROSS-CHECK (Orders vs Catalog):")
logger.info(f"[{csv_upload_id}]   SKUs in orders: {len(order_skus)}")
logger.info(f"[{csv_upload_id}]   SKUs in catalog: {len(catalog_sku_set)}")
logger.info(f"[{csv_upload_id}]   SKUs in BOTH: {len(order_skus & catalog_sku_set)}")

if missing_in_catalog:
    logger.error(f"[{csv_upload_id}] ❌ SKUs in orders but NOT in catalog ({len(missing_in_catalog)}): {sorted(missing_in_catalog)}")
    logger.error(f"[{csv_upload_id}]    This will cause catalog lookup failures and 0 bundles!")
else:
    logger.info(f"[{csv_upload_id}] ✅ All order SKUs found in catalog!")
```

**Why This Helps:**
- ✅ **CRITICAL**: Shows exact SKU mismatch between orders and catalog
- ✅ Predicts 0 bundles BEFORE FBT generation starts
- ✅ Lists which SKUs are missing from catalog
- ✅ Verifies catalog prices exist

---

### 5. FBT Bundle Generation - Catalog Lookup Failures
**File**: `services/bundle_generator.py:4812-4828`

**What's Logged:**
```python
if not p1 or not p2:
    catalog_miss_count += 1
    logger.warning(f"[{csv_upload_id}]   ❌ Catalog miss: SKU1='{sku1}' SKU2='{sku2}' | p1_exists={p1 is not None}, p2_exists={p2 is not None}")
    if not p1:
        logger.warning(f"[{csv_upload_id}]      SKU '{sku1}' not found in catalog (tried exact match)")
    if not p2:
        logger.warning(f"[{csv_upload_id}]      SKU '{sku2}' not found in catalog (tried exact match)")
    continue

if price1 <= 0 or price2 <= 0:
    price_fail_count += 1
    logger.warning(f"[{csv_upload_id}]   ❌ Price invalid: SKU1='{sku1}' SKU2='{sku2}' | price1=${price1}, price2=${price2}")
    continue
```

**Why This Helps:**
- ✅ Show exact SKU values that fail lookup
- ✅ Distinguish between sku1 and sku2 failures
- ✅ Clarify exact match was attempted
- ✅ Track which SKUs have invalid prices

---

## 🔍 Example Log Flow

### Successful Case (SKUs Match):

```
[upload_123] 📊 ORDER LINES SKU DISTRIBUTION:
[upload_123]   Total unique SKUs: 4
[upload_123]   Total unique variant_ids: 4
[upload_123]   Top SKUs by frequency:
[upload_123]     SKU='sku-main-a': 14 line items
[upload_123]     SKU='sku-main-b': 13 line items
[upload_123]     SKU='sku-main-c': 9 line items
[upload_123]     SKU='sku-main-d': 8 line items

[upload_123] 📋 CATALOG CSV PROCESSING - COMPLETED
[upload_123]   All catalog SKUs: ['sku-main-a', 'sku-main-b', 'sku-main-c', 'sku-main-d']
[upload_123] 💰 CATALOG PRICES (showing all):
[upload_123]     SKU='sku-main-a' | price=$30.0 | variant_id=12345
[upload_123]     SKU='sku-main-b' | price=$60.0 | variant_id=12346
[upload_123]     SKU='sku-main-c' | price=$20.0 | variant_id=12347
[upload_123]     SKU='sku-main-d' | price=$15.0 | variant_id=12348

[upload_123] 📊 QUICK-START PHASE 1 - SKU ANALYSIS:
[upload_123]   All SKUs from DB: ['sku-main-a', 'sku-main-b', 'sku-main-c', 'sku-main-d']

[upload_123] 📋 QUICK-START PHASE 2 - CATALOG ANALYSIS:
[upload_123]   All catalog SKUs: ['sku-main-a', 'sku-main-b', 'sku-main-c', 'sku-main-d']
[upload_123] 🔍 SKU CROSS-CHECK (Orders vs Catalog):
[upload_123]   SKUs in orders: 4
[upload_123]   SKUs in catalog: 4
[upload_123]   SKUs in BOTH: 4
[upload_123] ✅ All order SKUs found in catalog!

[upload_123] 🎉 FBT BUNDLE GENERATION - COMPLETED
[upload_123]   Bundles created: 4
```

---

### Failure Case (SKU Mismatch):

```
[upload_456] 📊 ORDER LINES SKU DISTRIBUTION:
[upload_456]   Total unique SKUs: 9
[upload_456]   Top SKUs by frequency:
[upload_456]     SKU='no-sku-0649676094': 5 line items
[upload_456]     SKU='no-sku-2601565502': 4 line items
[upload_456]     SKU='ACC-AA4964F2': 3 line items
[upload_456]     SKU='no-sku-0648561982': 2 line items

[upload_456] 📋 CATALOG CSV PROCESSING - COMPLETED
[upload_456]   All catalog SKUs: ['sku-001', 'sku-002', 'sku-003', ...]
[upload_456] 💰 CATALOG PRICES (showing all):
[upload_456]     SKU='sku-001' | price=$30.0 | variant_id=99999
[upload_456]     SKU='sku-002' | price=$60.0 | variant_id=88888

[upload_456] 📊 QUICK-START PHASE 1 - SKU ANALYSIS:
[upload_456]   All SKUs from DB: ['no-sku-0649676094', 'no-sku-2601565502', 'ACC-AA4964F2', ...]
[upload_456] ⚠️  7 SKUs start with 'no-sku-' prefix: ['no-sku-0649676094', ...]

[upload_456] 📋 QUICK-START PHASE 2 - CATALOG ANALYSIS:
[upload_456]   All catalog SKUs: ['sku-001', 'sku-002', ...]
[upload_456] 🔍 SKU CROSS-CHECK (Orders vs Catalog):
[upload_456]   SKUs in orders: 9
[upload_456]   SKUs in catalog: 17
[upload_456]   SKUs in BOTH: 0
[upload_456] ❌ SKUs in orders but NOT in catalog (9): ['ACC-AA4964F2', 'no-sku-0648561982', 'no-sku-0649119038', 'no-sku-0649676094', 'no-sku-0649708862', 'no-sku-2601565502', 'no-sku-2601594814', 'no-sku-3975770430', 'no-sku-4456517950']
[upload_456]    This will cause catalog lookup failures and 0 bundles!

[upload_456] 🔗 FBT BUNDLE GENERATION - STARTED
[upload_456]   ❌ Catalog miss: SKU1='no-sku-0649676094' SKU2='no-sku-2601565502' | p1_exists=False, p2_exists=False
[upload_456]      SKU 'no-sku-0649676094' not found in catalog (tried exact match)
[upload_456]      SKU 'no-sku-2601565502' not found in catalog (tried exact match)
[upload_456]   ❌ Catalog miss: SKU1='no-sku-0648561982' SKU2='no-sku-0649119038' | p1_exists=False, p2_exists=False
[upload_456]      SKU 'no-sku-0648561982' not found in catalog (tried exact match)
...

[upload_456] ❌ ZERO FBT BUNDLES CREATED!
[upload_456]   Possible reasons:
[upload_456]     - Catalog lookup failures: 12
```

---

## 🚀 Next Steps

1. **Deploy** these logging changes (build in progress)
2. **Upload fresh CSV files** to trigger the new logging
3. **Check logs** to see:
   - Exact SKU values from CSV
   - SKU transformation through pipeline
   - Exact mismatch between orders and catalog
   - Root cause of 0 bundles

---

## 🔬 What This Reveals

This logging will answer:

1. ✅ **Are SKUs being extracted correctly from CSV?**
   - See raw SKU values at extraction time
   - Identify NULL/empty SKUs

2. ✅ **Do order SKUs match catalog SKUs?**
   - Cross-check shows exact mismatch
   - Lists missing SKUs

3. ✅ **Why do catalog lookups fail?**
   - Shows exact SKU values that fail
   - Distinguishes between sku1 and sku2 failures

4. ✅ **Are 'no-sku-*' prefixes placeholders?**
   - Warns when these patterns are detected
   - Suggests data quality issues

5. ✅ **Do all SKUs have valid prices?**
   - Lists SKUs with zero/NULL prices
   - Shows price failures during FBT generation

---

## 📋 Files Modified

1. **services/csv_processor.py**
   - Lines 717-726: SKU extraction logging
   - Lines 380-400: Order SKU distribution with NULL tracking
   - Lines 579-597: Catalog SKU distribution with all SKUs listed

2. **services/bundle_generator.py**
   - Lines 1334-1367: Phase 1 SKU analysis with prefix breakdown
   - Lines 1496-1546: Phase 2 catalog analysis with cross-checking
   - Lines 4812-4828: Enhanced FBT catalog lookup failure logging

---

## ✅ Deployment

**Build ID**: `b87df14c` (QUEUED)
**Commit**: `0d1ba8b`
**Status**: Cloud Build deploying...

Once deployed, upload fresh CSV files to see the complete SKU tracing!
