# Extensive Logging Added - Complete Traceability

## Summary

I've added comprehensive logging to track exactly what's happening during CSV processing and bundle generation. This will help us understand why quick-start generated 0 bundles.

---

## 🎯 What We're Logging

### 1. **CSV Upload - Orders Processing**
**File**: `services/csv_processor.py:357-408`

**Logs Added:**
```python
logger.info(f"[{upload_id}] 📦 ORDERS CSV PROCESSING - STARTED")
logger.info(f"[{upload_id}]   Total rows to process: {len(rows)}")

# SKU Distribution
logger.info(f"[{upload_id}] 📊 ORDER LINES SKU DISTRIBUTION:")
logger.info(f"[{upload_id}]   Total unique SKUs: {len(sku_counts)}")
for sku, count in sorted(sku_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    logger.info(f"[{upload_id}]     {sku}: {count} line items")

# Order Structure
logger.info(f"[{upload_id}] 📦 ORDERS WITH MULTIPLE ITEMS:")
logger.info(f"[{upload_id}]   Orders with 2+ items: {len(order_sku_pairs)}/{len(orders_map)}")
for order_id, skus in list(order_sku_pairs.items())[:5]:
    logger.info(f"[{upload_id}]     {order_id}: {skus}")
```

**What This Tells Us:**
- ✅ How many unique SKUs are in the uploaded orders
- ✅ Which SKUs appear most frequently
- ✅ Which orders have multiple items (needed for FBT bundles)
- ✅ Exact SKU combinations in each order

---

### 2. **CSV Upload - Catalog Processing**
**File**: `services/csv_processor.py:568-587`

**Logs Added:**
```python
logger.info(f"[{upload_id}] 📋 CATALOG CSV PROCESSING - COMPLETED")
logger.info(f"[{upload_id}]   Total catalog entries: {len(snaps)}")
logger.info(f"[{upload_id}]   Unique SKUs in catalog: {len(set(catalog_skus))}")
logger.info(f"[{upload_id}]   Sample SKUs: {list(set(catalog_skus))[:10]}")

# Prices
logger.info(f"[{upload_id}] 💰 CATALOG PRICES:")
for sku, price in prices[:10]:
    logger.info(f"[{upload_id}]     {sku}: ${price}")
```

**What This Tells Us:**
- ✅ How many products are in the catalog
- ✅ Which SKUs have pricing data
- ✅ If catalog SKUs match order SKUs

---

### 3. **Quick-Start - Phase 1 (Order Loading)**
**File**: `services/bundle_generator.py:1326-1348`

**Logs Added:**
```python
logger.info(f"[{csv_upload_id}] 📊 QUICK-START PHASE 1 - SKU ANALYSIS:")
logger.info(f"[{csv_upload_id}]   Total order lines: {len(order_lines)}")
logger.info(f"[{csv_upload_id}]   Unique SKUs found: {len(unique_skus_loaded)}")
logger.info(f"[{csv_upload_id}]   SKU list: {sorted(unique_skus_loaded)}")

# Order structure
logger.info(f"[{csv_upload_id}]   Unique orders: {len(order_ids)}")
if len(order_ids) <= 5:
    for oid in order_ids[:5]:
        order_skus = [getattr(line, 'sku', None) for line in order_lines if getattr(line, 'order_id', None) == oid]
        logger.info(f"[{csv_upload_id}]     Order {oid}: {order_skus}")
```

**What This Tells Us:**
- ✅ What data was loaded from the database
- ✅ If data matches what we uploaded
- ✅ Exact order structure (which SKUs per order)

---

### 4. **Quick-Start - Phase 2 (Catalog Loading)**
**File**: `services/bundle_generator.py:1470-1498`

**Logs Added:**
```python
logger.info(f"[{csv_upload_id}] 📋 QUICK-START PHASE 2 - CATALOG ANALYSIS:")
logger.info(f"[{csv_upload_id}]   SKUs in catalog: {len(catalog_skus)}")
logger.info(f"[{csv_upload_id}]   Catalog SKU list: {sorted(catalog_skus)}")

# Price check
logger.info(f"[{csv_upload_id}]   SKUs with valid prices: {len(skus_with_price)}")
for sku in skus_with_price[:5]:
    price = float(getattr(catalog[sku], 'price', 0))
    logger.info(f"[{csv_upload_id}]     {sku}: ${price}")

# Cross-check
order_skus = set(filter(None, [getattr(line, 'sku', None) for line in order_lines]))
missing_in_catalog = order_skus - catalog_sku_set
if missing_in_catalog:
    logger.warning(f"[{csv_upload_id}] ⚠️  SKUs in orders but NOT in catalog: {missing_in_catalog}")
else:
    logger.info(f"[{csv_upload_id}] ✅ All order SKUs found in catalog!")
```

**What This Tells Us:**
- ✅ Which SKUs are in the catalog loaded from DB
- ✅ Which SKUs have valid prices
- ✅ **CRITICAL**: If there's a mismatch between order SKUs and catalog SKUs

---

### 5. **FBT Bundle Generation - SKU Pair Analysis**
**File**: `services/bundle_generator.py:4692-4722`

**Logs Added:**
```python
logger.info(f"[{csv_upload_id}] 🔗 FBT BUNDLE GENERATION - STARTED")
logger.info(f"[{csv_upload_id}]   Input: {len(filtered_lines)} order lines")
logger.info(f"[{csv_upload_id}]   Orders grouped: {len(order_groups)}")

logger.info(f"[{csv_upload_id}] 📊 SKU PAIRS FOUND:")
logger.info(f"[{csv_upload_id}]   Total unique pairs: {len(sku_pairs)}")
for pair, count in sorted(sku_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
    logger.info(f"[{csv_upload_id}]     {pair[0]} + {pair[1]}: {count} times")
```

**What This Tells Us:**
- ✅ Exact SKU pairs that were found in orders
- ✅ How many times each pair was bought together
- ✅ If the co-occurrence counting logic works

---

### 6. **FBT Bundle Generation - Filtering & Failures**
**File**: `services/bundle_generator.py:4724-4776`

**Logs Added:**
```python
logger.info(f"[{csv_upload_id}] ✅ Scoring complete:")
logger.info(f"[{csv_upload_id}]   Pairs after filtering: {len(scored_pairs)}/{len(sku_pairs)}")
logger.info(f"[{csv_upload_id}]   Filtered out (weak): {filtered_out_count}")

# During catalog lookup
logger.warning(f"[{csv_upload_id}]   Catalog miss: {sku1}+{sku2} (p1={p1 is not None}, p2={p2 is not None})")

# During price check
logger.warning(f"[{csv_upload_id}]   Price invalid: {sku1}+{sku2} (p1=${price1}, p2=${price2})")
```

**What This Tells Us:**
- ✅ How many pairs passed/failed the covis_sim filter
- ✅ **CRITICAL**: Which pairs failed catalog lookup
- ✅ **CRITICAL**: Which pairs had invalid prices

---

### 7. **FBT Bundle Generation - Final Summary**
**File**: `services/bundle_generator.py:4853-4863`

**Logs Added:**
```python
logger.info(f"[{csv_upload_id}] 🎉 FBT BUNDLE GENERATION - COMPLETED")
logger.info(f"[{csv_upload_id}]   Bundles created: {len(recommendations)}")
logger.info(f"[{csv_upload_id}]   Catalog misses: {catalog_miss_count}")
logger.info(f"[{csv_upload_id}]   Price failures: {price_fail_count}")

if len(recommendations) == 0:
    logger.error(f"[{csv_upload_id}] ❌ ZERO FBT BUNDLES CREATED!")
    logger.error(f"[{csv_upload_id}]   Possible reasons:")
    logger.error(f"[{csv_upload_id}]     - SKU pairs filtered out: {filtered_out_count}")
    logger.error(f"[{csv_upload_id}]     - Catalog lookup failures: {catalog_miss_count}")
    logger.error(f"[{csv_upload_id}]     - Invalid prices: {price_fail_count}")
```

**What This Tells Us:**
- ✅ Final bundle count
- ✅ Root cause of 0 bundles (if it happens)

---

## 📊 Log Flow for Next Upload

When you upload data next time, you'll see logs like:

```
[upload_id] 📦 ORDERS CSV PROCESSING - STARTED
[upload_id]   Total rows to process: 44

[upload_id] 📊 ORDER LINES SKU DISTRIBUTION:
[upload_id]   Total unique SKUs: 4
[upload_id]     sku-main-a: 14 line items
[upload_id]     sku-main-b: 13 line items
[upload_id]     sku-main-c: 9 line items
[upload_id]     sku-main-d: 8 line items

[upload_id] 📦 ORDERS WITH MULTIPLE ITEMS:
[upload_id]   Orders with 2+ items: 20/24
[upload_id]     gid://shopify/Order/900001: ['sku-main-a', 'sku-main-b']
[upload_id]     gid://shopify/Order/900002: ['sku-main-a', 'sku-main-b']
[upload_id]     ...

[upload_id] ✅ Created 24 orders in database
[upload_id] ✅ Created 44 order lines in database

---

[upload_id] 📋 CATALOG CSV PROCESSING - COMPLETED
[upload_id]   Total catalog entries: 4
[upload_id]   Unique SKUs in catalog: 4
[upload_id]   Sample SKUs: ['sku-main-a', 'sku-main-b', 'sku-main-c', 'sku-main-d']

[upload_id] 💰 CATALOG PRICES:
[upload_id]     sku-main-a: $30.0
[upload_id]     sku-main-b: $60.0
[upload_id]     sku-main-c: $20.0
[upload_id]     sku-main-d: $15.0

---

QUICK-START BUNDLE GENERATION:

[upload_id] 📊 QUICK-START PHASE 1 - SKU ANALYSIS:
[upload_id]   Total order lines: 44
[upload_id]   Unique SKUs found: 4
[upload_id]   SKU list: ['sku-main-a', 'sku-main-b', 'sku-main-c', 'sku-main-d']
[upload_id]   Unique orders: 24

[upload_id] 📋 QUICK-START PHASE 2 - CATALOG ANALYSIS:
[upload_id]   SKUs in catalog: 4
[upload_id]   Catalog SKU list: ['sku-main-a', 'sku-main-b', 'sku-main-c', 'sku-main-d']
[upload_id]   SKUs with valid prices: 4
[upload_id]     sku-main-a: $30.0
[upload_id]     sku-main-b: $60.0
[upload_id] ✅ All order SKUs found in catalog!

[upload_id] 🔗 FBT BUNDLE GENERATION - STARTED
[upload_id]   Input: 44 order lines
[upload_id]   Orders grouped: 24

[upload_id] 📊 SKU PAIRS FOUND:
[upload_id]   Total unique pairs: 4
[upload_id]     sku-main-a + sku-main-b: 8 times
[upload_id]     sku-main-a + sku-main-c: 5 times
[upload_id]     sku-main-b + sku-main-d: 4 times
[upload_id]     sku-main-c + sku-main-d: 3 times

[upload_id] ✅ Scoring complete:
[upload_id]   Pairs after filtering: 4/4
[upload_id]   Filtered out (weak): 0

[upload_id] 🎉 FBT BUNDLE GENERATION - COMPLETED
[upload_id]   Bundles created: 4
[upload_id]   Catalog misses: 0
[upload_id]   Price failures: 0
```

OR if something goes wrong:

```
[upload_id] ⚠️  SKUs in orders but NOT in catalog: {'sku-main-e', 'sku-main-f'}
[upload_id]   Catalog miss: sku-main-a+sku-main-e (p1=True, p2=False)
[upload_id] ❌ ZERO FBT BUNDLES CREATED!
[upload_id]   Possible reasons:
[upload_id]     - SKU pairs filtered out: 0
[upload_id]     - Catalog lookup failures: 4
[upload_id]     - Invalid prices: 0
```

---

## 🔍 How to Use These Logs

### To Diagnose the Previous Run (b317e65f):
```bash
gcloud logging read \
  'resource.type=cloud_run_revision AND
   resource.labels.service_name=bundle-api AND
   textPayload:"b317e65f" AND
   (textPayload:"📦" OR textPayload:"📋" OR textPayload:"📊" OR textPayload:"🔗")' \
  --limit=500 \
  --format="value(timestamp,textPayload)"
```

### For Future Uploads:
Just watch the Cloud Run logs in real-time - all the emoji icons make it easy to find the important sections!

---

## ✅ Files Modified

1. **services/csv_processor.py**
   - Added SKU distribution logging for orders
   - Added catalog SKU and price logging
   - Added multi-item order analysis

2. **services/bundle_generator.py**
   - Added Phase 1 order line analysis
   - Added Phase 2 catalog analysis with cross-checking
   - Added detailed FBT pair counting
   - Added filtering failure tracking
   - Added catalog miss & price fail tracking
   - Added comprehensive error summary

---

## 🎯 Next Steps

1. **Deploy these changes** to Cloud Run
2. **Upload your data again** (same files)
3. **Check logs** to see exactly what happened:
   - Did orders upload correctly?
   - Did catalog upload correctly?
   - Did SKUs match between orders and catalog?
   - Which filter caused the 0 bundles?

The logs will tell us **exactly** where the problem is! 🔍
