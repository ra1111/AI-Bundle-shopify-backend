# Fix: Frontend Not Loading Bundle Types

## Problem

**Frontend sees:** Bundles without types (null/untyped), defaults to showing all as "FBT"

**CockroachDB has:** 102 bundles (60 BOGO, 30 FBT, 12 VOLUME) all with `is_approved = true`

**Diagnosis:** The FastAPI service IS connected to CockroachDB (verified DATABASE_URL secret), but frontend isn't receiving bundle_type values.

---

## Root Cause Analysis

The issue could be one of these:

### 1. **Shop ID Mismatch** (Most Likely)
- CockroachDB has bundles for shop: `"actual-shop.myshopify.com"`
- Frontend queries with: `"different-shop.myshopify.com"`
- Result: API returns 0 bundles or wrong bundles

### 2. **Approval Status Filter**
- CockroachDB has bundles with `is_approved = true`
- Frontend queries with `approved=false` or no filter
- Result: API returns unapproved bundles (which might not exist)

### 3. **Column Name in Old Data**
- Old bundles might have `discount_type` instead of `bundle_type`
- Backend model expects `bundle_type`
- Result: Field is null for old data

---

## Verification Steps

### **Step 1: Test API Directly**

Run this script to see what the API actually returns:

```bash
cd /Users/rahular/Documents/AI\ Bundler/python_server
./test_api_bundles.sh
```

Enter your actual shop ID when prompted (e.g., `shop.myshopify.com`)

**Check:**
- ✅ Does it return bundles?
- ✅ Do bundles have `bundle_type` field?
- ✅ Are there `"BOGO"`, `"FBT"`, `"VOLUME"` values?
- ✅ Are bundles `is_approved: true`?

---

### **Step 2: Check Database Directly**

If you have access to CockroachDB SQL console, run:

```sql
-- Check total bundles
SELECT
    shop_id,
    bundle_type,
    is_approved,
    COUNT(*) as count
FROM bundle_recommendations
GROUP BY shop_id, bundle_type, is_approved
ORDER BY shop_id, bundle_type;

-- Check sample bundle
SELECT
    id,
    shop_id,
    bundle_type,
    objective,
    is_approved,
    created_at
FROM bundle_recommendations
WHERE is_approved = true
LIMIT 5;

-- Check for null bundle_type
SELECT COUNT(*) as null_count
FROM bundle_recommendations
WHERE bundle_type IS NULL;
```

---

### **Step 3: Check Frontend Query**

In your Remix app, check what parameters the frontend is using:

```typescript
// What is this?
const shopId = "???"  // Should match database exactly

// What filters are used?
const response = await fetch(
  `/api/shopify/recommendations?shopId=${shopId}&approved=true&limit=50`
  //                                                ^^^^^^^^^^^^^^
  //                                   Make sure approved=true!
)
```

---

## Possible Fixes

### **Fix 1: Shop ID Mismatch**

If shop IDs don't match, update the frontend to use the correct shop ID:

**Backend normalizes shop IDs** (see routers/shopify_upload.py:235):
```python
from services.utils import resolve_shop_id
normalized_shop_id = resolve_shop_id(shop_id)
```

This function:
- Strips whitespace
- Converts to lowercase
- Removes `.myshopify.com` suffix if present

**So these are all the same:**
- `"Shop.MyShopify.com"`
- `"shop.myshopify.com"`
- `"shop"`

**Make sure frontend uses the SAME shop ID that was used during upload!**

---

### **Fix 2: Add approved=true Filter**

Update frontend to query only approved bundles:

**Before:**
```typescript
fetch(`/api/shopify/recommendations?shopId=${shopId}`)
```

**After:**
```typescript
fetch(`/api/shopify/recommendations?shopId=${shopId}&approved=true&limit=50`)
//                                                     ^^^^^^^^^^^^^^^
//                                            Filter for approved bundles
```

---

### **Fix 3: Database Column Mismatch (Old Data)**

If you have old bundles with `discount_type` instead of `bundle_type`, run migration:

**Option A: Update Column Values**
```sql
-- If bundles have NULL bundle_type but have discount_reference
UPDATE bundle_recommendations
SET bundle_type = CASE
    WHEN discount_reference LIKE '%fbt%' THEN 'FBT'
    WHEN discount_reference LIKE '%bogo%' THEN 'BOGO'
    WHEN discount_reference LIKE '%volume%' THEN 'VOLUME'
    ELSE 'FBT'  -- default
END
WHERE bundle_type IS NULL OR bundle_type = '';
```

**Option B: Delete Old Bundles**
```sql
-- Delete bundles without bundle_type
DELETE FROM bundle_recommendations
WHERE bundle_type IS NULL OR bundle_type = '';
```

---

## Testing After Fix

### **Test 1: API Returns Correct Data**
```bash
curl "https://bundle-api-250755735924.us-central1.run.app/api/shopify/recommendations?shopId=YOUR_SHOP_ID&approved=true&limit=5" | jq '.recommendations[] | {bundle_type, is_approved}'
```

**Expected:**
```json
{
  "bundle_type": "BOGO",
  "is_approved": true
}
{
  "bundle_type": "FBT",
  "is_approved": true
}
{
  "bundle_type": "VOLUME",
  "is_approved": true
}
```

---

### **Test 2: Frontend Receives Types**

In your Remix app, add logging:

```typescript
const response = await fetch(`/api/shopify/recommendations?shopId=${shopId}&approved=true`)
const data = await response.json()

console.log("Bundle types:", data.recommendations.map(r => r.bundle_type))
// Expected: ["BOGO", "BOGO", "FBT", "VOLUME", ...]
```

---

### **Test 3: UI Shows Correct Types**

Frontend should now display:
- ✅ BOGO bundles with BOGO UI
- ✅ FBT bundles with FBT UI
- ✅ Volume bundles with Volume UI

---

## Quick Fix Script

If you want to quickly verify the issue, run this:

```bash
#!/bin/bash
SHOP_ID="your-shop.myshopify.com"  # Change this
BACKEND="https://bundle-api-250755735924.us-central1.run.app"

echo "Testing bundle recommendations for: $SHOP_ID"
echo ""

echo "1. All bundles:"
curl -s "${BACKEND}/api/shopify/recommendations?shopId=${SHOP_ID}&limit=5" | jq '.count'

echo ""
echo "2. Approved bundles:"
curl -s "${BACKEND}/api/shopify/recommendations?shopId=${SHOP_ID}&approved=true&limit=5" | jq '.count'

echo ""
echo "3. Bundle types:"
curl -s "${BACKEND}/api/shopify/recommendations?shopId=${SHOP_ID}&approved=true&limit=10" | jq '.recommendations[] | .bundle_type'
```

---

## Summary Checklist

- [ ] Run `./test_api_bundles.sh` with your actual shop ID
- [ ] Verify API returns bundles with `bundle_type` field
- [ ] Check frontend is using correct shop ID
- [ ] Check frontend is using `approved=true` filter
- [ ] Verify database has bundles for that shop_id
- [ ] Check bundle_type is not NULL in database
- [ ] Test frontend displays correct bundle types

---

## Expected Result

After fix:
- Frontend queries: `/api/shopify/recommendations?shopId=shop.myshopify.com&approved=true`
- API returns: 102 bundles (60 BOGO, 30 FBT, 12 VOLUME)
- Frontend displays: Correct UI for each bundle type
- No more "all bundles show as FBT"!

---

## Need More Help?

1. **Share test results:**
   Run `./test_api_bundles.sh` and share the output

2. **Share database query:**
   Run the SQL queries above and share results

3. **Share frontend code:**
   Show the exact API call your frontend makes

4. **Check Cloud Run logs:**
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bundle-api" --limit 50 | grep "recommendations"
   ```

This will show if the API is even being called and what parameters it receives.
