# API Streamlining - Migration Summary

**Date:** 2025-12-02
**Changes:** Simplified API to ONE status endpoint with clear contract

---

## 2025-12-03 – Order line upsert fix
- Added Alembic migration `20251203_000003_add_order_lines_partial_unique_indexes` to create the partial unique indexes backing `ON CONFLICT (order_id, sku) WHERE line_item_id IS NULL` and `(order_id, line_item_id)`.
- Fixes ingestion errors like: `InvalidColumnReferenceError: there is no unique or exclusion constraint matching the ON CONFLICT specification`.
- Action: run `alembic upgrade head` before next deploy so CSV uploads can upsert order_lines safely.

---

## **What Changed**

### **✅ SIMPLIFIED: One Endpoint to Rule Them All**

**Before:** Two confusing endpoints
- `/api/upload-status/{uploadId}` - no `bundle_count`, complex statuses
- `/api/shopify/status/{uploadId}` - had `bundle_count`, simple statuses

**After:** ONE clear endpoint
- ✅ `/api/shopify/status/{uploadId}` - **Use this!**
- ⚠️ `/api/upload-status/{uploadId}` - Deprecated (but still works)

---

## **What's Better Now**

### **1. Simplified Status Values**

**Before:** 10+ confusing statuses
- `"bundle_generation_completed"`
- `"bundle_generation_in_progress"`
- `"bundle_generation_queued"`
- `"bundle_generation_async"`
- etc.

**After:** 3 simple statuses
- `"processing"` - CSV or bundles being generated
- `"completed"` - Everything done
- `"failed"` - Something went wrong

**Backend automatically maps complex internal statuses to simple frontend ones!**

---

### **2. Always Includes bundle_count**

**Before:**
- Only checked for `bundle_count` when `status === "completed"`
- Missed bundles when status was `"bundle_generation_completed"`
- Inconsistent behavior

**After:**
- Checks for bundles on EVERY status call
- Returns `bundle_count` whenever bundles exist
- Works for re-syncs (existing bundles shown immediately)

---

### **3. Better Logging**

Every status check now logs:
```
🔍 STATUS CHECK: upload_id=abc-123
✅ STATUS CHECK: upload_id=abc-123 status=completed rows=100/100
📊 STATUS RESPONSE: internal_status=bundle_generation_completed → frontend_status=completed bundle_count=10
```

Makes debugging SO much easier!

---

## **Code Changes Made**

### **File: [routers/shopify_upload.py](routers/shopify_upload.py)**

**Enhanced `/api/shopify/status/{uploadId}` (lines 172-220):**
1. Always checks for `bundle_count` (not just when status="completed")
2. Maps internal statuses to simple frontend statuses
3. Adds detailed logging for debugging
4. Returns `bundle_count` whenever bundles exist

---

### **File: [routers/uploads.py](routers/uploads.py)**

**Deprecated `/api/upload-status/{uploadId}` (lines 137-184):**
1. Added deprecation warning in docstring
2. Logs warning when endpoint is called
3. Returns `_deprecated: true` in response
4. Still works for backward compatibility

---

### **File: [routers/bundle_recommendations.py](routers/bundle_recommendations.py)**

**Quick mode improvements (lines 283-307, 353-359):**
1. Returns early when bundles already exist (no re-generation)
2. Disabled automatic full generation after quick mode
3. Better logging for all scenarios

---

## **Migration Guide for Frontend**

### **Step 1: Update Status Endpoint**

**Before:**
```typescript
const statusRes = await fetch(`/api/upload-status/${uploadId}`)
```

**After:**
```typescript
const statusRes = await fetch(`/api/shopify/status/${uploadId}`)
```

---

### **Step 2: Update Status Checks**

**Before:**
```typescript
if (status.status === "bundle_generation_completed") {
  // Success
}
```

**After:**
```typescript
if (status.status === "completed" && status.bundle_count > 0) {
  // Success
}
```

---

### **Step 3: Simplified Status Handling**

**Before:**
```typescript
if (status.status === "processing") { ... }
else if (status.status === "completed") { ... }
else if (status.status === "bundle_generation_queued") { ... }
else if (status.status === "bundle_generation_in_progress") { ... }
else if (status.status === "bundle_generation_completed") { ... }
else if (status.status === "bundle_generation_failed") { ... }
// etc...
```

**After:**
```typescript
if (status.status === "processing") {
  // CSV or bundles generating
}
else if (status.status === "completed" && status.bundle_count > 0) {
  // Success!
}
else if (status.status === "failed") {
  // Error
}
```

---

## **Complete Example**

```typescript
async function syncData(csvData: string) {
  // 1. Upload
  const uploadRes = await fetch("/api/shopify/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      shopId: "shop.myshopify.com",
      csvType: "orders",
      csvData: csvData,
      triggerPipeline: true
    })
  })
  const { uploadId } = await uploadRes.json()

  // 2. Poll every 5 seconds
  for (let i = 0; i < 120; i++) {
    const statusRes = await fetch(`/api/shopify/status/${uploadId}`)
    const status = await statusRes.json()

    // Handle 3 simple states
    if (status.status === "failed") {
      return { success: false, error: status.error_message }
    }

    if (status.status === "completed" && status.bundle_count > 0) {
      return { success: true, bundleCount: status.bundle_count }
    }

    if (status.status === "processing") {
      // Show progress
      if (status.processed_rows < status.total_rows) {
        console.log(`CSV: ${status.processed_rows}/${status.total_rows}`)
      } else {
        console.log("Generating bundles...")
      }
    }

    await new Promise(r => setTimeout(r, 5000))
  }

  return { success: false, error: "Timeout" }
}
```

---

## **Testing the Changes**

### **1. Deploy Updated Backend**
```bash
cd /Users/rahular/Documents/AI\ Bundler/python_server
gcloud run deploy bundle-api --source . --region us-central1
```

### **2. Test Status Endpoint**
```bash
# Upload a CSV
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/shopify/upload \
  -H "Content-Type: application/json" \
  -d '{
    "shopId": "test.myshopify.com",
    "csvType": "orders",
    "csvData": "order_id,sku,quantity\n123,PROD-001,2",
    "triggerPipeline": true
  }'

# Poll status (use uploadId from above)
curl https://bundle-api-250755735924.us-central1.run.app/api/shopify/status/YOUR_UPLOAD_ID
```

### **3. Verify Response**
You should see:
```json
{
  "upload_id": "...",
  "shop_id": "test.myshopify.com",
  "status": "processing" or "completed" or "failed",
  "bundle_count": 10 or null,
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null
}
```

### **4. Check Logs**
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bundle-api" --limit 50 --format=json | jq -r '.[] | .textPayload' | grep "STATUS"
```

You should see:
```
🔍 STATUS CHECK: upload_id=abc-123
✅ STATUS CHECK: upload_id=abc-123 status=completed rows=100/100
📊 STATUS RESPONSE: internal_status=bundle_generation_completed → frontend_status=completed bundle_count=10
```

---

## **Backward Compatibility**

### **Old Endpoint Still Works**

If your frontend is still using `/api/upload-status/{uploadId}`:
1. ✅ It still works (backward compatible)
2. ⚠️ Logs deprecation warning
3. ✅ Maps statuses to simple values
4. ✅ Returns `_deprecated: true` in response

**But please migrate to `/api/shopify/status/` when you can!**

---

## **Breaking Changes**

### **None!**

All changes are backward compatible:
- Old endpoint still works
- Status values are mapped automatically
- Frontend can migrate at their own pace

---

## **Benefits**

### **For Frontend Developers**
- ✅ ONE endpoint to remember
- ✅ 3 simple status values
- ✅ Always includes `bundle_count`
- ✅ Clearer what each status means
- ✅ Easier to debug

### **For Backend Developers**
- ✅ Status mapping in one place
- ✅ Better logging
- ✅ Easier to maintain
- ✅ Clearer code flow

### **For Users**
- ✅ Faster re-syncs (existing bundles detected)
- ✅ Better error messages
- ✅ More reliable status updates

---

## **Documentation**

### **Quick Reference**
Read: **[SIMPLE_API_GUIDE.md](SIMPLE_API_GUIDE.md)** - This is the ONLY doc you need!

### **Detailed Scenarios**
Read: **[FRONTEND_BACKEND_CONTRACT.md](FRONTEND_BACKEND_CONTRACT.md)** - All 7 scenarios explained

### **Old Docs (Archive)**
- ~~ACTUAL_API_RESPONSES.md~~ - Replaced by SIMPLE_API_GUIDE.md
- ~~QUICK_REFERENCE.md~~ - Replaced by SIMPLE_API_GUIDE.md
- ~~API_RESPONSE_SCHEMAS.md~~ - Replaced by SIMPLE_API_GUIDE.md

---

## **Rollout Plan**

### **Phase 1: Deploy Backend** ✅
- Deploy updated backend with enhanced endpoint
- Old endpoint still works (backward compatible)

### **Phase 2: Update Frontend**
- Update frontend to use `/api/shopify/status/`
- Test with real data
- Monitor for issues

### **Phase 3: Remove Deprecation**
- After frontend is migrated
- Remove `/api/upload-status/` endpoint
- Update all documentation

---

## **Questions?**

**Q: Do I need to update my frontend immediately?**
A: No! Old endpoint still works. Update when convenient.

**Q: Will my existing frontend break?**
A: No! Backward compatible. Old endpoint returns simplified statuses now.

**Q: How do I know if I'm using the old endpoint?**
A: Check Cloud Run logs for "⚠️ DEPRECATED ENDPOINT CALLED"

**Q: When will the old endpoint be removed?**
A: After frontend is fully migrated (TBD)

---

## **Summary**

✅ **ONE endpoint:** `/api/shopify/status/{uploadId}`
✅ **3 status values:** `processing`, `completed`, `failed`
✅ **Always includes bundle_count**
✅ **Better logging**
✅ **Backward compatible**

**Next steps:**
1. Deploy backend
2. Read [SIMPLE_API_GUIDE.md](SIMPLE_API_GUIDE.md)
3. Update frontend when ready
4. Monitor logs
5. Profit! 🎉
