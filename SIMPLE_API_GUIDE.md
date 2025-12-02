# Simple API Guide - Bundle Generation

**ONE endpoint to rule them all: `/api/shopify/status/{uploadId}`**

---

## **How It Works**

1. Upload CSV → Get `uploadId`
2. Poll `/api/shopify/status/{uploadId}` every 5 seconds
3. When `bundle_count > 0` → Done!

---

## **Step 1: Upload CSV**

**Endpoint:** `POST /api/shopify/upload`

**Request:**
```json
{
  "shopId": "shop.myshopify.com",
  "csvType": "orders",
  "csvData": "order_id,sku,quantity\n123,PROD-001,2\n...",
  "triggerPipeline": true
}
```

**Response:** (Returns immediately)
```json
{
  "uploadId": "abc-123-def-456",
  "status": "processing",
  "message": "Upload accepted. Poll /api/shopify/status/{uploadId} for progress."
}
```

---

## **Step 2: Check Status**

**Endpoint:** `GET /api/shopify/status/{uploadId}`

**This ONE endpoint tells you everything:**
- CSV processing progress
- Bundle generation status
- When bundles are ready
- Error messages

---

## **Step 3: Poll Until Ready**

### **Response During CSV Processing** (5-15 seconds)
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "processing",
  "total_rows": 100,
  "processed_rows": 45,
  "error_message": null,
  "bundle_count": null
}
```
**Frontend shows:** "Processing CSV: 45/100 rows"

---

### **Response: CSV Done, Bundles Generating** (30-120 seconds)
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "processing",  // Still processing!
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": null  // Still null
}
```
**Frontend shows:** "Generating bundles..."

---

### **Response: Bundles Ready!** ✅
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "completed",  // ← Done!
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": 10  // ← Bundles ready!
}
```
**Frontend shows:** "Success! 10 bundles created"

---

### **Response: Failed** ❌
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "failed",
  "error_message": "Missing required: order_id",
  "bundle_count": null
}
```
**Frontend shows:** "Upload failed: Missing 'order_id' column"

---

### **Response: Re-sync (Bundles Already Exist)** ⚡
```json
{
  "upload_id": "xyz-789",
  "status": "completed",  // Fast!
  "bundle_count": 10  // ← Immediate!
}
```
**Frontend shows:** "Data synced! Using existing bundles"

---

## **Status Values (Simple!)**

| Status | Meaning | What Frontend Does |
|--------|---------|-------------------|
| `"processing"` | CSV or bundles being generated | Show spinner + progress |
| `"completed"` | Everything done | Check `bundle_count` |
| `"failed"` | Something went wrong | Show error message |

**That's it! Only 3 status values.**

---

## **Complete Frontend Code**

```typescript
async function syncAndWaitForBundles(csvData: string) {
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
  for (let i = 0; i < 120; i++) {  // 10 minutes max
    const statusRes = await fetch(`/api/shopify/status/${uploadId}`)
    const status = await statusRes.json()

    // Failed
    if (status.status === "failed") {
      throw new Error(status.error_message || "Upload failed")
    }

    // Success!
    if (status.status === "completed" && status.bundle_count > 0) {
      return {
        success: true,
        bundleCount: status.bundle_count,
        shopId: status.shop_id
      }
    }

    // Still processing
    if (status.status === "processing") {
      if (status.processed_rows < status.total_rows) {
        console.log(`CSV: ${status.processed_rows}/${status.total_rows}`)
      } else {
        console.log("Generating bundles...")
      }
    }

    // Wait 5 seconds
    await new Promise(r => setTimeout(r, 5000))
  }

  throw new Error("Timeout after 10 minutes")
}
```

---

## **UI State Machine**

```typescript
type SyncState =
  | { state: "uploading" }
  | { state: "csv_processing", current: number, total: number }
  | { state: "bundles_generating" }
  | { state: "success", count: number }
  | { state: "error", message: string }

function getUIMessage(state: SyncState): string {
  switch (state.state) {
    case "uploading":
      return "Uploading data..."
    case "csv_processing":
      return `Processing CSV: ${state.current}/${state.total} rows`
    case "bundles_generating":
      return "Generating bundle recommendations..."
    case "success":
      return `Success! ${state.count} bundles created`
    case "error":
      return `Error: ${state.message}`
  }
}
```

---

## **All Scenarios**

| Scenario | Time | `bundle_count` When |
|----------|------|-------------------|
| **First sync** | 35-135s | After delay |
| **Re-sync** | 5-15s | Immediate |
| **CSV fails** | 2-5s | Never (status="failed") |
| **Bundles fail** | 15-45s | Never + error message |

---

## **Key Rules**

### ✅ **DO:**
- Poll `/api/shopify/status/{uploadId}` every 5 seconds
- Check BOTH `status === "completed"` AND `bundle_count > 0`
- Allow 10 minutes timeout
- Show user-friendly error messages

### ❌ **DON'T:**
- Use `/api/upload-status/` (deprecated!)
- Check only `status === "completed"` (bundles might not be ready)
- Stop polling after 30 seconds
- Show raw backend errors to users

---

## **Error Message Translations**

| Backend Error | Show to User |
|--------------|--------------|
| `"Missing required: order_id"` | "Your CSV is missing the 'order_id' column. Please check your export." |
| `"Could not detect CSV type"` | "Invalid CSV format. Please ensure you're exporting the correct data from Shopify." |
| `"Insufficient order history"` | "We need more order history to create bundles. Keep selling and try again in a few weeks!" |
| `"Variants missing for SKUs"` | "Some products don't have variant data. Please sync products first." |

---

## **Step 4: Get Bundles (Optional)**

If you want to fetch the actual bundle data:

**Endpoint:** `GET /api/shopify/recommendations?shopId={shopId}&limit=50`

**Response:**
```json
{
  "shop_id": "shop.myshopify.com",
  "count": 10,
  "recommendations": [
    {
      "id": "bundle-001",
      "discount_type": "fbt",
      "primary_sku": "PROD-001",
      "companion_sku": "PROD-002",
      "discount_percentage": 10.0,
      "confidence": 0.85,
      "is_approved": false
    }
    // ... more bundles
  ]
}
```

---

## **Testing**

### **Test 1: Happy Path**
```bash
# Upload
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/shopify/upload \
  -H "Content-Type: application/json" \
  -d '{
    "shopId": "test.myshopify.com",
    "csvType": "orders",
    "csvData": "order_id,sku,quantity\n123,PROD-001,2",
    "triggerPipeline": true
  }'

# Get uploadId from response, then poll:
curl https://bundle-api-250755735924.us-central1.run.app/api/shopify/status/YOUR_UPLOAD_ID
```

### **Test 2: Re-sync**
Upload CSV again with same shop → should complete quickly with existing bundles

### **Test 3: Invalid CSV**
Upload CSV missing required columns → should fail within 5 seconds

---

## **Summary: The Contract**

**Backend promises:**
1. Returns `uploadId` immediately from POST
2. Updates status to `"completed"` when everything done
3. Sets `bundle_count` when bundles exist
4. Only 3 status values: `"processing"`, `"completed"`, `"failed"`

**Frontend must:**
1. Poll `/api/shopify/status/{uploadId}` every 5 seconds
2. Check `status === "completed"` AND `bundle_count > 0` for success
3. Allow 10 minutes timeout
4. Handle all 3 status values

---

## **What Changed?**

### ✅ **Improvements:**
- **ONE endpoint** instead of two confusing ones
- **Simplified status** values (3 instead of 10)
- **Always includes** `bundle_count` (checks bundles on every call)
- **Maps internal statuses** to simple frontend-friendly values
- **Better logging** to debug issues

### ⚠️ **Deprecated:**
- `/api/upload-status/{uploadId}` - Still works but shows deprecation warning
- Use `/api/shopify/status/{uploadId}` instead

---

## **Questions?**

1. **"Why is status still 'processing' even though CSV is done?"**
   - Bundles are still generating. Check `bundle_count` - it will be `null` until bundles exist.

2. **"bundle_count is null but status is completed?"**
   - This shouldn't happen anymore with the new code. If it does, check logs.

3. **"Which endpoint should I use?"**
   - Use `/api/shopify/status/{uploadId}` - it's the ONLY one you need.

4. **"How long should I wait?"**
   - First sync: 35-135 seconds
   - Re-sync: 5-15 seconds
   - Allow 10 minutes max before timeout

---

## **Next Steps**

1. **Deploy updated backend:**
   ```bash
   cd /Users/rahular/Documents/AI\ Bundler/python_server
   gcloud run deploy bundle-api --source . --region us-central1
   ```

2. **Update frontend** to use `/api/shopify/status/{uploadId}`

3. **Test the happy path** with real data

4. **Monitor Cloud Run logs** - you'll see helpful debug messages

---

## **Need Help?**

Check Cloud Run logs for these helpful messages:
- `🔍 STATUS CHECK: upload_id=...` - Status endpoint called
- `📊 STATUS RESPONSE: ... bundle_count=10` - What was returned
- `⚠️ DEPRECATED ENDPOINT CALLED` - You're using old endpoint

All logs include upload_id so you can trace a specific upload.
