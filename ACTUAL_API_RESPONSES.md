# ACTUAL API RESPONSES - TESTED

**⚠️ CRITICAL: This documents the ACTUAL backend responses, not assumptions!**

---

## **Discovery: Two Different Status Endpoints!**

### **Endpoint 1: `/api/upload-status/{upload_id}`** (Main endpoint)
**Implementation:** [routers/uploads.py:137-154](routers/uploads.py#L137-L154)

**Response:**
```json
{
  "id": "abc-123",
  "filename": "orders.csv",
  "csvType": "orders",
  "runId": "run-456",
  "status": "bundle_generation_completed",  // ← Note: NOT just "completed"!
  "totalRows": 100,
  "processedRows": 100,
  "errorMessage": null
}
```

**❌ DOES NOT INCLUDE `bundle_count`!**

---

### **Endpoint 2: `/api/shopify/status/{upload_id}`** (Shopify-specific)
**Implementation:** [routers/shopify_upload.py:161-185](routers/shopify_upload.py#L161-L185)

**Response:**
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "completed",  // ← Different status values!
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": 10  // ✅ Includes bundle_count
}
```

---

## **ACTUAL Status Values**

### **For `/api/upload-status/{upload_id}`:**

Based on [services/storage.py:414](services/storage.py#L414):

| Status | Meaning |
|--------|---------|
| `"processing"` | CSV is being ingested |
| `"completed"` | CSV ingestion complete |
| `"failed"` | CSV ingestion failed |
| `"bundle_generation_queued"` | Bundles queued to generate |
| `"bundle_generation_in_progress"` | Bundles generating |
| `"bundle_generation_completed"` | ✅ Bundles ready! |
| `"bundle_generation_failed"` | ❌ Bundle generation failed |
| `"bundle_generation_timed_out"` | ⏱️ Generation timeout |
| `"bundle_generation_cancelled"` | 🛑 Cancelled |
| `"bundle_generation_async"` | Running in background |

### **For `/api/shopify/status/{upload_id}`:**

Based on [routers/shopify_upload.py:161-185](routers/shopify_upload.py#L161-L185):

| Status | Meaning |
|--------|---------|
| `"processing"` | CSV being ingested |
| `"completed"` | CSV complete (check `bundle_count` for bundles) |
| `"failed"` | Upload failed |

---

## **Test Commands**

### **Test 1: Check what YOUR backend returns**

```bash
# After uploading a CSV, use the ACTUAL upload ID
UPLOAD_ID="your-upload-id-here"

# Test endpoint 1 (main endpoint)
curl https://bundle-api-250755735924.us-central1.run.app/api/upload-status/${UPLOAD_ID}

# Expected response:
# {
#   "id": "...",
#   "filename": "...",
#   "csvType": "orders",
#   "status": "bundle_generation_completed",  ← Check this value!
#   "totalRows": 100,
#   "processedRows": 100,
#   "errorMessage": null
# }

# Test endpoint 2 (Shopify endpoint - if using Shopify integration)
curl https://bundle-api-250755735924.us-central1.run.app/api/shopify/status/${UPLOAD_ID}

# Expected response:
# {
#   "upload_id": "...",
#   "shop_id": "shop.myshopify.com",
#   "status": "completed",
#   "bundle_count": 10  ← Only in Shopify endpoint!
# }
```

---

## **CORRECTED Frontend Polling Logic**

### **Option A: Using `/api/upload-status/{uploadId}` (Main Endpoint)**

```typescript
async function pollUntilBundlesReady(uploadId: string) {
  const MAX_POLLS = 120
  const POLL_INTERVAL = 5000

  for (let i = 0; i < MAX_POLLS; i++) {
    const response = await fetch(`/api/upload-status/${uploadId}`)
    const data = await response.json()

    // CSV processing
    if (data.status === "processing") {
      showProgress(`Processing CSV: ${data.processedRows}/${data.totalRows}`)
      await sleep(POLL_INTERVAL)
      continue
    }

    // CSV failed
    if (data.status === "failed") {
      showError(data.errorMessage)
      return
    }

    // CSV complete
    if (data.status === "completed") {
      showMessage("CSV complete, bundles generating...")
      await sleep(POLL_INTERVAL)
      continue
    }

    // Bundles generating
    if (data.status === "bundle_generation_in_progress" ||
        data.status === "bundle_generation_queued" ||
        data.status === "bundle_generation_async") {
      showMessage("Generating bundles...")
      await sleep(POLL_INTERVAL)
      continue
    }

    // ✅ SUCCESS!
    if (data.status === "bundle_generation_completed") {
      // Must fetch bundles separately - bundle_count NOT in this endpoint!
      const bundles = await fetchBundles(data.csvType)
      showSuccess(`${bundles.length} bundles ready!`)
      return
    }

    // ❌ FAILED
    if (data.status === "bundle_generation_failed" ||
        data.status === "bundle_generation_timed_out" ||
        data.status === "bundle_generation_cancelled") {
      showError("Bundle generation failed")
      return
    }

    await sleep(POLL_INTERVAL)
  }

  showError("Timeout after 10 minutes")
}
```

### **Option B: Using `/api/shopify/status/{uploadId}` (Shopify Endpoint)**

**⚠️ Only use if you're using the Shopify integration route!**

```typescript
async function pollUntilBundlesReady(uploadId: string) {
  const MAX_POLLS = 120
  const POLL_INTERVAL = 5000

  for (let i = 0; i < MAX_POLLS; i++) {
    const response = await fetch(`/api/shopify/status/${uploadId}`)
    const data = await response.json()

    // Failed
    if (data.status === "failed") {
      showError(data.error_message)
      return
    }

    // CSV processing
    if (data.status === "processing") {
      showProgress(`Processing: ${data.processed_rows}/${data.total_rows}`)
      await sleep(POLL_INTERVAL)
      continue
    }

    // Completed
    if (data.status === "completed") {
      // Check if bundles exist
      if (data.bundle_count > 0) {
        // ✅ SUCCESS!
        showSuccess(`${data.bundle_count} bundles ready!`)
        return
      } else {
        // Bundles still generating
        showMessage("Generating bundles...")
        await sleep(POLL_INTERVAL)
        continue
      }
    }

    await sleep(POLL_INTERVAL)
  }

  showError("Timeout")
}
```

---

## **Which Endpoint Should Frontend Use?**

### **Use `/api/upload-status/{uploadId}` if:**
- ✅ You're uploading via `/api/upload-csv`
- ✅ You need detailed bundle generation status
- ✅ You're NOT using Shopify integration

**Pros:**
- More detailed status values
- Shows exact generation stage

**Cons:**
- Does NOT include `bundle_count`
- Must fetch bundles separately

---

### **Use `/api/shopify/status/{uploadId}` if:**
- ✅ You're uploading via `/api/shopify/upload`
- ✅ You want simpler polling logic
- ✅ You need `bundle_count` in status response

**Pros:**
- Includes `bundle_count` directly
- Simpler status values

**Cons:**
- Less detailed status (just "processing", "completed", "failed")

---

## **How to Fetch Bundles**

### **After `/api/upload-status/{uploadId}` says `bundle_generation_completed`:**

```typescript
// Get the shop_id from the upload response
const shopId = "shop.myshopify.com"

// Fetch bundles
const response = await fetch(
  `/api/shopify/recommendations?shopId=${shopId}&limit=50`
)
const data = await response.json()

console.log(`Got ${data.count} bundles`)
// {
//   "shop_id": "shop.myshopify.com",
//   "count": 10,
//   "recommendations": [...]
// }
```

### **After `/api/shopify/status/{uploadId}` says bundles ready:**

```typescript
// shop_id is already in the status response
const response = await fetch(
  `/api/shopify/recommendations?shopId=${status.shop_id}&limit=50`
)
const data = await response.json()
```

---

## **Updated State Machine**

### **For `/api/upload-status/{uploadId}`:**

```
INITIAL
   ↓
UPLOADING
   ↓
status: "processing"
   ↓
status: "completed"  (CSV done)
   ↓
status: "bundle_generation_queued"
   ↓
status: "bundle_generation_in_progress"
   ↓
status: "bundle_generation_completed" ✅
   ↓
Fetch bundles via /api/shopify/recommendations
```

### **For `/api/shopify/status/{uploadId}`:**

```
INITIAL
   ↓
UPLOADING
   ↓
status: "processing"
   ↓
status: "completed" + bundle_count: null
   ↓
[Poll again]
   ↓
status: "completed" + bundle_count: 10 ✅
   ↓
Bundles ready!
```

---

## **CRITICAL: Test Your Actual Flow**

Run this after uploading a CSV:

```bash
#!/bin/bash
UPLOAD_ID="paste-your-upload-id-here"
BACKEND="https://bundle-api-250755735924.us-central1.run.app"

echo "Testing /api/upload-status/${UPLOAD_ID}"
curl "${BACKEND}/api/upload-status/${UPLOAD_ID}" | jq '.'

echo ""
echo "Testing /api/shopify/status/${UPLOAD_ID}"
curl "${BACKEND}/api/shopify/status/${UPLOAD_ID}" | jq '.'
```

**Check:**
1. Which endpoint returns what you need?
2. What status values do you see?
3. Does `bundle_count` appear where you expect?

---

## **Summary: Key Differences**

| Feature | `/api/upload-status/{id}` | `/api/shopify/status/{id}` |
|---------|---------------------------|---------------------------|
| **Status values** | `bundle_generation_completed` | `completed` |
| **Includes bundle_count** | ❌ No | ✅ Yes |
| **Detail level** | High (10+ states) | Low (3 states) |
| **Best for** | Detailed tracking | Simple polling |
| **Use when** | Using `/api/upload-csv` | Using `/api/shopify/upload` |

---

## **Action Items**

1. **Test which endpoint you're actually using**
   - Upload a CSV via your frontend
   - Check network tab - which endpoint is being polled?
   - Check the response format

2. **Update frontend polling logic**
   - If using `/api/upload-status/`, check for `bundle_generation_completed`
   - If using `/api/shopify/status/`, check for `completed` + `bundle_count > 0`

3. **Fix status value checks**
   - Don't assume status values - use the ones documented above
   - Test with ACTUAL backend responses

4. **Handle bundle fetching**
   - If using `/api/upload-status/`, must call `/api/shopify/recommendations` separately
   - If using `/api/shopify/status/`, `bundle_count` tells you when ready

---

## **Next Steps**

1. Run the test commands above with a real upload ID
2. Paste the actual responses here
3. I'll update the frontend code to match YOUR actual backend
