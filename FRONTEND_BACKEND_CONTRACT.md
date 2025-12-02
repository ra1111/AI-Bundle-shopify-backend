# Frontend ↔ Backend Contract: All Scenarios

This document maps out ALL possible paths for CSV upload → bundle generation and defines exactly what the backend sends and what the frontend should do in each scenario.

---

## **API Endpoints Overview**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/shopify/upload` | Upload CSV data (returns immediately) |
| GET | `/api/shopify/status/{uploadId}` | Check CSV upload status |
| GET | `/generation-progress/{uploadId}` | Check bundle generation progress (detailed) |
| GET | `/api/shopify/recommendations?shopId={shopId}` | Get final bundles |

---

## **Status Values**

### **CsvUpload Status** (from `/api/shopify/status/{uploadId}`)
- `"processing"` - CSV is being ingested
- `"completed"` - CSV ingestion complete
- `"failed"` - CSV ingestion failed

### **Bundle Generation Status** (from `/generation-progress/{uploadId}`)
- `"queued"` - Waiting to start
- `"in_progress"` - Generating bundles
- `"completed"` - Bundles ready
- `"failed"` - Generation failed

---

## **Scenario 1: First-Time Install → Quick Mode Success**

### **Flow**
1. User uploads CSV from Remix app
2. Backend detects: first install + no existing bundles
3. Quick generation runs (~30-120s)
4. Quick bundles saved to DB
5. Upload marked as completed
6. **NO automatic full generation** (we disabled this)

### **Backend → Frontend**

#### **Step 1: POST /api/shopify/upload**
**Request:**
```json
{
  "shopId": "shop.myshopify.com",
  "csvType": "orders",
  "csvData": "order_id,sku,quantity...",
  "triggerPipeline": true
}
```

**Response:** (Returns immediately, ~50ms)
```json
{
  "uploadId": "abc-123",
  "runId": "run-456",
  "status": "processing",
  "shopId": "shop.myshopify.com",
  "csvType": "orders",
  "triggerPipeline": true,
  "message": "Upload accepted. Poll /api/shopify/status/{uploadId} for progress."
}
```

#### **Step 2: Frontend Polls GET /api/shopify/status/{uploadId}**

**Poll 1-3:** (CSV ingestion in progress, 0-10s)
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

**Poll 4-8:** (CSV done, quick bundles generating, 10-120s)
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "completed",  // CSV done!
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": null  // Bundles not ready yet
}
```

**Poll 9+:** (Quick bundles ready, 30-120s)
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "completed",
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": 10  // ✅ Bundles ready!
}
```

#### **Step 3: Frontend Gets Bundles**
**Request:** `GET /api/shopify/recommendations?shopId=shop.myshopify.com&limit=50`

**Response:**
```json
{
  "shop_id": "shop.myshopify.com",
  "count": 10,
  "recommendations": [
    {
      "id": "bundle-1",
      "discount_type": "fbt",
      "primary_sku": "PROD-001",
      "companion_sku": "PROD-002",
      "discount_percentage": 10.0,
      "confidence": 0.85,
      "ranking_score": 0.92,
      "is_approved": false,
      "created_at": "2025-12-02T10:30:00Z"
    }
    // ... 9 more bundles
  ]
}
```

### **Frontend Actions**

| Stage | Backend Signals | Frontend Action |
|-------|----------------|-----------------|
| **Upload submitted** | `uploadId` returned | Show "Uploading..." spinner |
| **CSV processing** | `status: "processing"` | Show "Processing CSV... 45/100 rows" |
| **CSV complete** | `status: "completed"`, `bundle_count: null` | Show "Generating bundles..." |
| **Bundles ready** | `status: "completed"`, `bundle_count: 10` | Stop polling, show "Success! 10 bundles created" |
| **Fetch bundles** | Call `/recommendations` | Display bundles in UI |

### **Frontend Polling Logic**
```typescript
async function pollUntilBundlesReady(uploadId: string) {
  const MAX_POLLS = 60  // 5 minutes
  const POLL_INTERVAL = 5000  // 5 seconds

  for (let i = 0; i < MAX_POLLS; i++) {
    const response = await fetch(`/api/shopify/status/${uploadId}`)
    const data = await response.json()

    // CSV still processing
    if (data.status === "processing") {
      updateUI(`Processing CSV: ${data.processed_rows}/${data.total_rows} rows`)
      await sleep(POLL_INTERVAL)
      continue
    }

    // CSV failed
    if (data.status === "failed") {
      showError(data.error_message)
      return
    }

    // CSV done, check bundles
    if (data.status === "completed") {
      if (data.bundle_count > 0) {
        // ✅ SUCCESS!
        showSuccess(`${data.bundle_count} bundles ready!`)
        loadBundles(data.shop_id)
        return
      } else {
        // Bundles still generating
        updateUI("Generating bundles...")
        await sleep(POLL_INTERVAL)
        continue
      }
    }
  }

  showError("Timeout: Bundle generation took too long")
}
```

---

## **Scenario 2: First-Time Install → Bundles Already Exist (Re-sync)**

### **Flow**
1. User already did quick generation before
2. They sync data again (new CSV upload)
3. Backend detects existing quick bundles
4. **Returns immediately** - no re-generation

### **Backend → Frontend**

#### **Step 1: POST /api/shopify/upload**
Same as Scenario 1.

#### **Step 2: Poll GET /api/shopify/status/{uploadId}**

**Poll 1-2:** (CSV processing, 0-10s)
```json
{
  "upload_id": "xyz-789",
  "status": "processing",
  "total_rows": 150,
  "processed_rows": 75,
  "error_message": null,
  "bundle_count": null
}
```

**Poll 3+:** (CSV done, existing bundles detected)
```json
{
  "upload_id": "xyz-789",
  "status": "completed",
  "total_rows": 150,
  "processed_rows": 150,
  "error_message": null,
  "bundle_count": 10  // Existing bundles!
}
```

**Generation Progress** (optional check at `/generation-progress/{uploadId}`):
```json
{
  "upload_id": "xyz-789",
  "status": "completed",
  "progress": 100,
  "step": "finalization",
  "message": "Bundle recommendations already exist (10 bundles)",
  "created_at": "2025-12-02T10:00:00Z",
  "updated_at": "2025-12-02T10:00:15Z"
}
```

### **Frontend Actions**

| Stage | Backend Signals | Frontend Action |
|-------|----------------|-----------------|
| **CSV processing** | `status: "processing"` | Show "Processing CSV..." |
| **CSV complete + existing bundles** | `status: "completed"`, `bundle_count: 10` immediately | Show "Data synced! Using existing 10 bundles" |

**Note:** The key difference from Scenario 1 is that `bundle_count` appears **immediately** when CSV is done, not after a delay.

---

## **Scenario 3: Quick Mode Timeout (Rare)**

### **Flow**
1. Quick generation takes > 120s (timeout threshold)
2. Backend logs timeout, falls back to normal generation
3. Normal generation completes (5-20 min)

### **Backend → Frontend**

#### **Polls 1-20:** (Quick mode timeout after 120s)
```json
{
  "upload_id": "timeout-123",
  "status": "processing",  // Still processing!
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": null
}
```

**Backend logs** (visible in Cloud Run):
```
⏱️ TIMEOUT: Quick-start exceeded 120s!
   Actual duration: 125000ms
   Falling back to full generation pipeline
```

#### **Polls 21+:** (Normal generation completes, 5-20 min later)
```json
{
  "upload_id": "timeout-123",
  "status": "completed",
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": 25  // More bundles from full generation
}
```

### **Frontend Actions**

| Stage | Backend Signals | Frontend Action |
|-------|----------------|-----------------|
| **2+ min, still processing** | `status: "processing"` after 120s | Show "This is taking longer than expected. We're running a comprehensive analysis..." |
| **Eventually completes** | `status: "completed"`, `bundle_count: 25` | Show "Success! 25 bundles created" |

---

## **Scenario 4: Quick Mode Fails → Fallback to Normal Generation**

### **Flow**
1. Quick generation throws error
2. Backend catches error, logs it
3. Falls back to normal generation
4. Normal generation completes

### **Backend → Frontend**

Same as Scenario 3 - from frontend perspective, it just takes longer.

**Backend logs** (Cloud Run):
```
❌ Quick-start FAILED after 15000ms!
   Error type: ValueError
   Error message: Insufficient data for FBT bundles
   Falling back to full generation pipeline
```

**Frontend sees:**
```json
{
  "status": "processing",  // Takes longer than quick mode
  "bundle_count": null
}
```

Then eventually:
```json
{
  "status": "completed",
  "bundle_count": 20
}
```

---

## **Scenario 5: CSV Ingestion Fails**

### **Flow**
1. CSV has invalid format, missing columns, or data errors
2. CSV processor fails
3. Upload marked as "failed"

### **Backend → Frontend**

#### **Poll 1-3:**
```json
{
  "upload_id": "bad-csv-123",
  "status": "processing",
  "total_rows": 0,
  "processed_rows": 0,
  "error_message": null,
  "bundle_count": null
}
```

#### **Poll 4+:** (CSV failed)
```json
{
  "upload_id": "bad-csv-123",
  "status": "failed",
  "total_rows": 50,
  "processed_rows": 23,
  "error_message": "orders.csv schema error → Missing required: order_id",
  "bundle_count": null
}
```

### **Frontend Actions**

| Stage | Backend Signals | Frontend Action |
|-------|----------------|-----------------|
| **CSV failed** | `status: "failed"` | Show error: "Upload failed: Missing required column 'order_id'" |
| **User action** | - | Show "Try Again" button |

### **Error Messages to Show User**

| Backend Error | User-Friendly Message |
|--------------|----------------------|
| `"Missing required: order_id"` | "Your CSV is missing the 'order_id' column. Please check your export settings." |
| `"Could not detect CSV type from headers"` | "We couldn't identify your CSV format. Please ensure you're exporting the correct data from Shopify." |
| `"CSV file is empty"` | "The uploaded file is empty. Please export your data and try again." |
| `"Variants missing for SKUs: ..."` | "Some products in your orders don't have matching variants. Please sync product data first." |

---

## **Scenario 6: Bundle Generation Fails (After CSV Success)**

### **Flow**
1. CSV ingestion succeeds
2. Bundle generation starts
3. Generation fails (e.g., insufficient data, DB error)

### **Backend → Frontend**

#### **CSV Success:**
```json
{
  "upload_id": "gen-fail-123",
  "status": "completed",
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": null
}
```

#### **Check Generation Progress:** `GET /generation-progress/{uploadId}`
```json
{
  "upload_id": "gen-fail-123",
  "status": "failed",
  "progress": 45,
  "step": "optimization",
  "message": "Insufficient order history: need 10+ orders, found 3",
  "created_at": "2025-12-02T10:00:00Z",
  "updated_at": "2025-12-02T10:02:30Z"
}
```

### **Frontend Actions**

| Stage | Backend Signals | Frontend Action |
|-------|----------------|-----------------|
| **CSV done, bundles failed** | `/status`: `completed`, `/generation-progress`: `failed` | Show "Data uploaded successfully, but we need more order history to create bundles. Keep selling and try again in a few weeks!" |

---

## **Scenario 7: Not First-Time Install (Subsequent Syncs)**

### **Flow**
1. Shop has synced before (not first install)
2. Quick mode **does NOT run** (only for first install)
3. CSV ingestion completes
4. **No automatic bundle generation** (user must trigger manually)

### **Backend → Frontend**

#### **Step 1: POST /api/shopify/upload**
```json
{
  "triggerPipeline": false  // ← Key difference!
}
```

#### **Step 2: Poll GET /api/shopify/status/{uploadId}**

**Poll 1-3:**
```json
{
  "status": "processing",
  "total_rows": 200,
  "processed_rows": 150
}
```

**Poll 4+:** (CSV done, no bundles triggered)
```json
{
  "status": "completed",
  "total_rows": 200,
  "processed_rows": 200,
  "error_message": null,
  "bundle_count": 10  // Existing bundles from before
}
```

### **Frontend Actions**

| Stage | Backend Signals | Frontend Action |
|-------|----------------|-----------------|
| **CSV done** | `status: "completed"` quickly | Show "Data synced successfully!" |
| **Bundles** | `bundle_count: 10` (existing) | Show existing bundles, no spinner |
| **Manual trigger** | User clicks "Regenerate Bundles" | Call separate endpoint to trigger full generation |

---

## **Complete Frontend State Machine**

```typescript
type SyncState =
  | { stage: "uploading" }
  | { stage: "csv_processing", progress: { current: number, total: number } }
  | { stage: "csv_failed", error: string }
  | { stage: "bundles_generating" }
  | { stage: "bundles_ready", count: number }
  | { stage: "bundles_failed", error: string }
  | { stage: "data_synced_no_bundles" }

async function syncData(csvData: string): Promise<SyncState> {
  // 1. Upload
  const { uploadId } = await uploadCSV(csvData)

  // 2. Poll status
  while (true) {
    const status = await checkStatus(uploadId)

    // CSV processing
    if (status.status === "processing") {
      return {
        stage: "csv_processing",
        progress: { current: status.processed_rows, total: status.total_rows }
      }
    }

    // CSV failed
    if (status.status === "failed") {
      return { stage: "csv_failed", error: status.error_message }
    }

    // CSV succeeded
    if (status.status === "completed") {
      // Bundles ready immediately (scenario 2 or 7)
      if (status.bundle_count > 0) {
        return { stage: "bundles_ready", count: status.bundle_count }
      }

      // No bundles, might be generating
      const genProgress = await checkGenerationProgress(uploadId)

      if (genProgress.status === "completed") {
        return { stage: "bundles_ready", count: status.bundle_count }
      }

      if (genProgress.status === "failed") {
        return { stage: "bundles_failed", error: genProgress.message }
      }

      if (genProgress.status === "in_progress") {
        return { stage: "bundles_generating" }
      }

      // No generation in progress (scenario 7)
      return { stage: "data_synced_no_bundles" }
    }

    await sleep(5000)
  }
}
```

---

## **UI Messages for Each State**

| State | Message | Action |
|-------|---------|--------|
| `uploading` | "Uploading data..." | Spinner |
| `csv_processing` | "Processing CSV: 45/100 rows" | Progress bar |
| `csv_failed` | "Upload failed: [error]" | "Try Again" button |
| `bundles_generating` | "Generating bundle recommendations..." | Spinner |
| `bundles_ready` | "Success! 10 bundles created" | "View Bundles" button |
| `bundles_failed` | "Need more data: [error]" | "Learn More" link |
| `data_synced_no_bundles` | "Data synced! Using existing bundles" | "View Bundles" button |

---

## **Expected Frontend Behavior Summary**

### **✅ DO:**
- Poll `/api/shopify/status/{uploadId}` every 5 seconds
- Check for `bundle_count > 0` to know when bundles are ready
- Handle `status: "failed"` gracefully with user-friendly errors
- Show different messages for CSV processing vs bundle generation
- Allow 5-10 minutes timeout for bundle generation
- Handle case where bundles already exist (immediate `bundle_count`)

### **❌ DON'T:**
- Stop polling after only 30 seconds
- Show raw backend errors to users
- Assume bundles are ready just because CSV status is "completed"
- Poll faster than every 3 seconds (unnecessary load)
- Fail if quick mode times out (it falls back automatically)

---

## **Backend Guarantees**

The backend **ALWAYS**:
1. Returns `uploadId` immediately from POST `/upload`
2. Updates `status` to "completed" when CSV is done
3. Updates `bundle_count` when bundles exist in DB
4. Returns lowercase status values (`"completed"`, not `"COMPLETED"`)
5. Falls back to normal generation if quick mode fails
6. Returns early if bundles already exist (scenario 2)

---

## **Testing Checklist**

- [ ] Test first-time install with valid data → bundles appear
- [ ] Test re-sync with existing bundles → no re-generation
- [ ] Test invalid CSV → proper error message
- [ ] Test insufficient data → graceful failure message
- [ ] Test quick mode timeout → eventually completes
- [ ] Test subsequent sync (not first install) → CSV completes, no bundle generation
- [ ] Test CORS from your Remix domain
- [ ] Test polling timeout handling
- [ ] Test network errors during polling

---

## **Quick Reference: When Does Bundle Generation Run?**

| Scenario | Quick Generation | Full Generation |
|----------|-----------------|-----------------|
| **First install + no bundles** | ✅ Yes (auto) | ❌ No (we disabled it) |
| **First install + existing bundles** | ❌ No (returns early) | ❌ No |
| **Subsequent sync** | ❌ No (not first install) | ❌ No (must trigger manually) |
| **Quick timeout** | ⏱️ Times out → | ✅ Fallback |
| **Quick error** | ❌ Fails → | ✅ Fallback |
| **Manual trigger** | ❌ No | ✅ Yes (if user requests) |

---

## **Next Steps**

1. **Backend**: Deploy the logging improvements from earlier
2. **Frontend**: Implement the polling state machine above
3. **Testing**: Test all 7 scenarios with real data
4. **Monitoring**: Watch Cloud Run logs to verify flow
5. **UX**: Add appropriate loading states and error messages
