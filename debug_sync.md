# Debugging Frontend Sync Issue

## Step 1: Test the Status Endpoint Directly

After uploading a CSV, test the status endpoint:

```bash
# Replace {upload_id} with actual ID from upload response
curl https://your-cloud-run-url.run.app/api/shopify/status/{upload_id}
```

Expected response:
```json
{
  "upload_id": "abc-123",
  "shop_id": "shop.myshopify.com",
  "status": "completed",  // or "processing" or "failed"
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": 10
}
```

---

## Step 2: Check Database Status Directly

Query CockroachDB to see actual status:

```sql
SELECT id, shop_id, status, total_rows, processed_rows, created_at, updated_at
FROM csv_uploads
ORDER BY created_at DESC
LIMIT 5;
```

---

## Step 3: Check Browser Console

In your Remix app:
1. Open DevTools → Network tab
2. Upload a CSV
3. Watch for polling requests to `/api/shopify/status/...`
4. Check:
   - Is it calling the correct URL?
   - What status codes are returned? (200, 404, 500?)
   - What's in the response body?
   - Any CORS errors in Console tab?

---

## Common Frontend Issues

### Issue 1: Case Sensitivity
**Backend returns:** `"completed"`
**Frontend checks:** `status === "COMPLETED"`

**Fix in Remix:**
```typescript
const isDone = status.toLowerCase() === "completed"
```

### Issue 2: Wrong Polling URL
**Wrong:** `/api/bundles/status/${uploadId}`
**Correct:** `/api/shopify/status/${uploadId}`

### Issue 3: Polling Timeout
```typescript
// BAD: Times out too quickly
const MAX_POLLS = 10 // Only 50 seconds!

// GOOD: Enough time for processing
const MAX_POLLS = 60 // 5 minutes
const POLL_INTERVAL = 5000 // 5 seconds
```

### Issue 4: Not Checking triggerPipeline Flag
If `triggerPipeline: false`, the upload completes after CSV ingestion.
If `triggerPipeline: true`, it waits for bundle generation too.

---

## Step 4: Add Logging to Backend

Temporarily add logging to see what frontend is requesting:

```python
# In routers/shopify_upload.py line 162
@router.get("/status/{upload_id}", response_model=UploadStatusResponse)
async def get_upload_status(upload_id: str, db: AsyncSession = Depends(get_db)):
    logger.info(f"🔍 STATUS CHECK: upload_id={upload_id}")

    upload = await db.get(CsvUpload, upload_id)
    if not upload:
        logger.warning(f"❌ STATUS CHECK: upload {upload_id} NOT FOUND")
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

    logger.info(f"✅ STATUS CHECK: upload_id={upload_id} status={upload.status}")
    # ... rest of function
```

Then check Cloud Run logs to see if frontend is even calling the endpoint.

---

## Step 5: Frontend Polling Template

Here's how your Remix action/loader should poll:

```typescript
// app/routes/bundles.upload.tsx

export const action = async ({ request }: ActionFunctionArgs) => {
  const formData = await request.formData()
  const csvData = formData.get("csvData")

  // 1. Upload CSV
  const uploadResponse = await fetch("https://your-backend.run.app/api/shopify/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      shopId: "shop.myshopify.com",
      csvType: "orders",
      csvData: csvData,
      triggerPipeline: true
    })
  })

  const { uploadId } = await uploadResponse.json()

  // 2. Poll for completion
  return pollUploadStatus(uploadId)
}

async function pollUploadStatus(uploadId: string) {
  const MAX_ATTEMPTS = 60  // 5 minutes total
  const POLL_INTERVAL = 5000  // 5 seconds

  for (let i = 0; i < MAX_ATTEMPTS; i++) {
    const statusResponse = await fetch(
      `https://your-backend.run.app/api/shopify/status/${uploadId}`
    )

    if (!statusResponse.ok) {
      console.error(`Status check failed: ${statusResponse.status}`)
      await new Promise(r => setTimeout(r, POLL_INTERVAL))
      continue
    }

    const status = await statusResponse.json()
    console.log(`Poll ${i+1}: status =`, status.status)

    if (status.status === "completed") {
      return { success: true, bundleCount: status.bundle_count }
    }

    if (status.status === "failed") {
      return { success: false, error: status.error_message }
    }

    // Still processing, wait and try again
    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL))
  }

  return { success: false, error: "Timeout waiting for CSV processing" }
}
```

---

## Quick Fix Checklist

- [ ] Frontend polls `/api/shopify/status/{uploadId}` (not `/api/upload-status/`)
- [ ] Frontend checks for `status === "completed"` (lowercase)
- [ ] Polling has sufficient timeout (5+ minutes for large CSVs)
- [ ] CORS is configured to allow your Remix app origin
- [ ] Database shows upload status changing to "completed"
- [ ] Cloud Run logs show status endpoint being called
- [ ] No errors in browser console

---

## Next Steps

1. **Test the backend directly** using curl/Postman
2. **Check Cloud Run logs** to see if frontend is calling status endpoint
3. **Check browser console** for errors
4. **Share logs** if still not working:
   - Backend logs from Cloud Run
   - Frontend console errors
   - Network tab showing polling requests
