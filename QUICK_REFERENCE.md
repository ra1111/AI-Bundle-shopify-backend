# Quick Reference: Frontend Integration

**TL;DR:** How to integrate with the bundle generation API.

---

## **The Happy Path (90% of cases)**

```typescript
// 1. Upload CSV
const { uploadId } = await POST("/api/shopify/upload", {
  shopId: "shop.myshopify.com",
  csvType: "orders",
  csvData: csvContent,
  triggerPipeline: true
})

// 2. Poll every 5 seconds
while (true) {
  const status = await GET(`/api/shopify/status/${uploadId}`)

  if (status.status === "failed") {
    showError(status.error_message)
    break
  }

  if (status.status === "completed" && status.bundle_count > 0) {
    showSuccess(`${status.bundle_count} bundles ready!`)
    break
  }

  await sleep(5000)
}

// 3. Get bundles
const { recommendations } = await GET(`/api/shopify/recommendations?shopId=...`)
displayBundles(recommendations)
```

---

## **Key Endpoints**

| Endpoint | Method | Purpose | Returns |
|----------|--------|---------|---------|
| `/api/shopify/upload` | POST | Upload CSV | `uploadId` (immediate) |
| `/api/shopify/status/{id}` | GET | Check progress | Status + bundle count |
| `/api/shopify/recommendations` | GET | Get bundles | Bundle list |

---

## **Status Values You'll See**

### **During CSV Upload (5-15 seconds)**
```json
{
  "status": "processing",
  "processed_rows": 45,
  "total_rows": 100,
  "bundle_count": null
}
```
**Show:** "Processing CSV: 45/100 rows"

### **CSV Done, Bundles Generating (30-120 seconds)**
```json
{
  "status": "completed",
  "processed_rows": 100,
  "total_rows": 100,
  "bundle_count": null  // ← Still null!
}
```
**Show:** "Generating bundles..."

### **Bundles Ready (SUCCESS!)**
```json
{
  "status": "completed",
  "bundle_count": 10  // ← Bundles ready!
}
```
**Show:** "Success! 10 bundles created"

### **Failed**
```json
{
  "status": "failed",
  "error_message": "Missing required: order_id"
}
```
**Show:** "Upload failed: Missing 'order_id' column"

---

## **Important: Don't Confuse These Two**

| Scenario | `status` | `bundle_count` | What it means |
|----------|----------|----------------|---------------|
| **Bundles generating** | `"completed"` | `null` | CSV done, wait for bundles |
| **Bundles ready** | `"completed"` | `10` | Everything done! |

**Key:** You must check **BOTH** `status === "completed"` **AND** `bundle_count > 0`

---

## **All Scenarios in 1 Minute**

| User Does | What Happens | Time | `bundle_count` |
|-----------|--------------|------|----------------|
| **First sync** | Quick bundles generate | 30-120s | Appears after delay |
| **Re-sync** | Uses existing bundles | 5-15s | Immediate |
| **Bad CSV** | Fails fast | 2-5s | `null` |
| **Not enough data** | CSV succeeds, bundles fail | 15-45s | `null` + error |

---

## **Polling Logic (Copy-Paste Ready)**

```typescript
async function waitForBundles(uploadId: string): Promise<number> {
  const MAX_POLLS = 120  // 10 minutes
  const POLL_INTERVAL = 5000  // 5 seconds

  for (let i = 0; i < MAX_POLLS; i++) {
    const status = await fetch(`/api/shopify/status/${uploadId}`).then(r => r.json())

    // Failed
    if (status.status === "failed") {
      throw new Error(status.error_message || "Upload failed")
    }

    // Success!
    if (status.status === "completed" && status.bundle_count > 0) {
      return status.bundle_count
    }

    // Still working
    if (status.status === "processing") {
      updateUI(`Processing CSV: ${status.processed_rows}/${status.total_rows}`)
    } else if (status.status === "completed" && !status.bundle_count) {
      updateUI("Generating bundles...")
    }

    await new Promise(r => setTimeout(r, POLL_INTERVAL))
  }

  throw new Error("Timeout after 10 minutes")
}
```

---

## **Common Mistakes to Avoid**

### ❌ **WRONG: Stop polling when CSV is done**
```typescript
// BAD!
if (status.status === "completed") {
  return  // ← Bundles might still be generating!
}
```

### ✅ **CORRECT: Check bundle_count**
```typescript
// GOOD!
if (status.status === "completed" && status.bundle_count > 0) {
  return  // ← Actually done!
}
```

---

### ❌ **WRONG: Poll too fast**
```typescript
// BAD! Wastes API calls
await sleep(1000)  // Every 1 second
```

### ✅ **CORRECT: Poll every 5 seconds**
```typescript
// GOOD!
await sleep(5000)  // Every 5 seconds
```

---

### ❌ **WRONG: Timeout too early**
```typescript
// BAD! Quick mode can take 2 minutes
const MAX_POLLS = 20  // Only 100 seconds
```

### ✅ **CORRECT: Allow 10 minutes**
```typescript
// GOOD!
const MAX_POLLS = 120  // 10 minutes
```

---

### ❌ **WRONG: Show raw errors to users**
```typescript
// BAD!
showError("orders.csv schema error → Missing required: order_id")
```

### ✅ **CORRECT: User-friendly messages**
```typescript
// GOOD!
showError("Your CSV is missing the 'order_id' column. Please check your Shopify export.")
```

---

## **Error Message Translations**

| Backend Error | Show to User |
|--------------|--------------|
| `"Missing required: order_id"` | "Your orders CSV is missing the 'order_id' column" |
| `"Could not detect CSV type"` | "Please ensure you're exporting the correct format from Shopify" |
| `"Insufficient order history"` | "We need more order history. Keep selling and sync again in a few weeks!" |
| `"Variants missing for SKUs"` | "Some products don't have variant data. Please sync products first." |

---

## **UI States Checklist**

- [ ] **Uploading** - Show spinner "Uploading data..."
- [ ] **CSV Processing** - Show progress "Processing: 45/100 rows"
- [ ] **Bundles Generating** - Show spinner "Generating bundles..."
- [ ] **Success** - Show "10 bundles created!" + View button
- [ ] **Failed (CSV)** - Show error + "Try Again" button
- [ ] **Failed (Bundles)** - Show "Need more data" + Learn More link
- [ ] **Timeout** - Show "Taking too long, please check back later"

---

## **Testing Your Integration**

### **Test 1: Happy Path**
1. Upload valid orders CSV
2. Should see "Processing CSV..." for 5-15s
3. Then "Generating bundles..." for 30-120s
4. Then "10 bundles created!"

### **Test 2: Re-sync**
1. Upload CSV again (same shop)
2. Should see "Processing CSV..." for 5-15s
3. Then immediately "Using existing bundles!"
4. **No** long bundle generation wait

### **Test 3: Invalid CSV**
1. Upload CSV missing required column
2. Should see error within 2-5s
3. Error message should be user-friendly

### **Test 4: Insufficient Data**
1. Upload CSV with only 2 orders
2. CSV should process successfully
3. Bundle generation should fail with helpful message

---

## **Debugging Checklist**

If syncing fails, check:

- [ ] **CORS**: Is your Remix domain in `CORS_ORIGINS` env var?
- [ ] **Endpoint**: Using `/api/shopify/status/{id}` not `/api/upload-status/`?
- [ ] **Status check**: Checking `status === "completed"` AND `bundle_count > 0`?
- [ ] **Timeout**: Allowing at least 5 minutes?
- [ ] **Case**: Checking lowercase `"completed"` not `"COMPLETED"`?
- [ ] **Network**: Check browser console for errors
- [ ] **Backend logs**: Check Cloud Run logs - is status endpoint being called?

---

## **CORS Configuration**

Make sure your Cloud Run service has this env var:

```bash
CORS_ORIGINS=https://admin.shopify.com,https://your-remix-app.com,http://localhost:3000
```

Without this, browser will block requests.

---

## **Complete Example (React/Remix)**

```typescript
import { useState } from "react"

export default function BundleSync() {
  const [state, setState] = useState<
    | { status: "idle" }
    | { status: "uploading" }
    | { status: "processing", progress: number }
    | { status: "generating" }
    | { status: "success", count: number }
    | { status: "error", message: string }
  >({ status: "idle" })

  async function syncData(csvContent: string) {
    setState({ status: "uploading" })

    try {
      // Upload
      const uploadRes = await fetch("/api/shopify/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shopId: "shop.myshopify.com",
          csvType: "orders",
          csvData: csvContent,
          triggerPipeline: true
        })
      })
      const { uploadId } = await uploadRes.json()

      // Poll
      for (let i = 0; i < 120; i++) {
        const statusRes = await fetch(`/api/shopify/status/${uploadId}`)
        const status = await statusRes.json()

        if (status.status === "failed") {
          setState({ status: "error", message: status.error_message })
          return
        }

        if (status.status === "processing") {
          setState({
            status: "processing",
            progress: (status.processed_rows / status.total_rows) * 100
          })
        } else if (status.status === "completed" && !status.bundle_count) {
          setState({ status: "generating" })
        } else if (status.status === "completed" && status.bundle_count > 0) {
          setState({ status: "success", count: status.bundle_count })
          return
        }

        await new Promise(r => setTimeout(r, 5000))
      }

      setState({ status: "error", message: "Timeout after 10 minutes" })

    } catch (error) {
      setState({ status: "error", message: error.message })
    }
  }

  return (
    <div>
      {state.status === "idle" && (
        <button onClick={() => syncData(csvData)}>Sync Data</button>
      )}

      {state.status === "uploading" && (
        <div>Uploading...</div>
      )}

      {state.status === "processing" && (
        <div>Processing CSV: {state.progress.toFixed(0)}%</div>
      )}

      {state.status === "generating" && (
        <div>Generating bundles...</div>
      )}

      {state.status === "success" && (
        <div>Success! {state.count} bundles created</div>
      )}

      {state.status === "error" && (
        <div>Error: {state.message}</div>
      )}
    </div>
  )
}
```

---

## **Next Steps**

1. ✅ Read [FRONTEND_BACKEND_CONTRACT.md](FRONTEND_BACKEND_CONTRACT.md) for all scenarios
2. ✅ Read [API_RESPONSE_SCHEMAS.md](API_RESPONSE_SCHEMAS.md) for exact response formats
3. ✅ Read [FLOW_DIAGRAMS.md](FLOW_DIAGRAMS.md) for visual flows
4. ✅ Copy the polling logic above
5. ✅ Test with real data
6. ✅ Check Cloud Run logs if issues

---

## **Need Help?**

**Frontend says "sync failed" but CSV uploaded:**
- Check if you're polling the status endpoint
- Check if you're waiting for `bundle_count > 0`
- Check Cloud Run logs to see if bundles were actually created

**CORS errors in browser:**
- Add your domain to `CORS_ORIGINS` env var in Cloud Run

**Timeouts:**
- Increase `MAX_POLLS` to 120 (10 minutes)
- Check if backend is actually processing (Cloud Run logs)

**bundle_count stays null:**
- This is normal for 30-120s while bundles generate
- Only worry if it stays null for 10+ minutes
- Check `/generation-progress/{uploadId}` for details
