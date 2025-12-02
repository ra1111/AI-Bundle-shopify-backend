# Visual Flow Diagrams: All Scenarios

---

## **Scenario 1: First-Time Install → Quick Mode Success** ✅

```
FRONTEND                          BACKEND                           DATABASE
   |                                 |                                  |
   | POST /api/shopify/upload        |                                  |
   |-------------------------------->|                                  |
   |                                 | Create CsvUpload                 |
   |                                 | status="processing"              |
   |                                 |--------------------------------->|
   | {uploadId, status:"processing"} |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Start polling every 5s]        |                                  |
   |                                 |                                  |
   | GET /status/{uploadId}          | [Background: Ingest CSV]         |
   |-------------------------------->|                                  |
   | {status:"processing", 45/100}   |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | GET /status/{uploadId}          | Update status="completed"        |
   |-------------------------------->|--------------------------------->|
   | {status:"completed",            |                                  |
   |  bundle_count:null}             |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Show "Generating bundles..."]  |                                  |
   |                                 |                                  |
   |                                 | [Check: first install?]          |
   |                                 | [Check: existing bundles?]       |
   |                                 | → No existing bundles            |
   |                                 |                                  |
   |                                 | [Run quick generation]           |
   |                                 | - Generate FBT bundles           |
   |                                 | - Generate BOGO bundles          |
   |                                 | - Generate volume bundles        |
   |                                 |                                  |
   |                                 | Save bundles to DB               |
   |                                 |--------------------------------->|
   |                                 |                                  |
   |                                 | Mark upload completed            |
   |                                 |--------------------------------->|
   |                                 |                                  |
   | GET /status/{uploadId}          |                                  |
   |-------------------------------->|                                  |
   | {status:"completed",            |                                  |
   |  bundle_count:10} ✅            |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Stop polling]                  |                                  |
   | [Show success!]                 |                                  |
   |                                 |                                  |
   | GET /recommendations?shopId=... |                                  |
   |-------------------------------->| Fetch bundles                    |
   |                                 |--------------------------------->|
   | {bundles: [...]}                |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Display bundles to user]       |                                  |
   |                                 |                                  |
```

**Timeline:** ~30-120 seconds total

---

## **Scenario 2: Re-sync with Existing Bundles** ⚡

```
FRONTEND                          BACKEND                           DATABASE
   |                                 |                                  |
   | POST /api/shopify/upload        |                                  |
   |-------------------------------->|                                  |
   |                                 | Create CsvUpload                 |
   |                                 |--------------------------------->|
   | {uploadId, status:"processing"} |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | GET /status/{uploadId}          | [Background: Ingest CSV]         |
   |-------------------------------->|                                  |
   | {status:"processing"}           |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   |                                 | Update status="completed"        |
   |                                 |--------------------------------->|
   |                                 |                                  |
   |                                 | [Check: existing bundles?]       |
   |                                 | → Found 10 existing bundles!     |
   |                                 |                                  |
   |                                 | Mark upload completed            |
   |                                 | Return early ⚡                  |
   |                                 |                                  |
   | GET /status/{uploadId}          |                                  |
   |-------------------------------->|                                  |
   | {status:"completed",            |                                  |
   |  bundle_count:10} ✅            |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Stop polling - bundles exist!] |                                  |
   | [Show "Data synced!"]           |                                  |
   |                                 |                                  |
```

**Timeline:** ~5-15 seconds total (much faster!)

**Key Difference:** `bundle_count` appears immediately, no generation delay.

---

## **Scenario 3: Quick Mode Timeout → Fallback** ⏱️

```
FRONTEND                          BACKEND                           DATABASE
   |                                 |                                  |
   | POST /api/shopify/upload        |                                  |
   |-------------------------------->|                                  |
   | {uploadId}                      |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | GET /status/{uploadId}          | [CSV ingestion done]             |
   |-------------------------------->|                                  |
   | {status:"completed",            |                                  |
   |  bundle_count:null}             |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [30s] Polling...                | [Quick generation running]       |
   | GET /status/{uploadId}          | - FBT tier running...            |
   |-------------------------------->| - Taking too long...             |
   | {status:"completed",            |                                  |
   |  bundle_count:null}             |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [60s] Still polling...          | [Quick generation running]       |
   | GET /status/{uploadId}          | - Still working...               |
   |-------------------------------->|                                  |
   | {status:"completed",            |                                  |
   |  bundle_count:null}             |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [120s] Still polling...         | ⏱️ TIMEOUT!                      |
   |                                 | Log: "Quick-start exceeded 120s" |
   |                                 | → Fall back to full generation   |
   |                                 |                                  |
   | [Show: "Taking longer than      |                                  |
   |  expected, running              | [Full generation pipeline]       |
   |  comprehensive analysis..."]    | - Association rules              |
   |                                 | - Co-occurrence analysis         |
   |                                 | - Popularity tiers               |
   |                                 |                                  |
   | [5-10 min] Polling...           | Save bundles                     |
   | GET /status/{uploadId}          |--------------------------------->|
   |-------------------------------->|                                  |
   | {status:"completed",            |                                  |
   |  bundle_count:25} ✅            |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Success - more bundles!]       |                                  |
   |                                 |                                  |
```

**Timeline:** 5-20 minutes (fallback to comprehensive generation)

---

## **Scenario 4: CSV Ingestion Fails** ❌

```
FRONTEND                          BACKEND                           DATABASE
   |                                 |                                  |
   | POST /api/shopify/upload        |                                  |
   | (CSV with missing columns)      |                                  |
   |-------------------------------->|                                  |
   | {uploadId}                      | Create CsvUpload                 |
   |<--------------------------------|--------------------------------->|
   |                                 |                                  |
   | GET /status/{uploadId}          | [Background: Parse CSV]          |
   |-------------------------------->| → Missing "order_id" column!     |
   | {status:"processing"}           |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   |                                 | Update status="failed"           |
   |                                 | error_message="..."              |
   |                                 |--------------------------------->|
   |                                 |                                  |
   | GET /status/{uploadId}          |                                  |
   |-------------------------------->|                                  |
   | {status:"failed",               |                                  |
   |  error_message:                 |                                  |
   |  "Missing required: order_id"}  |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Stop polling]                  |                                  |
   | [Show error message]            |                                  |
   | [Show "Try Again" button]       |                                  |
   |                                 |                                  |
```

**Timeline:** ~2-5 seconds (fails fast)

---

## **Scenario 5: Bundle Generation Fails** ❌

```
FRONTEND                          BACKEND                           DATABASE
   |                                 |                                  |
   | POST /api/shopify/upload        |                                  |
   |-------------------------------->|                                  |
   | {uploadId}                      |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | GET /status/{uploadId}          | [CSV ingestion: SUCCESS ✅]      |
   |-------------------------------->| Update status="completed"        |
   | {status:"completed",            |--------------------------------->|
   |  bundle_count:null}             |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   |                                 | [Quick generation starts]        |
   |                                 | → Check order count              |
   |                                 | → Only 3 orders found!           |
   |                                 | → Need minimum 10 orders         |
   |                                 |                                  |
   |                                 | Update generation progress       |
   |                                 | status="failed"                  |
   |                                 | message="Insufficient data"      |
   |                                 |--------------------------------->|
   |                                 |                                  |
   | GET /generation-progress/{id}   |                                  |
   |-------------------------------->|                                  |
   | {status:"failed",               |                                  |
   |  message:"Insufficient order    |                                  |
   |  history: need 10+ orders"}     |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Stop polling]                  |                                  |
   | [Show: "Data uploaded           |                                  |
   |  successfully, but need more    |                                  |
   |  order history to create        |                                  |
   |  bundles"]                      |                                  |
   |                                 |                                  |
```

**Timeline:** ~10-30 seconds (CSV succeeds, generation fails)

---

## **Scenario 6: Not First-Time Install (Subsequent Sync)** 🔄

```
FRONTEND                          BACKEND                           DATABASE
   |                                 |                                  |
   | POST /api/shopify/upload        |                                  |
   | {triggerPipeline: false}        |                                  |
   |-------------------------------->|                                  |
   | {uploadId}                      |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | GET /status/{uploadId}          | [CSV ingestion]                  |
   |-------------------------------->| Update status="completed"        |
   | {status:"completed",            |--------------------------------->|
   |  bundle_count:10}               |                                  |
   |<--------------------------------|                                  |
   |                                 |                                  |
   | [Fast! Shows existing bundles]  | [No generation triggered]        |
   |                                 | → Not first install              |
   |                                 | → triggerPipeline = false        |
   |                                 |                                  |
   | [Show: "Data synced!            |                                  |
   |  Using existing bundles"]       |                                  |
   |                                 |                                  |
```

**Timeline:** ~5-15 seconds (CSV only)

---

## **Decision Tree: Backend Logic**

```
CSV Upload Received
        |
        v
    Ingest CSV
        |
        v
    CSV Valid?
        |
   +----+----+
   |         |
  NO        YES
   |         |
   v         v
FAIL    Detect Install Type
 END         |
             v
        First Install?
             |
        +----+----+
        |         |
       YES        NO
        |         |
        v         v
   Quick Mode  Normal Mode
   Enabled?    (No bundles)
        |         |
   +----+----+   |
   |         |   |
  YES        NO  |
   |         |   |
   v         v   v
 Bundles   Skip  Complete
 Exist?    Gen   (existing
   |       |     bundles)
+--+--+    |
|     |    |
YES   NO   |
|     |    |
v     v    v
Skip  Run  END
Gen   Quick
|     Gen
|      |
|      v
|   Success?
|      |
|  +---+---+
|  |       |
|  YES     NO
|  |       |
|  v       v
| Save   Timeout/Error
| Quick    |
| Bundles  v
|  |    Fallback to
|  |    Full Gen
|  |       |
|  v       v
| Mark    Save
| Done    Bundles
|  |       |
|  v       v
+->END    END
```

---

## **Polling Strategy Flowchart**

```
                    Start Upload
                         |
                         v
                 Get uploadId
                         |
                         v
                    Start Timer
                         |
                         v
            +------------+------------+
            |                         |
            v                         |
      Poll /status                    |
            |                         |
            v                         |
      Check status                    |
            |                         |
     +------+------+                  |
     |      |      |                  |
   PROC   COMP   FAIL                 |
     |      |      |                  |
     v      |      v                  |
  Show      |   Show Error            |
  Progress  |   → END                 |
     |      |                         |
     +------+                         |
            |                         |
            v                         |
      bundle_count > 0?               |
            |                         |
     +------+------+                  |
     |             |                  |
    YES            NO                 |
     |             |                  |
     v             v                  |
  SUCCESS!    Still Generating        |
  → END            |                  |
                   v                  |
              Elapsed > 10min?        |
                   |                  |
            +------+------+           |
            |             |           |
           YES            NO          |
            |             |           |
            v             v           |
         TIMEOUT      Wait 5s         |
         → END            |           |
                          |           |
                          +<----------+
```

---

## **Summary Table: What Frontend Should Expect**

| Scenario | CSV Time | Bundle Time | Total Time | bundle_count Timing |
|----------|----------|-------------|------------|-------------------|
| **First install + quick success** | 5-15s | 30-120s | 35-135s | After delay |
| **Re-sync (bundles exist)** | 5-15s | 0s | 5-15s | Immediate |
| **Quick timeout → fallback** | 5-15s | 5-20min | 5-20min | After long delay |
| **CSV fails** | 2-5s | N/A | 2-5s | `null` |
| **Bundles fail** | 5-15s | 10-30s | 15-45s | `null` + error |
| **Subsequent sync (no trigger)** | 5-15s | 0s | 5-15s | Immediate (existing) |

---

## **Frontend State Transitions**

```
          INITIAL
             |
             v
     [User clicks "Sync Data"]
             |
             v
         UPLOADING ───────────> (Upload fails)
             |                        |
             v                        v
      CSV_PROCESSING              ERROR_STATE
             |
             +──────────> (CSV fails)
             |                 |
             v                 v
      CSV_COMPLETED        ERROR_STATE
             |
             +──────────> (Bundles exist immediately)
             |                 |
             v                 v
    BUNDLES_GENERATING    BUNDLES_READY
             |
             +──────────> (Generation fails)
             |                 |
             v                 v
      BUNDLES_READY       ERROR_STATE
```

**Valid Transitions:**
- `INITIAL → UPLOADING`
- `UPLOADING → CSV_PROCESSING | ERROR_STATE`
- `CSV_PROCESSING → CSV_COMPLETED | ERROR_STATE`
- `CSV_COMPLETED → BUNDLES_GENERATING | BUNDLES_READY`
- `BUNDLES_GENERATING → BUNDLES_READY | ERROR_STATE`
- `BUNDLES_READY → [terminal]`
- `ERROR_STATE → [terminal]`

---

## **Recommended Polling Intervals**

| Phase | Interval | Max Duration | Total Polls |
|-------|----------|--------------|-------------|
| **CSV Processing** | 3s | 30s | ~10 polls |
| **Bundle Generation (Quick)** | 5s | 2min | ~24 polls |
| **Bundle Generation (Full)** | 10s | 20min | ~120 polls |

**Adaptive Strategy:**
```typescript
function getPollInterval(elapsedSeconds: number): number {
  if (elapsedSeconds < 30) return 3000   // Fast polling during CSV
  if (elapsedSeconds < 120) return 5000  // Medium during quick gen
  return 10000                            // Slow during full gen
}
```

This prevents unnecessary API calls while still providing responsive updates.
