# Bundle Generation Workflow - Analysis & Recommendations

## Current Flow Analysis

### What's Happening Now

**Frontend → Backend Flow:**
1. User clicks "Sync Data" in frontend
2. Frontend uploads CSVs via `POST /api/upload-csv`
3. Backend returns immediately with `{uploadId, runId, status: "processing"}`
4. CSV processing happens in **background** (FastAPI BackgroundTasks)
5. Frontend polls `GET /api/upload-status/{uploadId}` to check progress
6. When CSV status = "completed", frontend shows "Sync Complete - Review Recommendations"

**❌ THE GAP:**
- After CSV upload completes, user clicks "Review Recommendations"
- Frontend calls `POST /api/generate-bundles` with `{csvUploadId}`
- Backend **synchronously runs** the entire 9-phase bundle generation pipeline (can take 90+ seconds)
- Frontend **has no visibility** into what's happening during this time
- User sees loading spinner with no progress indication

---

## Bundle Generation Pipeline (9 Phases)

### Discovered from Code Analysis

Located in: `/services/bundle_generator.py` - `generate_bundle_recommendations()`

**Current Timeouts & Limits:**
- Max time budget: `300 seconds` (5 minutes)
- Max total attempts: `500`
- Max consecutive failures: `10` (circuit breaker)

**The 9 Phases:**

1. **Data Mapping & Enrichment** (~2-5s)
   - Enriches order lines with variant data
   - File: `services/data_mapper.py`

2. **Objective Scoring** (~1-3s)
   - Computes 8 business objectives per product
   - File: `services/objectives.py`
   - Objectives: increase_aov, clear_slow_movers, seasonal_promo, new_launch, category_bundle, gift_box, subscription_push, margin_guard

3. **ML Candidate Generation** (~30-90s) **← LONGEST PHASE**
   - Multi-source ML generation:
     - Apriori association rules
     - FPGrowth algorithm
     - Item2Vec embeddings
     - Top pair mining
     - Fallback ladder for small shops
   - File: `services/ml/candidate_generator.py`
   - Generates candidates for each objective × bundle_type combination

4. **Deduplication** (~2-5s)
   - Removes duplicate SKU combinations
   - File: `services/deduplication.py`

5a. **Global Enterprise Optimization** (~10-20s)
   - Pareto optimization across all bundles
   - File: `services/ml/optimization_engine.py`
   - Only runs if >10 candidates

5b. **Weighted Ranking** (fallback if <10 candidates)
   - Linear weighted ranking
   - File: `services/ranker.py`

5c. **Forced Pair Fallback**
   - Ensures minimum bundle coverage
   - Injects top association rules if needed

6. **Bayesian Pricing Optimization** (~5-10s per bundle)
   - Optimizes bundle pricing and discount
   - File: `services/pricing.py`

7. **Explainability** (~1-2s)
   - Generates human-readable explanations
   - File: `services/explainability.py`

8. **AI Copy Generation** (~3-5s per bundle) **← OpenAI API calls**
   - Uses OpenAI GPT-3.5-Turbo (NOT Gemini!)
   - Generates titles, descriptions, taglines
   - File: `services/ai_copy_generator.py`

9. **Finalization & Storage** (~2-5s)
   - Persists to database
   - File: `services/storage.py`

**Total Time: 60-150 seconds typically**

---

## The Communication Gap Problem

### Current State:
```
Frontend                          Backend
  |                                 |
  | POST /api/upload-csv            |
  |-------------------------------->|
  |<--------------------------------| {uploadId, status:"processing"}
  |                                 |
  | (polls /api/upload-status)      |
  |-------------------------------->|
  |<--------------------------------| {status:"completed"}  ← CSV done
  |                                 |
  | User sees "Review Recommendations" button
  | User clicks button              |
  |                                 |
  | POST /api/generate-bundles      |
  |-------------------------------->|
  |                                 | ← Phase 1: Data Mapping
  |                                 | ← Phase 2: Objective Scoring
  |                                 | ← Phase 3: ML Candidates (60s!)
  | (WAITING... no feedback)        | ← Phase 4: Dedup
  | (User sees spinner)             | ← Phase 5: Optimization
  |                                 | ← Phase 6: Pricing
  |                                 | ← Phase 7: Explain
  |                                 | ← Phase 8: AI Copy (OpenAI calls)
  |                                 | ← Phase 9: Storage
  |                                 |
  |<--------------------------------| {success:true, result:{...}}
  |                                 |
  | Shows recommendations           |
```

### Problems:
1. ❌ **No progress visibility** - User doesn't know what's happening
2. ❌ **No phase indication** - Can't tell if it's stuck or working
3. ❌ **No time estimate** - User has no idea how long to wait
4. ❌ **Poor UX** - Just a spinning loader for 60-150 seconds
5. ❌ **No intermediate feedback** - Can't see partial results
6. ❌ **Timeout risk** - Cloud Run default timeout might kill long-running requests

---

## Recommended Solutions

### Solution 1: Server-Sent Events (SSE) - **RECOMMENDED**

**Pros:**
- ✅ Simple to implement (built into browsers)
- ✅ Unidirectional (server → client) - perfect for progress updates
- ✅ Works over HTTP (no protocol upgrade needed)
- ✅ Compatible with Cloud Run
- ✅ Automatic reconnection
- ✅ Lower overhead than WebSockets

**Cons:**
- ⚠️ Only server→client (but that's all we need)
- ⚠️ Some proxies may buffer (but Cloud Run handles it well)

**Implementation:**
```python
# routers/bundles.py
from fastapi.responses import StreamingResponse
import asyncio
import json

@router.post("/generate-bundles/stream")
async def generate_bundles_stream(request: dict):
    csv_upload_id = request.get("csvUploadId")

    async def event_generator():
        try:
            yield f"data: {json.dumps({'phase': 'starting', 'progress': 0})}\n\n"

            # Phase 1
            yield f"data: {json.dumps({'phase': 'data_mapping', 'progress': 10, 'message': 'Enriching order data...'})}\n\n"
            # ... run phase 1 ...

            # Phase 2
            yield f"data: {json.dumps({'phase': 'objective_scoring', 'progress': 20, 'message': 'Scoring business objectives...'})}\n\n"
            # ... run phase 2 ...

            # Phase 3 (longest - send sub-progress)
            yield f"data: {json.dumps({'phase': 'ml_candidates', 'progress': 30, 'message': 'Generating bundle candidates...', 'candidates_found': 0})}\n\n"
            # ... as candidates are found, send updates ...
            yield f"data: {json.dumps({'phase': 'ml_candidates', 'progress': 50, 'message': 'Generating bundle candidates...', 'candidates_found': 25})}\n\n"

            # Continue for all phases...

            yield f"data: {json.dumps({'phase': 'complete', 'progress': 100, 'bundles_created': 42})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'phase': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

```javascript
// Frontend
const eventSource = new EventSource(`/api/generate-bundles/stream?csvUploadId=${uploadId}`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  setProgress(data.progress);
  setPhase(data.phase);
  setMessage(data.message);

  if (data.phase === 'complete') {
    eventSource.close();
    // Navigate to recommendations
  }
};

eventSource.onerror = () => {
  eventSource.close();
  // Handle error
};
```

---

### Solution 2: WebSockets - Alternative

**Pros:**
- ✅ Bidirectional communication
- ✅ Real-time updates
- ✅ Can cancel operations

**Cons:**
- ⚠️ More complex to implement
- ⚠️ Requires protocol upgrade
- ⚠️ More overhead
- ⚠️ Connection management complexity

---

### Solution 3: Hybrid (Background Job + Polling with Rich Status) - **SIMPLEST**

Keep current architecture but enhance the status endpoint:

**Pros:**
- ✅ Minimal changes
- ✅ No new infrastructure
- ✅ Works everywhere

**Cons:**
- ⚠️ Polling overhead
- ⚠️ Not real-time
- ⚠️ Delay in updates

**Implementation:**
```python
# Enhanced status model
class BundleGenerationStatus:
    upload_id: str
    status: str  # "processing", "completed", "failed"
    current_phase: str  # "data_mapping", "ml_candidates", etc.
    progress_percentage: int  # 0-100
    phase_message: str  # "Generating candidates for seasonal bundles..."
    bundles_generated: int
    estimated_time_remaining_seconds: int
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
```

---

## Final Recommendation

### Use **SSE (Solution 1)** for production

**Why:**
1. Perfect fit for one-way progress updates
2. Built into browsers (EventSource API)
3. Works reliably with Cloud Run
4. Low latency, real-time updates
5. Simple to implement

### UI/UX Design

**Step-by-step progress indicator:**

```
┌─────────────────────────────────────────────────────┐
│  Generating Bundle Recommendations                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ✓ Data enrichment complete                        │
│  ✓ Objective scoring complete                      │
│  ⟳ Generating ML candidates... (45/100)            │
│    └─ Found 12 high-confidence bundles so far      │
│  ○ Optimizing bundles                              │
│  ○ Generating AI descriptions                      │
│  ○ Finalizing recommendations                      │
│                                                     │
│  [████████████░░░░░░░░] 60%                        │
│                                                     │
│  Estimated time remaining: 30 seconds              │
└─────────────────────────────────────────────────────┘
```

---

## Logging & Monitoring Improvements Needed

### Add to Each Phase:
1. **Start timestamp**
2. **End timestamp**
3. **Duration in ms**
4. **Input metrics** (e.g., "Processing 42 products")
5. **Output metrics** (e.g., "Generated 25 candidates")
6. **Error counts** (if any sub-operations failed)

### Example Enhanced Logging:
```python
phase_start = time.time()
logger.info(f"[{csv_upload_id}] Phase 1: Data Mapping STARTED")

# ... do work ...

duration_ms = int((time.time() - phase_start) * 1000)
logger.info(f"[{csv_upload_id}] Phase 1: Data Mapping COMPLETED - duration={duration_ms}ms enriched_lines={metrics['resolved_variants']}")
```

---

## Next Steps

1. **Immediate (Quick Win):**
   - Add enhanced logging with timing to all 9 phases
   - Add phase progress tracking to bundle generator
   - Return phase information in existing polling endpoint

2. **Short-term (Best UX):**
   - Implement SSE streaming endpoint
   - Update frontend to use EventSource
   - Design step-by-step progress UI

3. **Long-term (Advanced):**
   - Consider moving to true background job queue (Celery/RQ) for multi-instance scaling
   - Add ability to cancel/pause bundle generation
   - Store intermediate results for resumability

---

## Code Changes Required

### Files to Modify:

1. **`services/bundle_generator.py`** - Add progress callbacks and timing
2. **`routers/uploads.py`** - Add SSE endpoint
3. **`database.py`** - Add bundle_generation_progress table/fields
4. **Frontend** - Replace synchronous call with EventSource

### Estimated Development Time:
- Enhanced logging: **2 hours**
- SSE implementation: **4-6 hours**
- Frontend integration: **3-4 hours**
- Testing: **2-3 hours**
- **Total: 1-2 days**

---

## Summary

**The Gap:** Bundle generation happens silently in the background with no user feedback for 60-150 seconds.

**The Solution:** Use Server-Sent Events (SSE) to stream real-time progress updates for each of the 9 phases.

**The Benefit:** Users see exactly what's happening, how long it will take, and don't worry the system is stuck.
