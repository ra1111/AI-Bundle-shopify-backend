# Frontend Response Data Implementation Guide

## Key Source Files

### 1. API Router Files

#### `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py`
- **POST /api/generate-bundles** (lines 51-76): Initiates background bundle generation
  - Returns `{"success": true, "message": "..."}`
- **GET /api/bundle-recommendations** (lines 78-140): Final recommendations
  - Returns array of `BundleRecommendation` objects
  - Optional filter by `uploadId` parameter
- **GET /api/bundle-recommendations/{upload_id}/partial** (lines 143-172): Progressive results
  - Returns partial bundles during generation
  - Includes `"isPartial": true` flag
- **POST /api/generate-bundles/{upload_id}/resume** (lines 175-184): Resume deferred generation
  - Returns acknowledgment message

#### `/home/user/AI-Bundle-shopify-backend/routers/generation_progress.py`
- **GET /api/generation-progress/{upload_id}** (lines 20-68): **PRIMARY POLLING ENDPOINT**
  - Returns current progress state with step, progress %, message
  - Includes optional `staged_state` with wave data
  - Run ID for tracking across requests

### 2. Service Files

#### `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py`
**Core Generation Logic (1600+ lines)**

Key methods:
- `generate_bundle_recommendations()`: Main generation pipeline
- `generate_quick_start_bundles()`: Fast preview for first-time users
- `_emit_heartbeat()`: Progress updates sent to frontend
- `_build_staged_progress_payload()`: Constructs wave data
- `_run_staged_publish()`: Wave-based publishing

Progress updates via `update_generation_progress()` calls:
```python
await update_generation_progress(
    csv_upload_id,
    step="enrichment",      # Current phase
    progress=25,            # 0-100 percentage
    status="in_progress",   # in_progress|completed|failed
    message="...",          # Human-readable message
    metadata={...}          # Phase-specific data
)
```

#### `/home/user/AI-Bundle-shopify-backend/services/progress_tracker.py`
**Progress Persistence (207 lines)**

- `update_generation_progress()`: Upserts progress to database
  - Uses PostgreSQL upsert (on_conflict_do_update)
  - Handles transaction aborts with retry
  - TTL: 24 hours
- `get_generation_checkpoint()`: Fetches checkpoint for resume

### 3. Database Files

#### `/home/user/AI-Bundle-shopify-backend/database.py`

**GenerationProgress Table (lines 227-250):**
```
- upload_id (PK)
- shop_domain
- step (TEXT)
- progress (0-100 INTEGER)
- status (in_progress|completed|failed)
- message (TEXT)
- metadata_json (JSONB) - Phase-specific data
- updated_at (DATETIME with timezone)
```

**BundleRecommendation Table (lines 448-476):**
```
- id (PK)
- csv_upload_id (FK)
- shop_id
- bundle_type
- objective
- products (JSONB)
- pricing (JSONB)
- ai_copy (JSONB) - Generated marketing text
- confidence, predicted_lift, support, lift
- ranking_score
- discount_reference
- is_approved, is_used
- rank_position
- created_at
```

**CsvUpload.bundle_generation_metrics (JSONB):**
Stores complete metrics including:
- `staged_wave_state`: Full wave publishing state
- `bundle_counts`: Per-type counts
- `drop_reasons`: Why bundles were rejected
- `checkpoints`: For resumable generation

### 4. Notification System

#### `/home/user/AI-Bundle-shopify-backend/services/notifications.py`
- `notify_partial_ready()`: Called after quick-start (currently just logs)
- `notify_bundle_ready()`: Called after full generation (currently just logs)

Can be extended to integrate with:
- Email service
- Slack webhooks
- Custom webhooks
- Push notifications

---

## Response Data Examples

### Progress Response (During Generation)
```json
{
  "upload_id": "uuid-123",
  "shop_domain": "myshop.myshopify.com",
  "step": "ml_generation",
  "progress": 65,
  "status": "in_progress",
  "message": "Generating ML candidates…",
  "metadata": {
    "active_tasks": 8,
    "checkpoint": {
      "phase": "ml_generation",
      "timestamp": "2024-01-15T10:30:45.123Z"
    },
    "partial_bundle_count": 24
  },
  "updated_at": "2024-01-15T10:30:50.000Z"
}
```

### Wave Publishing Response
```json
{
  "upload_id": "uuid-123",
  "shop_domain": "myshop.myshopify.com",
  "step": "staged_publish",
  "progress": 100,
  "status": "in_progress",
  "message": "Publishing wave 2 of 5",
  "staged": true,
  "staged_state": {
    "run_id": "run-456",
    "waves": [
      {
        "index": 0,
        "target": 3,
        "published": 3,
        "drops": {"OUT_OF_STOCK": 0},
        "took_ms": 1200,
        "finalize_tx": "committed"
      },
      {
        "index": 1,
        "target": 5,
        "published": 5,
        "drops": {"DUPLICATE_SKU": 1},
        "took_ms": 1500,
        "finalize_tx": "committed"
      }
    ],
    "totals": {"published": 8, "dropped": 1},
    "cursor": {
      "stage_idx": 1,
      "published": 8,
      "last_bundle_id": "bundle-xyz"
    },
    "backpressure": {"active": false},
    "next_wave_eta_sec": 2
  },
  "updated_at": "2024-01-15T10:31:05.000Z"
}
```

### Final Recommendations Response
```json
[
  {
    "id": "rec-001",
    "csvUploadId": "upload-123",
    "shopId": "shop-456",
    "bundleType": "FBT",
    "objective": "increase_aov",
    "products": [
      {
        "sku": "SKU001",
        "name": "Product A",
        "price": "29.99",
        "variant_id": "var-123"
      },
      {
        "sku": "SKU002",
        "name": "Product B",
        "price": "19.99",
        "variant_id": "var-124"
      }
    ],
    "pricing": {
      "bundle_price": "44.99",
      "original_price": "49.98",
      "discount_percent": 10,
      "discount_type": "percentage"
    },
    "aiCopy": {
      "title": "Product Bundle Deal",
      "description": "Get both products together and save 10%. Perfect combination for..."
    },
    "confidence": "0.85",
    "predictedLift": "0.25",
    "support": "0.12",
    "lift": "2.15",
    "rankingScore": 8.5,
    "discountReference": "BUNDLE_001",
    "isApproved": false,
    "isUsed": false,
    "rankPosition": 1,
    "createdAt": "2024-01-15T10:35:20.000Z"
  }
]
```

---

## Generation Pipeline Phases

### 1. Enrichment (5-25%)
- Load product metadata
- Enrich with catalog attributes
- **Metadata:** `{"checkpoint": {...}}`

### 2. Scoring (30-45%)
- Evaluate business objectives
- Assign objective scores
- **Metadata:** `{"checkpoint": {...}}`

### 3. ML Generation (50-70%)
- Generate candidate bundles using ML
- Run association rules engine
- **Metadata:** `{"active_tasks": n, "partial_bundle_count": n}`

### 4. Optimization (75-85%)
- Deduplicate candidates
- Apply fallback rules
- Inject high-margin bundles
- **Metadata:** `{"dataset_profile": {...}, "remaining_bundles": n}`

### 5. AI Descriptions (88-92%)
- Generate product descriptions using LLM
- Calculate pricing
- **Metadata:** `{"initial_bundle_count": n}`

### 6. Finalization (95-100%)
- Persist to database
- Rank final results
- **Metadata:** Complete metrics with bundle counts

### 7. Staged Publish (100%, optional)
- Publish in waves (3, 5, 10, 20, 40)
- Progressive delivery to frontend
- **Metadata:** Full wave state with progress

---

## Configuration Flags

### Feature Flags (Feature Manager)
```python
feature_flags.get_flag("bundling.staged_publish_enabled", True)
feature_flags.get_flag("bundling.staged_thresholds", [3, 5, 10, 20, 40])
feature_flags.get_flag("bundling.staged.wave_cooldown_ms", 0)
```

### Environment Variables
```bash
QUICK_START_ENABLED=true              # Enable fast preview for first-time users
QUICK_START_TIMEOUT_SECONDS=120       # 2 minutes for preview
QUICK_START_MAX_PRODUCTS=50           # Limit products for speed
QUICK_START_MAX_BUNDLES=10            # Limit bundles for speed

BUNDLE_GENERATION_HARD_TIMEOUT_ENABLED=false
BUNDLE_GENERATION_TIMEOUT_SECONDS=360 # 6 minutes hard limit
BUNDLE_GENERATION_SOFT_WATCHDOG_SECONDS=1200 # 20 min watchdog

BUNDLE_HEARTBEAT_SECONDS=30           # Progress update frequency
BUNDLE_SOFT_TIMEOUT_WARNING=45        # Warn when time running out

PHASE3_CONCURRENCY_LIMIT=6            # ML task concurrency
ASYNC_DEFER_MIN_TIME=45               # Minimum time for deferral
ASYNC_DEFER_CANDIDATE_THRESHOLD=120   # Max candidates before defer
```

---

## Integration Checklist for Frontend

### 1. Initiate Generation
```javascript
POST /api/generate-bundles
Body: { csvUploadId: "string" }
Response: { success: true, message: "..." }
```

### 2. Poll Progress (Recommended: every 2-5 seconds)
```javascript
GET /api/generation-progress/{uploadId}
Response: {
  step, progress, status, message, 
  metadata, staged_state, ...
}

// Show progress bar with step
// Display message to user
// If staged, show wave progress
// If partial available, show preview
```

### 3. Handle Different Steps
- **0-5%**: Initializing
- **5-25%**: Loading products (enrichment)
- **30-45%**: Analyzing data (scoring)
- **50-70%**: Generating bundles (ML)
- **75-85%**: Optimizing (deduplication, fallbacks)
- **88-92%**: Writing descriptions
- **95-100%**: Finalizing

### 4. Staged Publish (if enabled)
Monitor `staged_state.cursor.stage_idx` and `waves` array for wave-by-wave progress

### 5. Fetch Results
```javascript
// Partial results (during generation)
GET /api/bundle-recommendations/{uploadId}/partial

// Final results (after completion)
GET /api/bundle-recommendations?uploadId={uploadId}
```

### 6. Handle Errors
- Check `status === "failed"`
- Display `message` to user
- Check `metadata.partial_recommendations_found` for partial results
- Offer resume option if `step === "optimization"`

### 7. Resume Generation
```javascript
POST /api/generate-bundles/{uploadId}/resume
Response: { success: true, message: "..." }
```

---

## Performance Notes

1. **Quick-Start Mode** provides results in 45 seconds for first-time users
2. **Wave Publishing** prevents blocking - users see results progressively
3. **Soft Watchdog** (80% time) defers heavy phases to background
4. **Heartbeat** updates every 30 seconds (configurable)
5. **Checkpoints** allow resuming from failure points

---

## Debugging Tips

### Check Current Progress
```sql
SELECT step, progress, status, message, metadata, updated_at
FROM generation_progress
WHERE upload_id = '{upload_id}'
ORDER BY updated_at DESC LIMIT 1;
```

### Check Wave State
```sql
SELECT bundle_generation_metrics -> 'staged_wave_state' as waves
FROM csv_uploads
WHERE id = '{upload_id}';
```

### Check All Recommendations Generated
```sql
SELECT COUNT(*), bundle_type, objective
FROM bundle_recommendations
WHERE csv_upload_id = '{upload_id}'
GROUP BY bundle_type, objective;
```

### Monitor Progress in Logs
```bash
# Look for progress updates
grep "Progress updated" /var/log/bundle_generator.log
grep "Wave\|staged" /var/log/bundle_generator.log
```

---

## See Also

- `/home/user/AI-Bundle-shopify-backend/FRONTEND_RESPONSE_STEPS.md` - Complete data format reference
- `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py` - API implementations
- `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py` - Generation logic
