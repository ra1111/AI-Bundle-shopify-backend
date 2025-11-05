# Frontend Response Data Flow During Bundle Generation

## Overview
This document details all data sent from the backend to the frontend during bundle generation, including API endpoints, response structures, step-by-step progress updates, and streaming/progressive delivery mechanisms.

---

## 1. API ENDPOINTS FOR BUNDLE GENERATION

### 1.1 Bundle Generation Initiation
**Endpoint:** `POST /api/generate-bundles`

**Request:**
```json
{
  "csvUploadId": "string"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Bundle recommendations generation started for CSV upload {upload_id}"
}
```

**File:** `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:51-76`

---

### 1.2 Progress Polling Endpoint (Primary Data Source)
**Endpoint:** `GET /api/generation-progress/{upload_id}`

**Response Structure:**
```json
{
  "upload_id": "string",
  "shop_domain": "string",
  "step": "string",                    // Current phase (see steps below)
  "progress": 0-100,                   // Percentage complete
  "status": "in_progress|completed|failed",
  "message": "string",                 // Human-readable status message
  "metadata": {
    // Dynamic metadata per step (see Section 2)
  },
  "updated_at": "2024-01-01T00:00:00Z",
  
  // Optional staged publish fields
  "staged": true|false,
  "staged_state": {
    "run_id": "string",
    "waves": [...],
    "totals": {...},
    "cursor": {...},
    "backpressure": {...},
    "next_wave_eta_sec": number
  },
  "run_id": "string"                   // Available in staged state
}
```

**File:** `/home/user/AI-Bundle-shopify-backend/routers/generation_progress.py:20-68`

---

### 1.3 Partial Bundles Preview (Progressive Results)
**Endpoint:** `GET /api/bundle-recommendations/{upload_id}/partial`

**Response:** Array of partial bundle recommendations
```json
[
  {
    "id": "string",
    "csvUploadId": "string",
    "bundleType": "FBT|VOLUME_DISCOUNT|MIX_MATCH|BXGY|FIXED",
    "objective": "string",
    "products": [...],
    "pricing": {...},
    "aiCopy": {...},
    "confidence": "decimal",
    "rankingScore": number,
    "discountReference": "string",
    "isPartial": true,
    "createdAt": "2024-01-01T00:00:00Z"
  }
]
```

**File:** `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:143-172`

---

### 1.4 Resume Bundle Generation
**Endpoint:** `POST /api/generate-bundles/{upload_id}/resume`

**Response:**
```json
{
  "success": true,
  "message": "Bundle generation resume queued for {upload_id}"
}
```

**File:** `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:175-184`

---

### 1.5 Final Bundle Recommendations
**Endpoint:** `GET /api/bundle-recommendations?uploadId={uploadId}`

**Response:** Array of final bundle recommendations
```json
[
  {
    "id": "string",
    "csvUploadId": "string",
    "shopId": "string",
    "bundleType": "FBT|VOLUME_DISCOUNT|MIX_MATCH|BXGY|FIXED",
    "objective": "string",
    "products": {
      // Product details with SKU, name, price, etc.
    },
    "pricing": {
      // Bundle pricing structure
    },
    "aiCopy": {
      "title": "string",
      "description": "string"
      // Generated marketing copy
    },
    "confidence": "decimal",
    "predictedLift": "decimal",
    "support": "decimal",
    "lift": "decimal",
    "rankingScore": number,
    "discountReference": "string",
    "isApproved": boolean,
    "isUsed": boolean,
    "rankPosition": integer,
    "createdAt": "2024-01-01T00:00:00Z"
  }
]
```

**File:** `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:78-140`

---

## 2. STEP-BY-STEP PROGRESS DATA

The bundle generation pipeline sends progress updates through the `GenerationProgress` table with the following steps and associated metadata:

### Progress Step Sequence and Data:

| Step | Progress | Status | Message | Metadata |
|------|----------|--------|---------|----------|
| `queueing` | 0% | in_progress | "Bundle generation queued; waiting for background worker." | - |
| `enrichment` | 5-25% | in_progress | "Starting enrichment…" or "Enrichment complete." | `{"checkpoint": {...}}` |
| `scoring` | 30-45% | in_progress | "Scoring objectives…" | `{"checkpoint": {...}}` |
| `ml_generation` | 50-70% | in_progress | "Generating ML candidates…" | `{"active_tasks": number}` |
| `optimization` | 75-85% | in_progress | "Optimizing bundle candidates…" | `{"dataset_profile": {...}}` |
| `ai_descriptions` | 88-92% | in_progress | "Preparing AI descriptions…" | `{"initial_bundle_count": number}` |
| `finalization` | 95-100% | completed/failed | "Bundle generation complete." | Various |
| `staged_publish` | 100% | in_progress | "Publishing wave {index}..." | Staged publish state |

**File:** `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py` (progress calls throughout)

### Metadata by Phase:

#### Enrichment Phase
```json
{
  "checkpoint": {
    "phase": "enrichment",
    "timestamp": "ISO8601",
    "bundle_count": number
  }
}
```

#### Scoring Phase
```json
{
  "checkpoint": {
    "phase": "scoring",
    "timestamp": "ISO8601"
  }
}
```

#### ML Generation Phase
```json
{
  "active_tasks": number,
  "checkpoint": {
    "phase": "ml_generation",
    "timestamp": "ISO8601"
  },
  "partial_bundle_count": number,
  "resume_phase": "string"
}
```

#### Optimization Phase
```json
{
  "dataset_profile": {
    "tier": "small|medium|large",
    "transaction_count": number,
    "unique_sku_count": number,
    "candidate_count": number,
    "avg_transactions_per_sku": number,
    "time_remaining_seconds": number,
    "defer_candidate": boolean
  },
  "checkpoint": {...},
  "remaining_bundles": number
}
```

#### AI Descriptions Phase
```json
{
  "initial_bundle_count": number
}
```

#### Finalization Phase
```json
{
  "checkpoint": {
    "phase": "finalization",
    "timestamp": "ISO8601",
    "bundle_count": number
  },
  "metrics": {
    "total_recommendations": number,
    "processing_time_ms": number,
    "bundle_counts": {
      "FBT": number,
      "VOLUME_DISCOUNT": number,
      "MIX_MATCH": number,
      "BXGY": number,
      "FIXED": number
    },
    "drop_reasons": {
      "OUT_OF_STOCK": number,
      "POLICY_BLOCK": number,
      "BELOW_MARGIN": number,
      "DUPLICATE_SKU": number,
      "LOW_SCORE": number,
      "PRICE_ANOMALY": number,
      "COPY_FAIL": number,
      "FINALIZE_TX_FAIL": number
    }
  }
}
```

**File:** `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py:500-633`

---

## 3. PROGRESSIVE/STREAMING DATA - STAGED PUBLISH

### 3.1 Staged Publish Wave Data

The system implements **progressive delivery through waves** - bundles are published in waves rather than all at once, allowing the frontend to show results incrementally.

**Staged State Structure:**
```json
{
  "run_id": "string",
  "staged": true,
  "waves": [
    {
      "index": 0,
      "target": 3,                    // Target bundle count for this wave
      "published": 3,                 // Actual published count
      "drops": {
        "OUT_OF_STOCK": 0,
        "POLICY_BLOCK": 0,
        "DUPLICATE_SKU": 0
        // ... other drop reasons
      },
      "took_ms": 1200,                // Duration of this wave
      "finalize_tx": "committed"      // Transaction status
    }
  ],
  "totals": {
    "published": 10,
    "dropped": 2
  },
  "cursor": {
    "stage_idx": 1,                   // Current wave index
    "published": 10,                  // Total published so far
    "last_bundle_id": "string"
  },
  "backpressure": {
    "active": false,
    "reason": null,
    "last_event": null
  },
  "next_wave_eta_sec": 2             // Estimated time until next wave
}
```

### 3.2 Wave Publishing Process

**Wave Thresholds (Configurable):**
Default: `[3, 5, 10, 20, 40]` bundles per wave

**Wave Cycle:**
1. Load staged state from database
2. Publish first wave (3 bundles)
3. Update progress with wave metadata
4. Wait for cooldown (default varies by config)
5. Publish next wave (5 bundles)
6. Continue until all bundles published or timeout

**Files:**
- Staged state loading: `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py:280-364`
- Wave publishing: `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py:1063-1193`

---

## 4. DATABASE MODELS (STORED DATA)

### 4.1 GenerationProgress Table

**Schema:**
```
upload_id (PK)          - Upload identifier
shop_domain             - Shop domain
step (TEXT)             - Current phase
progress (INTEGER)      - 0-100 percentage
status (TEXT)           - in_progress|completed|failed
message (TEXT)          - Human-readable status
metadata_json (JSONB)   - Phase-specific metadata
updated_at (DATETIME)   - Last update timestamp
```

**File:** `/home/user/AI-Bundle-shopify-backend/database.py:227-250`

### 4.2 BundleRecommendation Table

**Schema:**
```
id (PK)                 - Recommendation ID
csv_upload_id (FK)      - Associated upload
shop_id                 - Shop identifier
bundle_type (TEXT)      - FBT, VOLUME_DISCOUNT, MIX_MATCH, BXGY, FIXED
objective (TEXT)        - increase_aov, clear_slow_movers, etc.
products (JSONB)        - Product list with SKUs, names, prices
pricing (JSONB)         - Bundle pricing structure
ai_copy (JSONB)         - Generated marketing copy (title, description)
confidence (DECIMAL)    - Confidence score [0-1]
predicted_lift (DECIMAL) - Predicted sales lift
support (DECIMAL)       - Association rule support
lift (DECIMAL)          - Association rule lift
ranking_score (DECIMAL) - Overall bundle ranking score
discount_reference      - Discount code or policy reference
is_approved (BOOLEAN)   - Merchant approval status
is_used (BOOLEAN)       - Whether bundle is active
rank_position (INT)     - Position in ranked results
created_at (DATETIME)   - Creation timestamp
```

**File:** `/home/user/AI-Bundle-shopify-backend/database.py:448-476`

### 4.3 CsvUpload Table (Metrics Storage)

**Key JSONB Field: `bundle_generation_metrics`**

Contains:
- `total_recommendations`: Final bundle count
- `processing_time_ms`: Total generation time
- `generation_time_seconds`: Total duration
- `generation_timestamp`: When completed
- `bundle_counts`: Per-type bundle counts
- `drop_reasons`: Bundles dropped per reason
- `staged_wave_state`: Full staged publish state
- `checkpoints`: Resume checkpoints
- `latest_metrics_snapshot`: Latest metrics
- `async_deferred`: Whether deferred to background
- `dataset_profile`: Data tier and characteristics

**File:** `/home/user/AI-Bundle-shopify-backend/database.py:200-224`

---

## 5. NOTIFICATION SYSTEM

Notifications are currently logged but can be extended to email/Slack/webhooks:

### 5.1 Partial Ready Notification
Triggered after quick-start generation completes

**Payload:**
```json
{
  "upload_id": "string",
  "bundle_count": number,
  "details": {
    // Quick-start metrics
  }
}
```

### 5.2 Bundle Ready Notification
Triggered after full generation completes

**Payload:**
```json
{
  "upload_id": "string",
  "bundle_count": number,
  "resume_run": boolean,
  "details": {
    // Full generation metrics
  }
}
```

**File:** `/home/user/AI-Bundle-shopify-backend/services/notifications.py:13-42`

---

## 6. QUICK-START MODE (First-Time Users)

For first-time installations, the system runs a **quick-start mode** that provides fast preview bundles within 2 minutes:

**Quick-Start Steps:**
| Step | Progress | Time |
|------|----------|------|
| enrichment | 10% | ~0.5s |
| scoring | 40% | ~5s |
| ml_generation | 70% | ~30s |
| finalization | 90% | ~10s |
| Complete | 100% | ~45s total |

**Response Format:** Same as normal bundles, but with `discount_reference` starting with `__quick_start_`

After quick-start completes:
1. Partial results returned to frontend
2. Full generation scheduled in background
3. When full generation completes, quick-start bundles replaced
4. Final bundle ready notification sent

**File:** `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:257-344`

---

## 7. DATA FLOW DIAGRAM

```
Frontend
   |
   v
POST /api/generate-bundles
   |
   +---> Background Task Started
   |
   v
GET /api/generation-progress/{upload_id} (polling)
   |
   +---> Progress: 0%  (queueing)
   +---> Progress: 5-25% (enrichment) + checkpoint
   +---> Progress: 30-45% (scoring) + metrics
   +---> Progress: 50-70% (ml_generation) + active_tasks
   +---> Progress: 75-85% (optimization) + dataset_profile
   +---> Progress: 88-92% (ai_descriptions) + bundle_count
   +---> Progress: 95% (finalization)
   |     |
   |     v (if staged publish enabled)
   |     staged_publish step with wave data
   |     Wave 1: Published 3, Dropped 0
   |     Wave 2: Published 5, Dropped 1
   |     Wave 3: Published 10, Dropped 2
   |
   v
Progress: 100% (completed) + final metrics

GET /api/bundle-recommendations/{upload_id}/partial
   |---> Returns partial bundles (if not completed yet)

GET /api/bundle-recommendations?uploadId={uploadId}
   |---> Returns final recommendations when complete
```

---

## 8. CONFIGURATION/FEATURE FLAGS

Controlled via environment variables and feature flags:

**Key Flags:**
- `QUICK_START_ENABLED` (default: true)
- `bundling.staged_publish_enabled` (default: true)
- `bundling.staged_thresholds` (default: [3, 5, 10, 20, 40])
- `bundling.staged.wave_cooldown_ms` (default: varies)
- `BUNDLE_GENERATION_HARD_TIMEOUT_ENABLED` (default: false)
- `BUNDLE_GENERATION_TIMEOUT_SECONDS` (default: 360)
- `BUNDLE_HEARTBEAT_SECONDS` (default: 30)

**File:** `/home/user/AI-Bundle-shopify-backend/services/bundle_generator.py:711-880`

---

## 9. ERROR HANDLING AND TIMEOUTS

### Soft Timeout (Watchdog)
After 80% completion time, system transitions to async deferral if heavy phases remain:
```json
{
  "step": "optimization",
  "progress": 78,
  "status": "in_progress",
  "message": "Continuing asynchronously after 290 seconds."
}
```

### Hard Timeout
If enabled, hard timeout after specified seconds returns partial results:
```json
{
  "step": "finalization",
  "progress": 100,
  "status": "failed",
  "message": "Bundle generation timed out after 360 seconds.",
  "metadata": {
    "timeout_error": true,
    "timeout_phase": "phase_name",
    "partial_recommendations_found": number
  }
}
```

**Files:**
- Soft timeout: `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:366-405`
- Hard timeout: `/home/user/AI-Bundle-shopify-backend/routers/bundle_recommendations.py:415-451`

---

## 10. SUMMARY TABLE: WHAT DATA IS SENT AT EACH STEP

| When | What | How | Where |
|------|------|-----|-------|
| Generate starts | Acknowledgment | HTTP 200 | POST /generate-bundles |
| Every 30s | Progress + metadata | Progress poll | GET /generation-progress |
| ~45s in (quick-start) | Partial bundles | API call | GET /partial endpoint |
| Each wave (staged) | Wave completion | Progress metadata | generation-progress |
| Complete | Final 50 bundles | API query | GET /bundle-recommendations |
| Error | Error message + partial | Progress update | generation-progress |

---

## Database Query Examples

### Get current progress
```sql
SELECT step, progress, status, message, metadata, updated_at 
FROM generation_progress 
WHERE upload_id = '{upload_id}';
```

### Get all recommendations for upload
```sql
SELECT id, bundle_type, objective, products, pricing, ai_copy, 
       confidence, ranking_score, created_at
FROM bundle_recommendations 
WHERE csv_upload_id = '{upload_id}'
ORDER BY rank_position ASC, ranking_score DESC
LIMIT 50;
```

### Get staged publish state
```sql
SELECT bundle_generation_metrics -> 'staged_wave_state' as staged_state
FROM csv_uploads
WHERE id = '{upload_id}';
```

