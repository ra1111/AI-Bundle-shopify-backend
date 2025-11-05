# Frontend Response Data - Quick Reference

## Critical Files and Line Numbers

### API Endpoints

| Endpoint | File | Lines | Purpose |
|----------|------|-------|---------|
| POST /api/generate-bundles | routers/bundle_recommendations.py | 51-76 | Start generation |
| GET /api/generation-progress/{id} | routers/generation_progress.py | 20-68 | Poll progress (PRIMARY) |
| GET /api/bundle-recommendations | routers/bundle_recommendations.py | 78-140 | Get final results |
| GET /api/bundle-recommendations/{id}/partial | routers/bundle_recommendations.py | 143-172 | Get partial results |
| POST /api/generate-bundles/{id}/resume | routers/bundle_recommendations.py | 175-184 | Resume deferred |

### Core Generation Logic

| Component | File | Lines | What It Does |
|-----------|------|-------|--------------|
| Main generation | services/bundle_generator.py | 1600+ lines | Full pipeline orchestration |
| Quick-start mode | routers/bundle_recommendations.py | 257-344 | Fast 45-second preview |
| Progress tracking | services/progress_tracker.py | 36-173 | Upsert progress to DB |
| Wave publishing | services/bundle_generator.py | 1063-1193 | Staged publish logic |
| Heartbeat updates | services/bundle_generator.py | 500-533 | Regular progress sends |

### Database Schema

| Table | File | Lines | Contains |
|-------|------|-------|----------|
| GenerationProgress | database.py | 227-250 | Progress state (upload_id, step, progress, message, metadata) |
| BundleRecommendation | database.py | 448-476 | Final bundles (products, pricing, ai_copy, scores) |
| CsvUpload | database.py | 186-224 | Metrics in bundle_generation_metrics JSONB field |

### Configuration

| Setting | File | Lines | Default |
|---------|------|-------|---------|
| Feature flags | services/bundle_generator.py | 711-880 | Various (see env vars) |
| Quick-start config | routers/bundle_recommendations.py | 39-42 | ENABLED, 120s timeout, 50 products |
| Wave thresholds | services/bundle_generator.py | 825-826 | [3, 5, 10, 20, 40] |
| Heartbeat interval | services/bundle_generator.py | 820 | 30 seconds |
| Timeouts | routers/bundle_recommendations.py | 35-37 | 360s hard, 1200s watchdog |

---

## Response Structures

### Progress Response Fields
```
upload_id, shop_domain, step, progress, status, message, 
metadata, updated_at, [staged], [staged_state], [run_id]
```

### Progress Steps (in order)
1. `queueing` (0%)
2. `enrichment` (5-25%)
3. `scoring` (30-45%)
4. `ml_generation` (50-70%)
5. `optimization` (75-85%)
6. `ai_descriptions` (88-92%)
7. `finalization` (95-100%)
8. `staged_publish` (100%, waves)

### Bundle Recommendation Fields
```
id, csvUploadId, shopId, bundleType, objective,
products, pricing, aiCopy, confidence, predictedLift,
support, lift, rankingScore, discountReference,
isApproved, isUsed, rankPosition, createdAt
```

### Metadata by Step

| Step | Metadata Key | Contains |
|------|--------------|----------|
| enrichment | checkpoint | phase, timestamp, bundle_count |
| scoring | checkpoint | phase, timestamp |
| ml_generation | active_tasks, checkpoint, partial_bundle_count | - |
| optimization | dataset_profile, checkpoint, remaining_bundles | tier, txn_count, sku_count |
| ai_descriptions | initial_bundle_count | - |
| finalization | checkpoint, metrics | bundle_counts, drop_reasons, time |
| staged_publish | staged_state (full) | waves, totals, cursor, backpressure |

---

## Key Implementation Points

### 1. Polling Strategy
- **Endpoint:** GET /api/generation-progress/{uploadId}
- **Interval:** 2-5 seconds recommended
- **Response Time:** Immediate (cached in DB)
- **Fields to Monitor:** step, progress, status, message

### 2. Wave Publishing
- **Enabled by:** Feature flag `bundling.staged_publish_enabled`
- **Waves:** [3, 5, 10, 20, 40] bundles per wave
- **Cool-down:** Configurable (default 0-2 seconds)
- **Show in UI:** Use `staged_state.waves` array for progress

### 3. Partial Results
- **Available at:** ~45% completion (after quick-start)
- **Endpoint:** GET /api/bundle-recommendations/{uploadId}/partial
- **Marked with:** `"isPartial": true`
- **Use Case:** Show preview to user while full generation runs

### 4. Error Recovery
- **Failed Status:** Check `status === "failed"`
- **Partial Results:** Check `metadata.partial_recommendations_found`
- **Resume:** POST /api/generate-bundles/{uploadId}/resume
- **Retry Eligible:** After optimization phase starts

### 5. Quick-Start Mode
- **For:** First-time users
- **Duration:** 45 seconds
- **Results:** Limited to 10 bundles, 50 products
- **Follow-up:** Full generation runs in background
- **Detection:** Check CsvUpload.status for "quick_start" prefix

---

## Response Timing

| Phase | Expected Time | Frontend Action |
|-------|----------------|-----------------|
| Queueing | 0-2s | Show "Starting..." |
| Enrichment | 2-5s | Show progress bar, "Loading products..." |
| Scoring | 5-10s | Show progress bar, "Analyzing data..." |
| ML Generation | 10-40s | Show active tasks count |
| Optimization | 40-50s | Show remaining bundles |
| AI Descriptions | 50-60s | Show "Writing descriptions..." |
| Finalization | 60-80s | Show "Finalizing..." |
| Staged Publish | 80-120s+ | Show wave-by-wave progress |

---

## Troubleshooting

### Check if generation is stuck
```sql
SELECT upload_id, step, progress, updated_at, 
       EXTRACT(EPOCH FROM (NOW() - updated_at)) as seconds_since_update
FROM generation_progress
WHERE upload_id = '{uploadId}'
AND updated_at < NOW() - INTERVAL '5 minutes';
```

### Check final bundle count
```sql
SELECT COUNT(*) as bundle_count, bundle_type
FROM bundle_recommendations
WHERE csv_upload_id = '{uploadId}'
GROUP BY bundle_type;
```

### Check wave state
```sql
SELECT 
  (bundle_generation_metrics -> 'staged_wave_state' -> 'totals' ->> 'published')::int as published,
  (bundle_generation_metrics -> 'staged_wave_state' -> 'totals' ->> 'dropped')::int as dropped,
  jsonb_array_length(bundle_generation_metrics -> 'staged_wave_state' -> 'waves') as wave_count
FROM csv_uploads
WHERE id = '{uploadId}';
```

### Check drop reasons
```sql
SELECT (bundle_generation_metrics -> 'drop_reasons')::jsonb
FROM csv_uploads
WHERE id = '{uploadId}';
```

---

## Performance Tips

1. **Don't poll too fast** - 2-5 second intervals optimal
2. **Cache progress locally** - Reduce unnecessary DB hits
3. **Use heartbeat updates** - Every 30 seconds minimum
4. **Show waves progressively** - Update UI as waves publish
5. **Offer resume early** - After 15% if user wants to abandon
6. **Keep partial bundle UI separate** - Don't mix with final results

---

## Complete Request/Response Examples

### Start Generation
```
POST /api/generate-bundles
Content-Type: application/json

{"csvUploadId": "upload-123"}

Response:
{
  "success": true,
  "message": "Bundle recommendations generation started for CSV upload upload-123"
}
```

### Poll Progress
```
GET /api/generation-progress/upload-123

Response:
{
  "upload_id": "upload-123",
  "shop_domain": "example.myshopify.com",
  "step": "ml_generation",
  "progress": 65,
  "status": "in_progress",
  "message": "Generating ML candidates…",
  "metadata": {
    "active_tasks": 8,
    "partial_bundle_count": 24
  },
  "updated_at": "2024-01-15T10:30:50Z"
}
```

### Get Final Results
```
GET /api/bundle-recommendations?uploadId=upload-123

Response: [
  {
    "id": "rec-001",
    "csvUploadId": "upload-123",
    "bundleType": "FBT",
    "products": [...],
    "pricing": {...},
    "aiCopy": {...},
    "confidence": "0.85",
    "rankingScore": 8.5,
    ...
  }
]
```

---

## Files to Read in Order

1. **Start Here:** FRONTEND_RESPONSE_STEPS.md (this document)
2. **Implementation:** FRONTEND_RESPONSE_IMPLEMENTATION_GUIDE.md
3. **API Code:** routers/bundle_recommendations.py
4. **Progress Code:** routers/generation_progress.py
5. **Logic:** services/bundle_generator.py (key methods)
6. **DB:** database.py (models)
7. **Tracking:** services/progress_tracker.py

