# Quick Install Implementation Guide

Complete backend implementation for rapid bundle generation with status tracking, cooldown periods, and error handling.

---

## **Features Implemented**

✅ **Fast Upload** - Returns immediately without waiting
✅ **Async Processing** - Generates bundles in background
✅ **Status Polling** - Frontend polls every 2 seconds
✅ **Cooldown Period** - 30-day cooldown between installs
✅ **Retry Logic** - Allow retries after failures
✅ **Duplicate Prevention** - Block concurrent installs
✅ **Timeout Handling** - Mark stuck jobs as failed after 10 minutes
✅ **Error Tracking** - Detailed error messages returned to frontend

---

## **Architecture**

```
POST /api/bundles/quick-install
    ↓
[Create CsvUpload record with status=PROCESSING]
    ↓
[Return job_id immediately]
    ↓
[Trigger _process_quick_install_async in background]
    ├─ Parse CSV
    ├─ Generate bundles (using existing process_csv_background)
    ├─ Count bundle recommendations
    └─ Update status to COMPLETED (or FAILED)
    ↓
Frontend polls GET /api/bundles/quick-install/status
    ├─ Every 2 seconds
    ├─ Stops when status = COMPLETED or FAILED
    └─ Shows appropriate message/UI

Cleanup cron: POST /api/bundles/cron/cleanup-stuck-jobs
    ├─ Runs every 5 minutes (via Cloud Scheduler)
    ├─ Finds jobs stuck in PROCESSING > 10 minutes
    └─ Marks as FAILED with timeout error
```

---

## **API Endpoints**

### **1. POST /api/bundles/quick-install**

**Request:**
```typescript
Content-Type: multipart/form-data

file: File  // CSV file (required)
```

**Success Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PROCESSING",
  "message": "Quick install started. Bundles are being generated...",
  "is_retry": false
}
```

**Error: Already Processing (409):**
```json
{
  "detail": {
    "error": "ALREADY_PROCESSING",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "PROCESSING",
    "started_at": "2025-12-06T10:00:00",
    "elapsed_seconds": 45,
    "message": "Quick install already running. Please wait."
  }
}
```

**Error: Cooldown Active (409):**
```json
{
  "detail": {
    "error": "ALREADY_COMPLETED",
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "completed_at": "2025-11-26T10:00:00",
    "bundles_created": 75,
    "days_since_install": 10,
    "can_retry_in_days": 20,
    "message": "Quick install completed 10 days ago. You can run it again in 20 days."
  }
}
```

**Error: Invalid File (400):**
```json
{
  "detail": "Invalid file: CSV file required"
}
```

---

### **2. GET /api/bundles/quick-install/status**

Returns the status of the most recent quick install for the current shop.

**Response: Never Run**
```json
{
  "has_quick_install": false,
  "can_run": true,
  "message": "Quick install has not been run for this shop"
}
```

**Response: Processing**
```json
{
  "has_quick_install": true,
  "can_run": false,
  "status": "PROCESSING",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "started_at": "2025-12-06T10:00:00",
  "message": "Quick install is in progress"
}
```

**Response: Completed (Within Cooldown)**
```json
{
  "has_quick_install": true,
  "can_run": false,
  "status": "COMPLETED",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "completed_at": "2025-11-26T10:00:00",
  "bundles_created": 75,
  "days_since_install": 10,
  "can_retry_in_days": 20,
  "message": "Quick install completed - Can retry in 20 days"
}
```

**Response: Completed (Cooldown Expired)**
```json
{
  "has_quick_install": true,
  "can_run": true,
  "status": "COMPLETED",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "completed_at": "2025-10-27T10:00:00",
  "bundles_created": 75,
  "days_since_install": 40,
  "can_retry_in_days": 0,
  "message": "Quick install completed - Can retry in 0 days"
}
```

**Response: Failed**
```json
{
  "has_quick_install": true,
  "can_run": true,
  "status": "FAILED",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "error_message": "CSV parsing failed: Invalid product IDs",
  "message": "Quick install failed. You can try again."
}
```

---

### **3. GET /api/bundles/status/{job_id}**

Generic endpoint for checking bundle generation status (works for any job, not just quick install).

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "COMPLETED",
  "processed_count": 75,
  "error_count": 0,
  "error_message": null,
  "created_at": "2025-12-06T10:00:00",
  "processed_at": "2025-12-06T10:02:30"
}
```

---

### **4. POST /api/bundles/cron/cleanup-stuck-jobs**

Cleanup cron job - call periodically to mark stuck jobs as failed.

**Request:**
```
POST /api/bundles/cron/cleanup-stuck-jobs
Authorization: Bearer <your-cron-token>  // Optional: add auth to prevent abuse
```

**Response:**
```json
{
  "cleaned": 2,
  "message": "Marked 2 stuck jobs as FAILED"
}
```

---

## **Database Fields Used**

The implementation uses the existing `CsvUpload` model:

| Field | Purpose | Value |
|-------|---------|-------|
| `id` | Unique job identifier | UUID |
| `shop_id` | Shop identifier | Resolved shop ID |
| `filename` | Original file name | `quick_install_YYYYMMDDTHHMMSSz.csv` |
| `csv_type` | Type of CSV | `"quick_install"` |
| `status` | Current status | `processing`, `completed`, `failed` |
| `total_rows` | Number of rows | Actual row count from CSV |
| `processed_rows` | Bundles created | Count of bundle recommendations |
| `error_message` | Error details | Exception message or timeout message |
| `processing_params` | Metadata | `{"source": "quick_install", "file_name": "...", "row_count": ...}` |
| `created_at` | Job start time | ISO timestamp |
| `updated_at` | Last status update | ISO timestamp |

No database migrations required - uses existing schema.

---

## **Implementation Details**

### **1. Async Processing Without Waiting**

```python
# Upload endpoint creates job and returns immediately
@router.post("/quick-install")
async def quick_install(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # 1. Validate file
    # 2. Create CsvUpload record (status="processing")
    # 3. Queue async task
    background_tasks.add_task(
        _process_quick_install_async,
        csv_content,
        upload_id,
        shop_id
    )
    # 4. Return immediately
    return {"job_id": upload_id, "status": "PROCESSING"}
```

### **2. Background Processing with Status Updates**

```python
async def _process_quick_install_async(csv_content, upload_id, shop_id):
    try:
        # Process using existing CSV processor
        await process_csv_background(csv_content, upload_id, "quick_install")

        # Count bundles created
        bundle_count = await _count_bundle_recommendations(upload_id, db)

        # Update status to COMPLETED
        await db.execute(
            update(CsvUpload)
            .where(CsvUpload.id == upload_id)
            .values(
                status="completed",
                processed_rows=bundle_count,
                updated_at=datetime.utcnow()
            )
        )
    except Exception as e:
        # Update status to FAILED
        await db.execute(
            update(CsvUpload)
            .where(CsvUpload.id == upload_id)
            .values(
                status="failed",
                error_message=str(e),
                updated_at=datetime.utcnow()
            )
        )
```

### **3. Duplicate Prevention**

```python
# Check for existing install before creating new one
existing = await _get_shop_quick_install(shop_id, db)

if existing and existing.status == "processing":
    # Block: Already processing
    raise HTTPException(409, detail={"error": "ALREADY_PROCESSING", ...})

if existing and existing.status == "completed":
    days_since = (datetime.utcnow() - existing.updated_at).days
    if days_since < 30:
        # Block: Within cooldown period
        raise HTTPException(409, detail={"error": "ALREADY_COMPLETED", ...})
```

### **4. Cooldown Calculation**

```python
days_since = (datetime.utcnow() - existing.updated_at).days
days_remaining = QUICK_INSTALL_COOLDOWN_DAYS - days_since  # 30 - days_since

if days_remaining > 0:
    # Within cooldown: Block upload
    can_run = False
else:
    # Cooldown expired: Allow new install
    can_run = True
```

### **5. Cleanup of Stuck Jobs**

```python
@router.post("/cron/cleanup-stuck-jobs")
async def cleanup_stuck_jobs(db: AsyncSession):
    stuck_threshold = datetime.utcnow() - timedelta(minutes=10)

    # Find jobs stuck >10 minutes
    stuck_jobs = await db.execute(
        select(CsvUpload).where(
            CsvUpload.status == "processing",
            CsvUpload.created_at < stuck_threshold
        )
    ).scalars().all()

    # Mark as FAILED
    for job in stuck_jobs:
        job.status = "failed"
        job.error_message = "Processing timed out after 10 minutes"
        job.updated_at = datetime.utcnow()

    await db.commit()
    return {"cleaned": len(stuck_jobs)}
```

---

## **Frontend Integration**

### **1. Upload Flow**

```typescript
const uploadCSV = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  // 1. Upload (returns immediately)
  const uploadResponse = await fetch('/api/bundles/quick-install', {
    method: 'POST',
    body: formData
  });

  const { job_id, status } = await uploadResponse.json();
  setJobId(job_id);

  // 2. Start polling
  pollStatus(job_id);
};
```

### **2. Polling Loop**

```typescript
const pollStatus = (job_id: string) => {
  const interval = setInterval(async () => {
    const response = await fetch(`/api/bundles/quick-install/status`);
    const data = await response.json();

    setStatus(data.status);

    if (data.status === 'COMPLETED') {
      showSuccess(`Created ${data.bundles_created} bundles`);
      clearInterval(interval);
    } else if (data.status === 'FAILED') {
      showError(`Failed: ${data.error_message}`);
      clearInterval(interval);
    }
  }, 2000);  // Poll every 2 seconds
};
```

### **3. Resume on Page Refresh**

```typescript
useEffect(() => {
  // On mount, check if there's an active job
  const checkActiveJob = async () => {
    const response = await fetch('/api/bundles/quick-install/status');
    const data = await response.json();

    if (data.has_quick_install && data.status === 'PROCESSING') {
      // Resume polling
      pollStatus(data.job_id);
    } else if (data.has_quick_install && data.status === 'COMPLETED') {
      showCompletion(data);
    }
  };

  checkActiveJob();
}, []);
```

---

## **Testing**

### **1. Manual Testing: Successful Install**

```bash
# 1. Create test CSV
cat > test.csv << 'EOF'
product_id,name,price
1,Product A,29.99
2,Product B,39.99
3,Product C,49.99
EOF

# 2. Upload
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv"

# Response:
# {"job_id": "abc-123", "status": "PROCESSING"}

# 3. Poll status (wait a few seconds between calls)
curl http://localhost:8000/api/bundles/quick-install/status

# Should see: status changing from PROCESSING → COMPLETED
```

### **2. Manual Testing: Duplicate Prevention**

```bash
# 1. Upload first CSV
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv"

# 2. Try uploading again immediately (while first is processing)
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv"

# Should get 409 Conflict: ALREADY_PROCESSING
```

### **3. Manual Testing: Cooldown**

```bash
# After successful install:
curl http://localhost:8000/api/bundles/quick-install/status

# Should return:
# {
#   "has_quick_install": true,
#   "can_run": false,
#   "status": "COMPLETED",
#   "days_since_install": 0,
#   "can_retry_in_days": 30
# }

# Try uploading again:
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv"

# Should get 409 Conflict: ALREADY_COMPLETED
```

### **4. Automated Tests (pytest)**

```python
# tests/test_quick_install.py

@pytest.mark.asyncio
async def test_quick_install_success(client, db):
    """Test successful quick install flow"""
    csv_file = ("test.csv", "product_id,name\n1,Product A", "text/csv")

    response = client.post(
        "/api/bundles/quick-install",
        files={"file": csv_file}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROCESSING"
    assert "job_id" in data

    # Wait for processing
    await asyncio.sleep(2)

    # Check status
    status_response = client.get("/api/bundles/quick-install/status")
    status_data = status_response.json()
    assert status_data["status"] == "COMPLETED"
    assert status_data["bundles_created"] > 0


@pytest.mark.asyncio
async def test_quick_install_duplicate_prevention(client):
    """Test that concurrent uploads are blocked"""
    csv_file = ("test.csv", "product_id,name\n1,Product A", "text/csv")

    # First upload
    response1 = client.post(
        "/api/bundles/quick-install",
        files={"file": csv_file}
    )
    assert response1.status_code == 200

    # Second upload (while first processing)
    response2 = client.post(
        "/api/bundles/quick-install",
        files={"file": csv_file}
    )
    assert response2.status_code == 409
    data = response2.json()["detail"]
    assert data["error"] == "ALREADY_PROCESSING"


@pytest.mark.asyncio
async def test_quick_install_cooldown(client, db):
    """Test 30-day cooldown after successful install"""
    csv_file = ("test.csv", "product_id,name\n1,Product A", "text/csv")

    # First upload
    response1 = client.post(
        "/api/bundles/quick-install",
        files={"file": csv_file}
    )
    assert response1.status_code == 200
    job_id = response1.json()["job_id"]

    # Manually mark as completed
    await db.execute(
        update(CsvUpload)
        .where(CsvUpload.id == job_id)
        .values(status="completed", updated_at=datetime.utcnow())
    )
    await db.commit()

    # Try second upload within 30 days
    response2 = client.post(
        "/api/bundles/quick-install",
        files={"file": csv_file}
    )
    assert response2.status_code == 409
    data = response2.json()["detail"]
    assert data["error"] == "ALREADY_COMPLETED"
    assert data["can_retry_in_days"] == 30
```

---

## **Deployment**

### **1. Local Testing**
```bash
# Start server
python -m uvicorn main:app --reload --port 8000

# Test endpoints
./test_cors.sh  # CORS already fixed
```

### **2. Cloud Run Deployment**

```bash
# Build and deploy
gcloud run deploy bundle-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

# Verify endpoints are accessible
curl https://bundle-api-XXXX.run.app/api/bundles/quick-install/status
```

### **3. Setup Cleanup Cron (Cloud Scheduler)**

```bash
# Create Cloud Scheduler job to call cleanup endpoint every 5 minutes
gcloud scheduler jobs create http cleanup-quick-install \
  --location us-central1 \
  --schedule "*/5 * * * *" \
  --uri "https://bundle-api-XXXX.run.app/api/bundles/cron/cleanup-stuck-jobs" \
  --http-method POST \
  --oidc-service-account-email YOUR-SERVICE-ACCOUNT@PROJECT.iam.gserviceaccount.com \
  --oidc-token-audience "https://bundle-api-XXXX.run.app"
```

---

## **Logging**

All operations are logged with structured JSON logging for Cloud Logging:

```
[quick_install] Received upload from request
[quick_install] CSV validated: 75 rows
[quick_install] Created CsvUpload: job-123
[quick_install] Triggered async processing for job-123
[quick_install] ✅ Upload job-123 completed: 75 bundles created
```

Monitor in Cloud Logging:
```bash
gcloud logging read "resource.type=cloud_run_revision AND severity=INFO" \
  --limit 50 \
  --format json | jq '.[] | select(.textPayload | contains("quick_install"))'
```

---

## **Troubleshooting**

### **Problem: Status stays PROCESSING forever**
- Check logs: `gcloud logging read "resource.labels.revision_name=bundle-api-*" --limit 100`
- Manual fix: Call cleanup endpoint to mark as FAILED
- Set PROCESSING_TIMEOUT_MINUTES = 5 for faster timeout

### **Problem: Bundles not created but status=COMPLETED**
- Check if `process_csv_background` is working correctly
- Verify bundle generation logic in bundle_recommendations.py
- Check bundle_recommendations table: `SELECT COUNT(*) FROM bundle_recommendations WHERE csv_upload_id='job-123'`

### **Problem: Frontend polling gets 409 ALREADY_COMPLETED, but wants to retry**
- This is expected behavior - cooldown period is active
- Can retry after: `can_retry_in_days` days
- Or manually mark as failed: `UPDATE csv_uploads SET status='failed' WHERE id='job-123'`

---

## **Configuration**

Tunable parameters in `routers/quick_install.py`:

```python
QUICK_INSTALL_COOLDOWN_DAYS = 30        # Days between installs
PROCESSING_TIMEOUT_MINUTES = 10         # Max processing time
```

---

## **Next Steps**

- [ ] Implement actual shop_id extraction from authentication context
- [ ] Add authentication/authorization for cleanup endpoint
- [ ] Setup Cloud Scheduler for cleanup cron job
- [ ] Add metrics/monitoring for success rates and processing times
- [ ] Implement retry strategy with exponential backoff
- [ ] Add email notifications on completion/failure

---

## **Related Documentation**

- [CORS Configuration](CORS_FIX.md)
- [API Response Schemas](API_RESPONSE_SCHEMAS.md)
- [Sync Flow Diagrams](FLOW_DIAGRAMS.md)
