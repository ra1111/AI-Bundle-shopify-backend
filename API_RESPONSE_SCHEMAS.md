# API Response Schemas - Complete Reference

This document provides the **exact** response schemas for all API endpoints.

---

## **1. POST /api/shopify/upload**

### **Request Body**
```typescript
interface UploadRequest {
  shopId: string              // "shop.myshopify.com"
  csvType: "orders" | "variants" | "catalog" | "inventory"
  csvData: string             // Base64 or plain CSV string
  runId?: string              // Optional: group related uploads
  triggerPipeline: boolean    // true = run bundle generation
}
```

**Example:**
```json
{
  "shopId": "shop.myshopify.com",
  "csvType": "orders",
  "csvData": "order_id,sku,quantity\n123,PROD-001,2\n...",
  "triggerPipeline": true
}
```

### **Response (200 OK)**
```typescript
interface UploadResponse {
  uploadId: string            // Use this to poll status
  runId: string               // Groups related uploads
  status: "processing"        // Always "processing" initially
  shopId: string              // Normalized shop ID
  csvType: string             // Normalized CSV type
  triggerPipeline: boolean    // Echo of request
  message: string             // Instructions for frontend
}
```

**Example:**
```json
{
  "uploadId": "abc-123-def-456",
  "runId": "run-789",
  "status": "processing",
  "shopId": "shop.myshopify.com",
  "csvType": "orders",
  "triggerPipeline": true,
  "message": "Upload accepted. Poll /api/shopify/status/{uploadId} for progress."
}
```

### **Response (400 Bad Request)**
```json
{
  "detail": "Invalid csvType 'foo'. Must be one of: catalog_joined, inventory_levels, orders, variants"
}
```

---

## **2. GET /api/shopify/status/{uploadId}**

### **Response (200 OK) - Processing**
```typescript
interface UploadStatus {
  upload_id: string
  shop_id: string | null
  status: "processing" | "completed" | "failed"
  total_rows: number
  processed_rows: number
  error_message: string | null
  bundle_count: number | null  // null until bundles exist
}
```

**Example (CSV Processing):**
```json
{
  "upload_id": "abc-123-def-456",
  "shop_id": "shop.myshopify.com",
  "status": "processing",
  "total_rows": 100,
  "processed_rows": 45,
  "error_message": null,
  "bundle_count": null
}
```

**Example (CSV Done, Bundles Generating):**
```json
{
  "upload_id": "abc-123-def-456",
  "shop_id": "shop.myshopify.com",
  "status": "completed",
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": null  // ← Still null, bundles generating
}
```

**Example (Bundles Ready):**
```json
{
  "upload_id": "abc-123-def-456",
  "shop_id": "shop.myshopify.com",
  "status": "completed",
  "total_rows": 100,
  "processed_rows": 100,
  "error_message": null,
  "bundle_count": 10  // ← Bundles ready!
}
```

**Example (Failed):**
```json
{
  "upload_id": "abc-123-def-456",
  "shop_id": "shop.myshopify.com",
  "status": "failed",
  "total_rows": 50,
  "processed_rows": 23,
  "error_message": "orders.csv schema error → Missing required: order_id",
  "bundle_count": null
}
```

### **Response (404 Not Found)**
```json
{
  "detail": "Upload abc-123 not found"
}
```

---

## **3. GET /generation-progress/{uploadId}**

### **Response (200 OK) - In Progress**
```typescript
interface GenerationProgress {
  upload_id: string
  status: "queued" | "in_progress" | "completed" | "failed"
  progress: number          // 0-100
  step: string              // Current step name
  message: string           // Human-readable status
  created_at: string        // ISO 8601
  updated_at: string        // ISO 8601
}
```

**Example (Generating):**
```json
{
  "upload_id": "abc-123-def-456",
  "status": "in_progress",
  "progress": 45,
  "step": "optimization",
  "message": "Generating bundles…",
  "created_at": "2025-12-02T10:30:00Z",
  "updated_at": "2025-12-02T10:30:25Z"
}
```

**Example (Completed):**
```json
{
  "upload_id": "abc-123-def-456",
  "status": "completed",
  "progress": 100,
  "step": "finalization",
  "message": "Quick-start complete: 10 bundles in 45.2s",
  "created_at": "2025-12-02T10:30:00Z",
  "updated_at": "2025-12-02T10:30:45Z"
}
```

**Example (Failed):**
```json
{
  "upload_id": "abc-123-def-456",
  "status": "failed",
  "progress": 35,
  "step": "data_validation",
  "message": "Insufficient order history: need 10+ orders, found 3",
  "created_at": "2025-12-02T10:30:00Z",
  "updated_at": "2025-12-02T10:30:15Z"
}
```

**Example (Existing Bundles - Early Return):**
```json
{
  "upload_id": "abc-123-def-456",
  "status": "completed",
  "progress": 100,
  "step": "finalization",
  "message": "Bundle recommendations already exist (10 bundles)",
  "created_at": "2025-12-02T10:30:00Z",
  "updated_at": "2025-12-02T10:30:05Z"
}
```

### **Response (404 Not Found)**
```json
{
  "detail": "No generation progress found for upload abc-123"
}
```

---

## **4. GET /api/shopify/recommendations**

### **Query Parameters**
```typescript
interface RecommendationsQuery {
  shopId: string          // Required: "shop.myshopify.com"
  approved?: boolean      // Optional: filter by approval status
  limit?: number          // Optional: max results (default 20)
}
```

**Example:**
```
GET /api/shopify/recommendations?shopId=shop.myshopify.com&limit=50
```

### **Response (200 OK)**
```typescript
interface RecommendationsResponse {
  shop_id: string
  count: number
  recommendations: Recommendation[]
}

interface Recommendation {
  id: string
  discount_type: "fbt" | "bogo" | "volume"
  discount_reference: string
  primary_sku: string
  companion_sku: string | null
  discount_percentage: number | null
  min_quantity: number | null
  confidence: number | null        // 0.0 - 1.0
  predicted_lift: number | null    // Percentage lift
  ranking_score: number | null     // 0.0 - 1.0
  is_approved: boolean
  created_at: string               // ISO 8601
}
```

**Example:**
```json
{
  "shop_id": "shop.myshopify.com",
  "count": 3,
  "recommendations": [
    {
      "id": "bundle-001",
      "discount_type": "fbt",
      "discount_reference": "__quick_start_fbt_1",
      "primary_sku": "PROD-001",
      "companion_sku": "PROD-002",
      "discount_percentage": 10.0,
      "min_quantity": null,
      "confidence": 0.85,
      "predicted_lift": 15.5,
      "ranking_score": 0.92,
      "is_approved": false,
      "created_at": "2025-12-02T10:30:45Z"
    },
    {
      "id": "bundle-002",
      "discount_type": "bogo",
      "discount_reference": "__quick_start_bogo_1",
      "primary_sku": "PROD-003",
      "companion_sku": "PROD-003",
      "discount_percentage": 50.0,
      "min_quantity": 2,
      "confidence": 0.78,
      "predicted_lift": 22.3,
      "ranking_score": 0.88,
      "is_approved": false,
      "created_at": "2025-12-02T10:30:45Z"
    },
    {
      "id": "bundle-003",
      "discount_type": "volume",
      "discount_reference": "__quick_start_volume_1",
      "primary_sku": "PROD-004",
      "companion_sku": null,
      "discount_percentage": 15.0,
      "min_quantity": 3,
      "confidence": 0.82,
      "predicted_lift": 18.7,
      "ranking_score": 0.90,
      "is_approved": false,
      "created_at": "2025-12-02T10:30:45Z"
    }
  ]
}
```

---

## **5. POST /api/shopify/recommendations/{recommendationId}/approve**

### **Response (200 OK)**
```json
{
  "id": "bundle-001",
  "is_approved": true,
  "message": "Recommendation approved successfully"
}
```

### **Response (404 Not Found)**
```json
{
  "detail": "Recommendation bundle-001 not found"
}
```

---

## **Complete TypeScript Types**

```typescript
// ============================================================================
// REQUEST TYPES
// ============================================================================

interface ShopifyUploadRequest {
  shopId: string
  csvType: "orders" | "variants" | "catalog" | "inventory"
  csvData: string
  runId?: string
  triggerPipeline: boolean
}

// ============================================================================
// RESPONSE TYPES
// ============================================================================

interface ShopifyUploadResponse {
  uploadId: string
  runId: string
  status: "processing"
  shopId: string
  csvType: string
  triggerPipeline: boolean
  message: string
}

interface UploadStatus {
  upload_id: string
  shop_id: string | null
  status: "processing" | "completed" | "failed"
  total_rows: number
  processed_rows: number
  error_message: string | null
  bundle_count: number | null
}

interface GenerationProgress {
  upload_id: string
  status: "queued" | "in_progress" | "completed" | "failed"
  progress: number  // 0-100
  step: string
  message: string
  created_at: string  // ISO 8601
  updated_at: string  // ISO 8601
}

interface RecommendationsResponse {
  shop_id: string
  count: number
  recommendations: Recommendation[]
}

interface Recommendation {
  id: string
  discount_type: "fbt" | "bogo" | "volume"
  discount_reference: string
  primary_sku: string
  companion_sku: string | null
  discount_percentage: number | null
  min_quantity: number | null
  confidence: number | null
  predicted_lift: number | null
  ranking_score: number | null
  is_approved: boolean
  created_at: string  // ISO 8601
}

interface ErrorResponse {
  detail: string
}

// ============================================================================
// DISCRIMINATED UNION FOR TYPE-SAFE STATUS HANDLING
// ============================================================================

type UploadStatusState =
  | { status: "processing"; bundle_count: null }
  | { status: "completed"; bundle_count: number }
  | { status: "completed"; bundle_count: null }  // Bundles still generating
  | { status: "failed"; error_message: string }

// ============================================================================
// HELPER TYPE GUARDS
// ============================================================================

function isProcessing(status: UploadStatus): status is UploadStatus & { status: "processing" } {
  return status.status === "processing"
}

function isCompleted(status: UploadStatus): status is UploadStatus & { status: "completed" } {
  return status.status === "completed"
}

function isFailed(status: UploadStatus): status is UploadStatus & { status: "failed" } {
  return status.status === "failed"
}

function hasBundles(status: UploadStatus): status is UploadStatus & { bundle_count: number } {
  return status.bundle_count !== null && status.bundle_count > 0
}

// ============================================================================
// USAGE EXAMPLE
// ============================================================================

async function handleUploadStatus(status: UploadStatus) {
  if (isFailed(status)) {
    // TypeScript knows status.error_message exists
    showError(status.error_message)
    return
  }

  if (isProcessing(status)) {
    // TypeScript knows status.status === "processing"
    showProgress(status.processed_rows, status.total_rows)
    return
  }

  if (isCompleted(status)) {
    if (hasBundles(status)) {
      // TypeScript knows status.bundle_count is number
      showSuccess(`${status.bundle_count} bundles ready!`)
      loadBundles(status.shop_id!)
    } else {
      // Bundles still generating
      showLoading("Generating bundles...")
    }
  }
}
```

---

## **API Client Example**

```typescript
class BundleAPIClient {
  private baseUrl: string

  constructor(baseUrl: string = "https://your-backend.run.app") {
    this.baseUrl = baseUrl
  }

  async uploadCSV(request: ShopifyUploadRequest): Promise<ShopifyUploadResponse> {
    const response = await fetch(`${this.baseUrl}/api/shopify/upload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    })

    if (!response.ok) {
      const error: ErrorResponse = await response.json()
      throw new Error(error.detail)
    }

    return response.json()
  }

  async getUploadStatus(uploadId: string): Promise<UploadStatus> {
    const response = await fetch(`${this.baseUrl}/api/shopify/status/${uploadId}`)

    if (!response.ok) {
      const error: ErrorResponse = await response.json()
      throw new Error(error.detail)
    }

    return response.json()
  }

  async getGenerationProgress(uploadId: string): Promise<GenerationProgress> {
    const response = await fetch(`${this.baseUrl}/generation-progress/${uploadId}`)

    if (!response.ok) {
      const error: ErrorResponse = await response.json()
      throw new Error(error.detail)
    }

    return response.json()
  }

  async getRecommendations(
    shopId: string,
    options?: { approved?: boolean; limit?: number }
  ): Promise<RecommendationsResponse> {
    const params = new URLSearchParams({
      shopId,
      ...(options?.approved !== undefined && { approved: String(options.approved) }),
      ...(options?.limit && { limit: String(options.limit) })
    })

    const response = await fetch(
      `${this.baseUrl}/api/shopify/recommendations?${params}`
    )

    if (!response.ok) {
      const error: ErrorResponse = await response.json()
      throw new Error(error.detail)
    }

    return response.json()
  }

  async approveRecommendation(recommendationId: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/shopify/recommendations/${recommendationId}/approve`,
      { method: "POST" }
    )

    if (!response.ok) {
      const error: ErrorResponse = await response.json()
      throw new Error(error.detail)
    }
  }

  // Convenience method: Poll until bundles ready or error
  async pollUntilBundlesReady(
    uploadId: string,
    options: {
      maxPolls?: number
      pollInterval?: number
      onProgress?: (status: UploadStatus) => void
    } = {}
  ): Promise<UploadStatus> {
    const {
      maxPolls = 120,
      pollInterval = 5000,
      onProgress
    } = options

    for (let i = 0; i < maxPolls; i++) {
      const status = await this.getUploadStatus(uploadId)
      onProgress?.(status)

      // Success
      if (status.status === "completed" && status.bundle_count > 0) {
        return status
      }

      // Failure
      if (status.status === "failed") {
        throw new Error(status.error_message || "Upload failed")
      }

      // Still processing
      await new Promise(resolve => setTimeout(resolve, pollInterval))
    }

    throw new Error("Timeout: Bundle generation took too long")
  }
}

// Usage
const client = new BundleAPIClient()

try {
  // Upload
  const { uploadId } = await client.uploadCSV({
    shopId: "shop.myshopify.com",
    csvType: "orders",
    csvData: csvContent,
    triggerPipeline: true
  })

  // Poll with progress updates
  const finalStatus = await client.pollUntilBundlesReady(uploadId, {
    onProgress: (status) => {
      if (status.status === "processing") {
        console.log(`Processing: ${status.processed_rows}/${status.total_rows}`)
      } else if (status.status === "completed" && !status.bundle_count) {
        console.log("Generating bundles...")
      }
    }
  })

  // Get bundles
  const { recommendations } = await client.getRecommendations(finalStatus.shop_id!)
  console.log(`Success! ${recommendations.length} bundles created`)

} catch (error) {
  console.error("Error:", error.message)
}
```

---

## **Error Codes Reference**

| HTTP Status | Scenario | Response |
|-------------|----------|----------|
| **200** | Success | See schemas above |
| **400** | Invalid CSV type | `{ "detail": "Invalid csvType..." }` |
| **400** | Missing required columns | `{ "detail": "Missing required: order_id" }` |
| **400** | Data validation error | `{ "detail": "Variants missing for SKUs..." }` |
| **404** | Upload not found | `{ "detail": "Upload {id} not found" }` |
| **404** | Recommendation not found | `{ "detail": "Recommendation {id} not found" }` |
| **422** | Validation error | `{ "detail": [{"loc": [...], "msg": "..."}] }` |
| **500** | Internal server error | `{ "detail": "Internal server error" }` |

---

## **WebSocket Alternative (Future)**

If you want real-time updates instead of polling:

```typescript
// Future enhancement - not implemented yet
const ws = new WebSocket(`wss://your-backend.run.app/ws/upload/${uploadId}`)

ws.onmessage = (event) => {
  const update = JSON.parse(event.data)

  switch (update.type) {
    case "csv_progress":
      showProgress(update.processed_rows, update.total_rows)
      break
    case "csv_completed":
      showMessage("CSV processing complete")
      break
    case "bundles_generating":
      showMessage("Generating bundles...")
      break
    case "bundles_ready":
      showSuccess(`${update.bundle_count} bundles ready!`)
      break
    case "error":
      showError(update.error_message)
      break
  }
}
```

This would eliminate polling, but requires WebSocket support in Cloud Run.

---

## **Summary: Key Points for Frontend**

1. **Upload returns immediately** with `uploadId`
2. **Poll `/status/{uploadId}`** every 5 seconds
3. **Check `bundle_count > 0`** to know when bundles are ready
4. **Handle both** `status: "completed"` with and without `bundle_count`
5. **Show different UI** for CSV processing vs bundle generation
6. **Allow 5-10 minutes** total timeout
7. **Use TypeScript types** for type safety
8. **Handle errors gracefully** with user-friendly messages
9. **Stop polling** when `status === "completed" && bundle_count > 0`
10. **Fetch bundles** from `/recommendations` when ready
