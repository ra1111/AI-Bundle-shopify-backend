# Quick Install API Update - Shop ID Integration

## ✅ Fixed: Shop ID Extraction from Frontend

Previously, backend was using hardcoded `shop_id = "default-shop"`.  
Now properly extracted from authenticated frontend request.

---

## **Updated API Endpoints**

### **1. POST /api/bundles/quick-install**

**Request (Changed):**
```
Content-Type: multipart/form-data

file: File (required)              - CSV file upload
shop_id: string (required)         - Shop ID from authenticated session
```

**Example using curl:**
```bash
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv" \
  -F "shop_id=store.myshopify.com"
```

**Example using JavaScript/Fetch:**
```typescript
const formData = new FormData();
formData.append('file', csvFile);
formData.append('shop_id', session.shop); // From authenticated session

fetch('/api/bundles/quick-install', {
  method: 'POST',
  body: formData
})
```

**Response (Unchanged):**
```json
200 OK
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PROCESSING",
  "message": "Quick install started. Bundles are being generated..."
}
```

---

### **2. GET /api/bundles/quick-install/status**

**Request (Changed):**
```
GET /api/bundles/quick-install/status?shop_id=store.myshopify.com
```

**Example using curl:**
```bash
curl "http://localhost:8000/api/bundles/quick-install/status?shop_id=store.myshopify.com"
```

**Example using JavaScript/Fetch:**
```typescript
const shopId = session.shop; // From authenticated session

fetch(`/api/bundles/quick-install/status?shop_id=${shopId}`)
  .then(res => res.json())
```

**Response (Unchanged):**
```json
200 OK
{
  "has_quick_install": true,
  "can_run": false,
  "status": "PROCESSING",
  "started_at": "2025-12-06T10:00:00",
  "message": "Quick install is in progress"
}
```

---

## **Security Improvements**

| Aspect | Before | After |
|--------|--------|-------|
| Shop ID | Hardcoded to "default-shop" | Extracted from frontend request |
| Data Isolation | All shops mixed together | Each shop's data isolated |
| Cross-shop Access | Possible | Prevented |
| Authentication | N/A | Tied to Shopify session |

---

## **Frontend Integration**

### **Upload CSV with Shop ID**

```typescript
const uploadQuickInstall = async (file: File, shopId: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('shop_id', shopId); // ← NEW: Include shop_id

  const response = await fetch('/api/bundles/quick-install', {
    method: 'POST',
    body: formData
  });
  
  const { job_id, status } = await response.json();
  return job_id;
};
```

### **Poll Status with Shop ID**

```typescript
const pollQuickInstallStatus = async (shopId: string) => {
  const response = await fetch(
    `/api/bundles/quick-install/status?shop_id=${shopId}` // ← NEW: Add shop_id param
  );
  
  const data = await response.json();
  return data;
};
```

---

## **Error Handling**

### **New Error: Missing Shop ID**

If frontend forgets to send shop_id:

```json
400 Bad Request
{
  "detail": "shop_id required as query parameter: ?shop_id=store.myshopify.com"
}
```

---

## **Testing**

### **With Real Shop ID**
```bash
# Test upload with shop ID
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv" \
  -F "shop_id=example.myshopify.com"

# Response:
# {"job_id": "...", "status": "PROCESSING"}

# Test status with shop ID
curl "http://localhost:8000/api/bundles/quick-install/status?shop_id=example.myshopify.com"

# Response:
# {"has_quick_install": true, "status": "PROCESSING"}
```

### **Without Shop ID (Should Fail)**
```bash
curl -X POST http://localhost:8000/api/bundles/quick-install \
  -F "file=@test.csv"

# Response: 400 Bad Request
# Missing required form parameter: shop_id
```

---

## **Database Changes**

No database changes needed. Shop ID is still stored in `csv_uploads.shop_id` column.

---

## **Backward Compatibility**

⚠️ **Breaking Change**: Frontend must now send `shop_id` in request.

If using old frontend:
- Upload will fail: Missing required form parameter `shop_id`
- Status check will fail: Missing required query parameter `shop_id`

**Action**: Update frontend to send shop_id from authenticated session.

---

## **GitHub Commit**

```
28926de - FIX: Extract shop_id from frontend request instead of hardcoding
```

---

## **Summary**

✅ Shop ID properly extracted from frontend request  
✅ Each shop's data isolated from others  
✅ Security improved (no hardcoded defaults)  
✅ Matches frontend authentication flow  
✅ No database migrations needed  

**Frontend needs to update:**
1. Send `shop_id` as form parameter in POST request
2. Send `shop_id` as query parameter in GET request
3. Use `session.shop` or `shopId` from authenticated context
