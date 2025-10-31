# Complete Input Data Guide - AI Bundle Creator

## 🌐 Cloud Endpoint
**Production API Base URL:** `https://bundle-api-250755735924.us-central1.run.app`

---

## Overview

The AI Bundle Creator accepts **4 types of CSV files** to generate intelligent product bundle recommendations. This guide details every possible input field, mandatory vs optional columns, and all API endpoints.

---

## 📊 CSV File Types Summary

| CSV Type | Purpose | Required | File Types Accepted |
|----------|---------|----------|---------------------|
| **Orders** | Historical transaction data | ✅ **MANDATORY** | `orders`, `orders.csv` |
| **Variants** | Product variant information | ⚠️ Recommended | `variants`, `variants.csv` |
| **Inventory Levels** | Stock quantities | ⭐ Optional | `inventory_levels`, `inventory.csv` |
| **Catalog Joined** | Complete product catalog | ⭐ Optional (but improves results) | `catalog_joined`, `catalog`, `products`, `catalog.csv` |

---

## 1️⃣ Orders CSV (MANDATORY)

Contains historical order/transaction data for machine learning analysis.

### **Mandatory Columns**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `createdAt` | `created_at`, `orderDate`, `order_date`, `date` | DateTime | Order timestamp | `2025-01-15T10:30:00Z` |
| `lineItemQuantity` | `line_item_quantity`, `quantity`, `qty` | Integer | Product quantity | `2` |
| `originalUnitPrice` | `original_unit_price`, `price`, `unit_price` | Decimal | Product price | `29.99` |

### **Mandatory (At Least One)**

Must have **at least ONE** from each group:

**Group 1 - Order Identifier:**
- `order_id` OR `orderId` OR `name`

**Group 2 - Product Identifier:**
- `variantId` OR `variant_id` OR `sku`

### **Optional Columns (Improve Recommendations)**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `order_id` | `orderId`, `name` | String | Unique order ID | `#1001`, `ORD-2025-001` |
| `customer_id` | `customerId` | String | Customer identifier | `CUST-123` |
| `customer_email` | `customerEmail` | Email | Customer email | `user@example.com` |
| `customer_country` | `shippingCountryCode`, `country` | String | Country code | `US`, `IN`, `GB` |
| `customer_currency` | `currencyCode`, `currency` | String | Currency code | `USD`, `EUR`, `INR` |
| `clv_band` | `clvBand` | String | Customer lifetime value | `high`, `medium`, `low` |
| `channel` | `salesChannel` | String | Sales channel | `online`, `pos`, `mobile` |
| `device` | `deviceType` | String | Device type | `desktop`, `mobile`, `tablet` |
| `subtotal` | `subtotalPrice` | Decimal | Order subtotal | `79.98` |
| `discount` | `discountTotal` | Decimal | Discount amount | `10.00` |
| `discount_code` | `discountCode` | String | Discount code used | `SUMMER20` |
| `shipping` | `totalShippingPrice`, `shipping_cost` | Decimal | Shipping cost | `5.99` |
| `taxes` | `totalTax`, `tax` | Decimal | Tax amount | `6.40` |
| `total` | `totalPrice`, `order_total` | Decimal | Order total | `92.37` |
| `financial_status` | `displayFinancialStatus` | String | Payment status | `paid`, `pending`, `refunded` |
| `fulfillment_status` | `displayFulfillmentStatus` | String | Fulfillment status | `fulfilled`, `unfulfilled` |
| `returned` | `is_returned` | Boolean | Was order returned? | `true`, `false` |
| `basket_item_count` | `item_count` | Integer | Total items in order | `3` |
| `basket_line_count` | `line_count` | Integer | Number of line items | `2` |
| `basket_value` | `basket_total` | Decimal | Basket value | `79.98` |
| `variant_id` | `variantId` | String | Product variant ID | `VAR-001` |
| `sku` | `SKU`, `product_sku` | String | Product SKU | `SKU-001` |
| `line_item_id` | `lineItemId` | String | Line item ID | `LINE-001` |
| `line_item_name` | `lineItemName`, `product_name` | String | Product name | `Blue T-Shirt` |
| `vendor` | `brand`, `manufacturer` | String | Brand/vendor | `Nike`, `Adidas` |
| `product_type` | `productType`, `category` | String | Product category | `Clothing`, `Footwear` |
| `product_id` | `productId` | String | Product ID | `PROD-001` |
| `product_title` | `productTitle` | String | Product title | `Premium T-Shirt` |
| `variant_title` | `variantTitle` | String | Variant name | `Blue - Small` |
| `requires_shipping` | `requiresShipping` | Boolean | Requires shipping? | `true`, `false` |
| `is_gift_card` | `isGiftCard` | Boolean | Is gift card? | `true`, `false` |
| `taxable` | `is_taxable` | Boolean | Is taxable? | `true`, `false` |
| `image_url` | `imageUrl` | String | Product image URL | `https://...` |
| `line_discount` | `lineItemTotalDiscount` | Decimal | Line item discount | `5.00` |
| `discounted_total` | `discountedTotal` | Decimal | Discounted total | `24.99` |
| `shipping_province` | `shippingProvince`, `state` | String | State/province | `CA`, `Maharashtra` |
| `shipping_city` | `shippingCity`, `city` | String | City | `Los Angeles`, `Mumbai` |
| `shop_id` | `shopId`, `shop_domain`, `store_domain` | String | Shop identifier | `my-store` |

### **Orders CSV Example**
```csv
orderId,createdAt,lineItemQuantity,originalUnitPrice,sku,lineItemName,vendor,productType,customerEmail,shippingCountryCode
#1001,2025-01-15T10:30:00Z,2,29.99,SKU-001,Blue T-Shirt,Nike,Clothing,customer@example.com,US
#1001,2025-01-15T10:30:00Z,1,49.99,SKU-002,Running Shoes,Nike,Footwear,customer@example.com,US
#1002,2025-01-16T14:20:00Z,1,19.99,SKU-003,Water Bottle,Hydro,Accessories,user@example.com,IN
#1003,2025-01-17T09:15:00Z,3,15.99,SKU-004,Sports Socks,Nike,Clothing,buyer@example.com,GB
```

---

## 2️⃣ Variants CSV (Recommended)

Product variant metadata with pricing and inventory tracking information.

### **Mandatory Columns**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `product_id` | `productId` | String | Parent product ID | `PROD-001` |
| `variant_id` | `variantId` | String | Unique variant ID | `VAR-001` |
| `variant_title` | `variantTitle`, `title` | String | Variant name | `Blue T-Shirt - Small` |
| `price` | `variant_price` | Decimal | Current price | `29.99` |
| `inventory_item_id` | `inventoryItemId` | String | Inventory tracking ID | `INV-001` |

### **Optional Columns**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `sku` | `variant_sku`, `SKU` | String | SKU code | `SKU-001` |
| `compare_at_price` | `variant_compare_at_price`, `original_price` | Decimal | Original/compare price | `39.99` |
| `inventory_item_created_at` | `inventoryItemCreatedAt` | DateTime | Inventory item created | `2024-01-01T00:00:00Z` |
| `shop_id` | `shopId`, `shop_domain` | String | Shop identifier | `my-store` |

### **Variants CSV Example**
```csv
product_id,variant_id,variant_title,price,sku,inventory_item_id,compare_at_price
PROD-001,VAR-001,Blue T-Shirt - Small,29.99,SKU-001,INV-001,39.99
PROD-001,VAR-002,Blue T-Shirt - Medium,29.99,SKU-001-M,INV-002,39.99
PROD-001,VAR-003,Blue T-Shirt - Large,31.99,SKU-001-L,INV-003,41.99
PROD-002,VAR-004,Running Shoes - Size 9,49.99,SKU-002-9,INV-004,59.99
PROD-002,VAR-005,Running Shoes - Size 10,49.99,SKU-002-10,INV-005,59.99
```

---

## 3️⃣ Inventory Levels CSV (Optional)

Current stock levels by warehouse/store location.

### **Mandatory Columns**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `inventory_item_id` | `inventoryItemId` | String | Inventory item ID | `INV-001` |
| `location_id` | `locationId` | String | Warehouse/store location | `LOC-MAIN` |
| `available` | `quantity`, `stock` | Integer | Available quantity | `150` |
| `updated_at` | `updatedAt`, `last_updated` | DateTime | Last update time | `2025-01-20T08:00:00Z` |

### **Optional Columns**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `shop_id` | `shopId`, `shop_domain` | String | Shop identifier | `my-store` |

### **Inventory Levels CSV Example**
```csv
inventory_item_id,location_id,available,updated_at
INV-001,LOC-MAIN,150,2025-01-20T08:00:00Z
INV-002,LOC-MAIN,200,2025-01-20T08:00:00Z
INV-003,LOC-MAIN,75,2025-01-20T08:00:00Z
INV-004,LOC-MAIN,50,2025-01-20T08:00:00Z
INV-004,LOC-STORE1,25,2025-01-20T08:00:00Z
INV-005,LOC-MAIN,30,2025-01-20T08:00:00Z
```

---

## 4️⃣ Catalog Joined CSV (Optional but Highly Recommended)

Complete denormalized product catalog combining product, variant, and inventory data.

### **Mandatory Columns**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `product_id` | `productId` | String | Product identifier | `PROD-001` |
| `variant_id` | `variantId` | String | Variant identifier | `VAR-001` |
| `product_title` | `productTitle`, `title`, `name` | String | Product name | `Premium Blue T-Shirt` |
| `price` | `variant_price`, `current_price` | Decimal | Current price | `29.99` |
| `inventory_item_id` | `inventoryItemId` | String | Inventory item ID | `INV-001` |
| `product_status` | `status` | String | Product status | `active`, `draft`, `archived` |

### **Optional Columns (Improve ML Quality)**

| Column | Aliases | Type | Description | Example |
|--------|---------|------|-------------|---------|
| `sku` | `variant_sku`, `SKU` | String | SKU code | `SKU-001` |
| `product_type` | `productType`, `category` | String | Product category | `Clothing` |
| `vendor` | `brand`, `manufacturer` | String | Brand/vendor | `Nike` |
| `tags` | `product_tags` | String | Comma-separated tags | `summer,cotton,casual` |
| `variant_title` | `variantTitle` | String | Variant name | `Blue - Small` |
| `compare_at_price` | `variant_compare_at_price` | Decimal | Original price | `39.99` |
| `available_total` | `total_inventory`, `stock` | Integer | Total available stock | `150` |
| `product_created_at` | `productCreatedAt`, `created_at` | DateTime | Product created | `2024-01-01T00:00:00Z` |
| `product_published_at` | `productPublishedAt`, `published_at` | DateTime | Published date | `2024-01-15T00:00:00Z` |
| `inventory_item_created_at` | `inventoryItemCreatedAt` | DateTime | Inventory created | `2024-01-01T00:00:00Z` |
| `last_inventory_update` | `lastInventoryUpdate` | DateTime | Last inventory update | `2025-01-20T08:00:00Z` |
| `shop_id` | `shopId`, `shop_domain` | String | Shop identifier | `my-store` |

### **System-Computed Flags (Automatic)**

These are automatically calculated based on your data:

| Flag | Description | Calculation Logic |
|------|-------------|-------------------|
| `is_slow_mover` | Low sales velocity | Based on order frequency |
| `is_new_launch` | Recently launched | `product_created_at` < 30 days |
| `is_seasonal` | Seasonal product | Based on tags and patterns |
| `is_high_margin` | High profit margin | `compare_at_price` - `price` > threshold |

### **Catalog Joined CSV Example**
```csv
product_id,variant_id,product_title,price,sku,inventory_item_id,product_status,product_type,vendor,tags,available_total,compare_at_price,product_created_at
PROD-001,VAR-001,Premium Blue T-Shirt,29.99,SKU-001,INV-001,active,Clothing,Nike,"summer,cotton,casual",150,39.99,2024-01-01T00:00:00Z
PROD-002,VAR-004,Professional Running Shoes,49.99,SKU-002-9,INV-004,active,Footwear,Nike,"athletic,running,sports",75,59.99,2024-02-15T00:00:00Z
PROD-003,VAR-006,Stainless Water Bottle,19.99,SKU-003,INV-006,active,Accessories,Hydro,"fitness,eco-friendly",200,24.99,2024-03-01T00:00:00Z
```

---

## 🏪 Shop ID (Multi-Tenancy Support)

### **How Shop ID Works**

The system supports multi-tenant architecture where data can be scoped to specific shops/stores.

### **Shop ID Priority (Resolution Order)**

1. **Explicit API Parameter** - Highest priority
   ```bash
   -F "shopId=my-store"
   ```

2. **CSV Column Data** - Detected from CSV
   - Recognized columns: `shop_id`, `shopId`, `shop_domain`, `store_domain`, `domain`

3. **Environment Variable** - Server default
   ```bash
   DEFAULT_SHOP_ID=my-store
   ```

4. **Hard-coded Fallback** - System default
   - Default: `"demo-shop"`

### **When to Use Shop ID**

✅ **Use Shop ID for:**
- Multiple stores/shops in your system
- Multi-tenant SaaS applications
- Data isolation between clients
- Regional store separation

❌ **Not Needed for:**
- Single store operations
- Simple testing/development
- Default "demo-shop" is acceptable

### **How to Provide Shop ID**

**Method 1: In CSV Column**
```csv
shop_id,orderId,createdAt,quantity,price,sku
my-store.myshopify.com,#1001,2025-01-15,2,29.99,SKU-001
my-store.myshopify.com,#1002,2025-01-16,1,49.99,SKU-002
```

**Method 2: API Parameter**
```bash
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@orders.csv" \
  -F "csvType=orders" \
  -F "shopId=my-store.myshopify.com"
```

**Method 3: Environment Variable**
```bash
# In .env or server configuration
DEFAULT_SHOP_ID=my-store.myshopify.com
```

---

## 📤 API Endpoints & Usage

### **Base URL**
```
https://bundle-api-250755735924.us-central1.run.app
```

### **1. Upload CSV Files**

**Endpoint:** `POST /api/upload-csv`

**Parameters:**
- `file` (file) - CSV file to upload
- `csvType` (string, optional) - Type: `orders`, `variants`, `inventory_levels`, `catalog_joined`, `catalog`, `products`
  - If not provided, system auto-detects
- `shopId` (string, optional) - Shop identifier
- `runId` (string, optional) - For uploading multiple related files

**Example - Upload Orders:**
```bash
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@orders.csv" \
  -F "csvType=orders" \
  -F "shopId=my-store"
```

**Response:**
```json
{
  "uploadId": "550e8400-e29b-41d4-a716-446655440000",
  "runId": "run-2025-01-20-abc123",
  "shopId": "my-store",
  "filename": "orders.csv",
  "csvType": "orders",
  "status": "processing",
  "totalRows": 1523,
  "message": "CSV upload started"
}
```

**Example - Upload Multiple Files (Same Run):**
```bash
# 1. Upload orders (generates runId)
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@orders.csv" \
  -F "csvType=orders" \
  -F "shopId=my-store"
# Save runId from response: "run-2025-01-20-abc123"

# 2. Upload variants (same runId)
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@variants.csv" \
  -F "csvType=variants" \
  -F "runId=run-2025-01-20-abc123"

# 3. Upload inventory
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@inventory.csv" \
  -F "csvType=inventory_levels" \
  -F "runId=run-2025-01-20-abc123"

# 4. Upload catalog
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@catalog.csv" \
  -F "csvType=catalog_joined" \
  -F "runId=run-2025-01-20-abc123"
```

### **2. Check Upload Status**

**Endpoint:** `GET /api/upload-status/{uploadId}`

**Example:**
```bash
curl https://bundle-api-250755735924.us-central1.run.app/api/upload-status/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "uploadId": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "totalRows": 1523,
  "processedRows": 1523,
  "csvType": "orders",
  "shopId": "my-store",
  "createdAt": "2025-01-20T10:30:00Z"
}
```

### **3. List Recent Uploads**

**Endpoint:** `GET /api/uploads`

**Query Parameters:**
- `shopId` (string, optional) - Filter by shop
- `limit` (integer, optional) - Max results (default: 10)
- `offset` (integer, optional) - Pagination offset

**Example:**
```bash
curl "https://bundle-api-250755735924.us-central1.run.app/api/uploads?shopId=my-store&limit=5"
```

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "runId": "run-2025-01-20-abc123",
    "shopId": "my-store",
    "filename": "orders.csv",
    "csvType": "orders",
    "status": "completed",
    "totalRows": 1523,
    "createdAt": "2025-01-20T10:30:00Z"
  }
]
```

### **4. Generate Association Rules**

**Endpoint:** `POST /api/generate-rules`

**Body:**
```json
{
  "uploadId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example:**
```bash
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/generate-rules \
  -H "Content-Type: application/json" \
  -d '{"uploadId": "550e8400-e29b-41d4-a716-446655440000"}'
```

**Response:**
```json
{
  "message": "Association rules generation started",
  "uploadId": "550e8400-e29b-41d4-a716-446655440000",
  "totalRules": 156
}
```

### **5. Generate Bundle Recommendations**

**Endpoint:** `POST /api/generate-bundles`

**Body:**
```json
{
  "uploadId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example:**
```bash
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/generate-bundles \
  -H "Content-Type: application/json" \
  -d '{"uploadId": "550e8400-e29b-41d4-a716-446655440000"}'
```

**Response:**
```json
{
  "message": "Bundle generation started in background",
  "uploadId": "550e8400-e29b-41d4-a716-446655440000",
  "shopId": "my-store"
}
```

### **6. Get Bundle Recommendations**

**Endpoint:** `GET /api/bundle-recommendations`

**Query Parameters:**
- `shopId` (string, optional) - Filter by shop
- `uploadId` (string, optional) - Filter by upload
- `limit` (integer, optional) - Max results (default: 50)

**Example:**
```bash
curl "https://bundle-api-250755735924.us-central1.run.app/api/bundle-recommendations?shopId=my-store&limit=10"
```

**Response:**
```json
[
  {
    "id": "rec-001",
    "csvUploadId": "550e8400-e29b-41d4-a716-446655440000",
    "shopId": "my-store",
    "bundleType": "FBT",
    "objective": "increase_aov",
    "products": [
      {"sku": "SKU-001", "name": "Blue T-Shirt", "price": 29.99},
      {"sku": "SKU-002", "name": "Running Shoes", "price": 49.99}
    ],
    "pricing": {
      "original_total": 79.98,
      "bundle_price": 67.98,
      "discount_amount": 12.00,
      "discount_percentage": 15.0
    },
    "aiCopy": {
      "title": "Active Lifestyle Bundle",
      "description": "Complete your workout wardrobe with this perfect combination",
      "urgency": "Limited time offer - Save 15%!",
      "value_proposition": "Save $12 when you buy together"
    },
    "confidence": 0.87,
    "predictedLift": 1.45,
    "rankingScore": 0.92,
    "rankPosition": 1,
    "isApproved": false,
    "createdAt": "2025-01-20T11:00:00Z"
  }
]
```

### **7. Approve/Reject Recommendation**

**Endpoint:** `PATCH /api/bundle-recommendations/{id}/approve`

**Body:**
```json
{
  "isApproved": true
}
```

**Example:**
```bash
curl -X PATCH https://bundle-api-250755735924.us-central1.run.app/api/bundle-recommendations/rec-001/approve \
  -H "Content-Type: application/json" \
  -d '{"isApproved": true}'
```

### **8. Shopify Integration (Simplified Upload)**

**Endpoint:** `POST /api/shopify/upload`

**Body:**
```json
{
  "shopId": "my-store.myshopify.com",
  "orders": [
    {
      "orderId": "#1001",
      "createdAt": "2025-01-15T10:30:00Z",
      "lineItemQuantity": 2,
      "originalUnitPrice": 29.99,
      "sku": "SKU-001",
      "productName": "Blue T-Shirt"
    }
  ],
  "variants": [...],
  "inventory": [...],
  "catalog": [...],
  "triggerPipeline": true
}
```

**Example:**
```bash
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/shopify/upload \
  -H "Content-Type: application/json" \
  -d @shopify_data.json
```

### **9. Dashboard Statistics**

**Endpoint:** `GET /api/dashboard-stats`

**Query Parameters:**
- `shopId` (string, optional) - Filter by shop

**Example:**
```bash
curl "https://bundle-api-250755735924.us-central1.run.app/api/dashboard-stats?shopId=my-store"
```

**Response:**
```json
{
  "activeBundles": 12,
  "bundleRevenue": 15420.50,
  "avgBundleSize": 2.3,
  "conversionRate": 12.5,
  "totalRecommendations": 45,
  "approvedRecommendations": 12,
  "totalOrders": 1523,
  "totalProducts": 234
}
```

### **10. Export Recommendations**

**Endpoint:** `GET /api/export/recommendations`

**Query Parameters:**
- `shopId` (string, optional) - Filter by shop
- `uploadId` (string, optional) - Filter by upload
- `format` (string, optional) - Format: `json` or `csv` (default: json)

**Example - JSON:**
```bash
curl "https://bundle-api-250755735924.us-central1.run.app/api/export/recommendations?shopId=my-store&format=json" > recommendations.json
```

**Example - CSV:**
```bash
curl "https://bundle-api-250755735924.us-central1.run.app/api/export/recommendations?shopId=my-store&format=csv" > recommendations.csv
```

### **11. Health Check**

**Endpoint:** `GET /api/health`

**Example:**
```bash
curl https://bundle-api-250755735924.us-central1.run.app/api/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

---

## 🔄 Complete Workflow Examples

### **Workflow 1: Simple Single File Upload**

```bash
# Step 1: Upload orders CSV
UPLOAD_RESPONSE=$(curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@orders.csv" \
  -F "csvType=orders" \
  -F "shopId=my-store")

# Extract uploadId from response
UPLOAD_ID=$(echo $UPLOAD_RESPONSE | jq -r '.uploadId')

# Step 2: Wait for processing (check status)
curl https://bundle-api-250755735924.us-central1.run.app/api/upload-status/$UPLOAD_ID

# Step 3: Generate bundles
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/generate-bundles \
  -H "Content-Type: application/json" \
  -d "{\"uploadId\": \"$UPLOAD_ID\"}"

# Step 4: Wait ~2-5 minutes for generation

# Step 5: Get recommendations
curl "https://bundle-api-250755735924.us-central1.run.app/api/bundle-recommendations?shopId=my-store"
```

### **Workflow 2: Multi-File Upload (Best Quality)**

```bash
# Step 1: Upload orders (get runId)
RESPONSE=$(curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@orders.csv" \
  -F "csvType=orders" \
  -F "shopId=my-store")

RUN_ID=$(echo $RESPONSE | jq -r '.runId')
UPLOAD_ID=$(echo $RESPONSE | jq -r '.uploadId')

# Step 2: Upload variants (same runId)
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@variants.csv" \
  -F "csvType=variants" \
  -F "runId=$RUN_ID"

# Step 3: Upload inventory (same runId)
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@inventory.csv" \
  -F "csvType=inventory_levels" \
  -F "runId=$RUN_ID"

# Step 4: Upload catalog (same runId)
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/upload-csv \
  -F "file=@catalog.csv" \
  -F "csvType=catalog_joined" \
  -F "runId=$RUN_ID"

# Step 5: Generate bundles
curl -X POST https://bundle-api-250755735924.us-central1.run.app/api/generate-bundles \
  -H "Content-Type: application/json" \
  -d "{\"uploadId\": \"$UPLOAD_ID\"}"

# Step 6: Monitor and retrieve
curl "https://bundle-api-250755735924.us-central1.run.app/api/bundle-recommendations?shopId=my-store"
```

---

## 📋 CSV Format Requirements

### **General Rules**

| Requirement | Description |
|-------------|-------------|
| **Encoding** | UTF-8 (recommended) |
| **Delimiter** | Comma (`,`) |
| **Header Row** | Required - first row must be column names |
| **File Size** | Maximum 50MB per file |
| **Date Format** | ISO 8601 preferred: `2025-01-15T10:30:00Z` or `2025-01-15` |
| **Decimals** | Use period (`.`) as decimal separator: `29.99` not `29,99` |
| **Boolean** | `true`/`false`, `1`/`0`, `yes`/`no`, `TRUE`/`FALSE` |
| **Empty Values** | Empty strings or `NULL` for missing data |

### **Column Name Flexibility**

The system handles various naming conventions:

- ✅ **Case-insensitive**: `orderId`, `OrderID`, `orderid` all work
- ✅ **Snake_case or camelCase**: `order_id` = `orderId`
- ✅ **Aliases**: `createdAt` = `created_at` = `orderDate` = `date`
- ✅ **Spaces in names**: `order id` = `order_id`

---

## 🎯 Minimum Viable Input

**Absolute minimum to get started:**

1. **Orders CSV** with these 3 mandatory columns + 1 identifier:
   - `createdAt`
   - `lineItemQuantity`
   - `originalUnitPrice`
   - `sku` OR `variantId` OR `productName`

**Example Minimal CSV:**
```csv
createdAt,lineItemQuantity,originalUnitPrice,sku
2025-01-15,2,29.99,SKU-001
2025-01-15,1,49.99,SKU-002
2025-01-16,1,19.99,SKU-003
2025-01-16,1,29.99,SKU-001
2025-01-17,2,15.99,SKU-004
```

---

## 📊 Sample Data Generation

### **Python Script to Generate Test Orders**

```python
import csv
from datetime import datetime, timedelta
import random

products = [
    ("SKU-001", "Blue T-Shirt", 29.99, "Nike", "Clothing"),
    ("SKU-002", "Running Shoes", 49.99, "Nike", "Footwear"),
    ("SKU-003", "Water Bottle", 19.99, "Hydro", "Accessories"),
    ("SKU-004", "Sports Socks", 15.99, "Nike", "Clothing"),
    ("SKU-005", "Gym Bag", 39.99, "Adidas", "Accessories"),
    ("SKU-006", "Yoga Mat", 34.99, "Lululemon", "Fitness"),
    ("SKU-007", "Protein Shaker", 12.99, "Optimum", "Accessories"),
    ("SKU-008", "Training Shorts", 24.99, "Adidas", "Clothing"),
]

with open('sample_orders.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'orderId', 'createdAt', 'lineItemQuantity',
        'originalUnitPrice', 'sku', 'lineItemName',
        'vendor', 'productType', 'customerEmail', 'shippingCountryCode'
    ])

    for i in range(200):  # Generate 200 orders
        order_id = f"#100{i+1}"
        date = datetime.now() - timedelta(days=random.randint(0, 90))
        email = f"customer{i % 50}@example.com"
        country = random.choice(['US', 'IN', 'GB', 'CA', 'AU'])
        items = random.randint(1, 4)

        for _ in range(items):
            sku, name, price, vendor, ptype = random.choice(products)
            qty = random.randint(1, 3)
            writer.writerow([
                order_id, date.isoformat(), qty,
                price, sku, name, vendor, ptype, email, country
            ])

print("✅ Sample data created: sample_orders.csv")
print(f"   Generated 200 orders with multiple line items")
```

---

## ❓ Frequently Asked Questions

### **Q: Do I need all 4 CSV types?**
**A:** No. Only **Orders CSV is mandatory**. Other CSVs improve recommendation quality and enable advanced features.

### **Q: What if my column names don't match exactly?**
**A:** The system is flexible and handles many variations. Common aliases are supported (see column tables above).

### **Q: Can I upload CSVs in any order?**
**A:** Yes. The system handles files in any order when using the same `runId`.

### **Q: How do I know if my CSV was processed correctly?**
**A:** Check upload status: `GET /api/upload-status/{uploadId}`. Status will be `completed` if successful.

### **Q: What happens if upload fails?**
**A:** The API returns error details in the response. Check `error_message` field for specific issues.

### **Q: How many orders do I need for good results?**
**A:**
- **Minimum**: 50-100 orders (basic recommendations)
- **Good**: 500-1000 orders (quality recommendations)
- **Best**: 2000+ orders (high-quality ML results)

### **Q: Can I update/replace data?**
**A:** Yes. Upload new CSV with same `shopId`. Old data remains, new analysis uses latest upload.

### **Q: How long does bundle generation take?**
**A:** Typically 2-10 minutes depending on data size:
- <500 orders: ~2 minutes
- 500-2000 orders: ~5 minutes
- 2000+ orders: ~10 minutes

### **Q: Can I test with sample data?**
**A:** Yes! Use the Python script above or create minimal CSV with 50-100 sample orders.

### **Q: What if I have Shopify data?**
**A:** Export from Shopify Admin → Products → Export, then upload to `/api/upload-csv` OR use `/api/shopify/upload` endpoint.

### **Q: Do bundle recommendations expire?**
**A:** No. Recommendations persist until you generate new ones or delete them.

### **Q: Can I filter recommendations by specific criteria?**
**A:** Yes. Use query parameters: `?shopId=...&uploadId=...&bundleType=FBT`

---

## 🚀 Quick Start Checklist

- [ ] Prepare Orders CSV (mandatory)
- [ ] Optionally prepare Variants, Inventory, Catalog CSVs
- [ ] Determine if you need multi-shop support (set shopId)
- [ ] Upload CSV file(s) to `/api/upload-csv`
- [ ] Check upload status: `/api/upload-status/{uploadId}`
- [ ] Generate bundles: `POST /api/generate-bundles`
- [ ] Wait 2-10 minutes for processing
- [ ] Retrieve recommendations: `GET /api/bundle-recommendations`
- [ ] Approve good recommendations: `PATCH /api/bundle-recommendations/{id}/approve`
- [ ] Export for use: `GET /api/export/recommendations`

---

## 📚 Related Documentation

- **[DEPLOYMENT_SUCCESS.md](DEPLOYMENT_SUCCESS.md)** - Server deployment details
- **[COCKROACHDB_MIGRATION.md](COCKROACHDB_MIGRATION.md)** - Database architecture
- **API Documentation**: https://bundle-api-250755735924.us-central1.run.app/api/docs

---

**Questions?** Check the API docs or review error messages in upload status responses.
