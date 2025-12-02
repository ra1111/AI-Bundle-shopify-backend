#!/bin/bash

# Test the status endpoint after uploading a CSV
# Replace UPLOAD_ID with the actual uploadId from the upload response

UPLOAD_ID="your-upload-id-here"
BACKEND_URL="https://your-cloud-run-url.run.app"

echo "Testing status endpoint..."
curl -X GET "${BACKEND_URL}/api/shopify/status/${UPLOAD_ID}" \
  -H "Content-Type: application/json" \
  | jq '.'

# Expected response:
# {
#   "upload_id": "abc-123",
#   "shop_id": "shop.myshopify.com",
#   "status": "completed",
#   "total_rows": 100,
#   "processed_rows": 100,
#   "error_message": null,
#   "bundle_count": 10
# }
