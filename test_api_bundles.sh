#!/bin/bash

# Test script to check what the API actually returns

BACKEND_URL="https://bundle-api-250755735924.us-central1.run.app"

echo "========================================="
echo "Testing Bundle Recommendations API"
echo "========================================="
echo ""

# Ask for shop ID
read -p "Enter shop_id to test (e.g., shop.myshopify.com): " SHOP_ID

if [ -z "$SHOP_ID" ]; then
    echo "Error: shop_id required"
    exit 1
fi

echo ""
echo "1. Testing ALL bundles (no filter):"
echo "   GET /api/shopify/recommendations/${SHOP_ID}?limit=10"
echo ""
curl -s "${BACKEND_URL}/api/shopify/recommendations/${SHOP_ID}?limit=10" | jq '.' | head -100

echo ""
echo ""
echo "2. Testing APPROVED bundles only:"
echo "   GET /api/shopify/recommendations/${SHOP_ID}?approved=true&limit=10"
echo ""
curl -s "${BACKEND_URL}/api/shopify/recommendations/${SHOP_ID}?approved=true&limit=10" | jq '.' | head -100

echo ""
echo ""
echo "3. Checking bundle_type field:"
echo "   Extracting just bundle_type from first 5 bundles:"
echo ""
curl -s "${BACKEND_URL}/api/shopify/recommendations/${SHOP_ID}?approved=true&limit=5" | jq '.recommendations[] | {id, bundle_type, is_approved}'

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="
