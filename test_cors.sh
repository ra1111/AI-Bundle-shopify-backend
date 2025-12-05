#!/bin/bash
# Test CORS configuration for bundle-api

BACKEND_URL="https://bundle-api-250755735924.us-central1.run.app"
FRONTEND_ORIGIN="https://pest-template-nov-not.trycloudflare.com"

echo "🧪 Testing CORS Configuration"
echo "================================"
echo ""

echo "1️⃣ Testing OPTIONS (preflight) for /api/shopify/status/..."
curl -si -X OPTIONS "${BACKEND_URL}/api/shopify/status/test" \
  -H "Origin: ${FRONTEND_ORIGIN}" \
  -H "Access-Control-Request-Method: GET" \
  | grep -E "(HTTP/|access-control)"

echo ""
echo ""

echo "2️⃣ Testing OPTIONS (preflight) for /api/generation-progress/..."
curl -si -X OPTIONS "${BACKEND_URL}/api/generation-progress/test" \
  -H "Origin: ${FRONTEND_ORIGIN}" \
  -H "Access-Control-Request-Method: GET" \
  | grep -E "(HTTP/|access-control)"

echo ""
echo ""

echo "3️⃣ Testing actual GET request with CORS headers"
curl -si "${BACKEND_URL}/api/shopify/status/test" \
  -H "Origin: ${FRONTEND_ORIGIN}" \
  | grep -E "(HTTP/|access-control)"

echo ""
echo ""
echo "✅ If you see HTTP/2 200 and access-control-allow-origin headers, CORS is working!"
