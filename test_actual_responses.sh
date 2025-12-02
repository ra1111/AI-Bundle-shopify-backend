#!/bin/bash

# Test script to verify what your backend actually returns
# Run this AFTER uploading a CSV to see real responses

set -e

BACKEND_URL="https://bundle-api-250755735924.us-central1.run.app"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}BACKEND API RESPONSE TESTER${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check if upload ID provided
if [ -z "$1" ]; then
    echo -e "${RED}ERROR: Upload ID required${NC}"
    echo ""
    echo "Usage: ./test_actual_responses.sh <upload_id>"
    echo ""
    echo "Example:"
    echo "  ./test_actual_responses.sh abc-123-def-456"
    echo ""
    exit 1
fi

UPLOAD_ID="$1"

echo -e "${GREEN}Testing with Upload ID: ${UPLOAD_ID}${NC}"
echo ""

# Test Endpoint 1: /api/upload-status
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Endpoint 1: /api/upload-status/${UPLOAD_ID}${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

RESPONSE_1=$(curl -s "${BACKEND_URL}/api/upload-status/${UPLOAD_ID}")
HTTP_CODE_1=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/upload-status/${UPLOAD_ID}")

if [ "$HTTP_CODE_1" == "200" ]; then
    echo -e "${GREEN}✓ HTTP 200 OK${NC}"
    echo ""
    echo "Response:"
    echo "$RESPONSE_1" | jq '.'
    echo ""

    # Extract key fields
    STATUS=$(echo "$RESPONSE_1" | jq -r '.status')
    CSV_TYPE=$(echo "$RESPONSE_1" | jq -r '.csvType')
    HAS_BUNDLE_COUNT=$(echo "$RESPONSE_1" | jq 'has("bundle_count")')

    echo "Key fields:"
    echo "  - status: ${STATUS}"
    echo "  - csvType: ${CSV_TYPE}"
    echo "  - Has bundle_count field: ${HAS_BUNDLE_COUNT}"

elif [ "$HTTP_CODE_1" == "404" ]; then
    echo -e "${RED}✗ HTTP 404 Not Found${NC}"
    echo ""
    echo "$RESPONSE_1" | jq '.'
else
    echo -e "${RED}✗ HTTP ${HTTP_CODE_1}${NC}"
    echo ""
    echo "$RESPONSE_1"
fi

echo ""
echo ""

# Test Endpoint 2: /api/shopify/status
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Endpoint 2: /api/shopify/status/${UPLOAD_ID}${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

RESPONSE_2=$(curl -s "${BACKEND_URL}/api/shopify/status/${UPLOAD_ID}")
HTTP_CODE_2=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/shopify/status/${UPLOAD_ID}")

if [ "$HTTP_CODE_2" == "200" ]; then
    echo -e "${GREEN}✓ HTTP 200 OK${NC}"
    echo ""
    echo "Response:"
    echo "$RESPONSE_2" | jq '.'
    echo ""

    # Extract key fields
    STATUS_2=$(echo "$RESPONSE_2" | jq -r '.status')
    BUNDLE_COUNT=$(echo "$RESPONSE_2" | jq -r '.bundle_count')
    HAS_BUNDLE_COUNT_2=$(echo "$RESPONSE_2" | jq 'has("bundle_count")')

    echo "Key fields:"
    echo "  - status: ${STATUS_2}"
    echo "  - bundle_count: ${BUNDLE_COUNT}"
    echo "  - Has bundle_count field: ${HAS_BUNDLE_COUNT_2}"

elif [ "$HTTP_CODE_2" == "404" ]; then
    echo -e "${RED}✗ HTTP 404 Not Found${NC}"
    echo ""
    echo "$RESPONSE_2" | jq '.'
else
    echo -e "${RED}✗ HTTP ${HTTP_CODE_2}${NC}"
    echo ""
    echo "$RESPONSE_2"
fi

echo ""
echo ""

# Summary
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}SUMMARY${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$HTTP_CODE_1" == "200" ] && [ "$HTTP_CODE_2" == "200" ]; then
    echo -e "${GREEN}✓ Both endpoints working${NC}"
    echo ""
    echo "Which endpoint should your frontend use?"
    echo ""

    if [ "$HAS_BUNDLE_COUNT" == "true" ]; then
        echo -e "  ${GREEN}→ Use /api/upload-status/${NC} if you need bundle_count"
    else
        echo -e "  ${YELLOW}→ /api/upload-status/ does NOT include bundle_count${NC}"
    fi

    if [ "$HAS_BUNDLE_COUNT_2" == "true" ]; then
        echo -e "  ${GREEN}→ Use /api/shopify/status/ if you need bundle_count${NC}"
    else
        echo -e "  ${YELLOW}→ /api/shopify/status/ does NOT include bundle_count${NC}"
    fi

elif [ "$HTTP_CODE_1" == "200" ]; then
    echo -e "${GREEN}✓ /api/upload-status/ working${NC}"
    echo -e "${RED}✗ /api/shopify/status/ not found${NC}"
    echo ""
    echo "→ Use /api/upload-status/ endpoint"

elif [ "$HTTP_CODE_2" == "200" ]; then
    echo -e "${RED}✗ /api/upload-status/ not found${NC}"
    echo -e "${GREEN}✓ /api/shopify/status/ working${NC}"
    echo ""
    echo "→ Use /api/shopify/status/ endpoint"

else
    echo -e "${RED}✗ Both endpoints returned errors${NC}"
    echo ""
    echo "Possible issues:"
    echo "  - Invalid upload ID"
    echo "  - Upload not yet created"
    echo "  - Backend not deployed"
fi

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
