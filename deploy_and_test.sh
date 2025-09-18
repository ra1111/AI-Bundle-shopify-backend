#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v4}"
PROJECT="foresight-club"
REGION="us-central1"
REPO="bundle-api-repo"
IMAGE="bundle-api"
SERVICE="bundle-api"
SQL_INSTANCE="${PROJECT}:${REGION}:bundle-db"
SECRET="DATABASE_URL"

echo "Building image ${TAG}..."
gcloud builds submit \
  --project "${PROJECT}" \
  --tag "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}:${TAG}"

echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --image "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}:${TAG}" \
  --allow-unauthenticated \
  --update-secrets "${SECRET}=${SECRET}:latest" \
  --add-cloudsql-instances "${SQL_INSTANCE}" \
  --set-env-vars "INIT_DB_ON_STARTUP=true"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --format='value(status.url)')"
echo "Service URL: ${SERVICE_URL}"

curl -sv "${SERVICE_URL}/" -o /tmp/root.json
curl -sv "${SERVICE_URL}/api/health" -o /tmp/health.json

post_csv() {
  local type="$1" file="$2"
  echo "Uploading ${type}..."
  resp=$(curl -s -X POST "${SERVICE_URL}/api/upload-csv" -H "Expect:" \
    -F "file=@${file}" -F "csvType=${type}")
  echo "${resp}"
  echo "${resp}" | jq -r '.uploadId // empty'
}

ORDERS_ID="$(post_csv orders store_MIN_orders.csv)"
VARIANTS_ID="$(post_csv variants shop2_products_variants_MIN.csv)"
INVENTORY_ID="$(post_csv inventory_levels shop2_inventory_levels_MIN.csv)"
CATALOG_ID="$(post_csv catalog_joined shop2_catalog_joined_MIN.csv)"

curl "${SERVICE_URL}/api/upload-status/${VARIANTS_ID}"
curl "${SERVICE_URL}/api/upload-status/${CATALOG_ID}"

curl -X POST "${SERVICE_URL}/api/generate-rules" \
  -H "Content-Type: application/json" \
  -d "{\"csvUploadId\":\"${ORDERS_ID}\"}"

curl -X POST "${SERVICE_URL}/api/generate-bundles" \
  -H "Content-Type: application/json" \
  -d "{\"csvUploadId\":\"${CATALOG_ID}\"}"
