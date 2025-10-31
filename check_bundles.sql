-- Check if bundles were generated for the recent upload
SELECT
    id,
    csv_upload_id,
    shop_id,
    bundle_type,
    objective,
    confidence,
    ranking_score,
    is_approved,
    created_at
FROM bundle_recommendations
WHERE csv_upload_id = '39e5a4a8-cee8-43bd-a47e-86bdafae8b73'
ORDER BY ranking_score DESC
LIMIT 20;

-- Count total bundles generated
SELECT COUNT(*) as total_bundles
FROM bundle_recommendations
WHERE csv_upload_id = '39e5a4a8-cee8-43bd-a47e-86bdafae8b73';

-- Check bundles by type
SELECT
    bundle_type,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(ranking_score) as avg_score
FROM bundle_recommendations
WHERE csv_upload_id = '39e5a4a8-cee8-43bd-a47e-86bdafae8b73'
GROUP BY bundle_type;

-- See full bundle details (products, pricing, AI copy)
SELECT
    id,
    bundle_type,
    objective,
    products,
    pricing,
    ai_copy,
    confidence,
    ranking_score
FROM bundle_recommendations
WHERE csv_upload_id = '39e5a4a8-cee8-43bd-a47e-86bdafae8b73'
ORDER BY ranking_score DESC
LIMIT 5;

-- Check all recent bundles (any upload)
SELECT
    csv_upload_id,
    shop_id,
    COUNT(*) as bundle_count,
    MAX(created_at) as latest_generated
FROM bundle_recommendations
WHERE shop_id = 'rahular1.myshopify.com'
GROUP BY csv_upload_id, shop_id
ORDER BY latest_generated DESC;
