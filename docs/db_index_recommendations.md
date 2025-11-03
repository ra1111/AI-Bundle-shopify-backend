# Database Index Recommendations

This checklist captures the indexes we should add (or verify) to keep the new
data‑mapping and candidate generation pipeline fast as volumes grow.  The list
is grouped by table; use descriptive names so we can reference them easily
in migrations and observability.

## `order_lines`
- `idx_order_lines_shop_run` **(shop_id, run_id, csv_upload_id)** – covering fetches by run.
- `idx_order_lines_shop_sku` **(shop_id, sku, csv_upload_id)** INCLUDE(line_total, quantity, unit_price).
- `idx_order_lines_upload_created_at` **(csv_upload_id, created_at)** for recent-run filters.

## `variants`
- `idx_variants_shop_sku` **(shop_id, sku)** INCLUDE(variant_id, inventory_item_id, price).
- `idx_variants_shop_run` **(shop_id, csv_upload_id)** to pair with data‑mapping prefetch.
- (Optional) **UNIQUE** on `(variant_id, csv_upload_id)` if not already enforced.

## `inventory_levels`
- `idx_inventory_levels_shop_item` **(shop_id, inventory_item_id)** INCLUDE(available, location_id).
- `idx_inventory_levels_run` **(csv_upload_id, inventory_item_id)** for run-scoped loads.

## `catalog_snapshots`
- `idx_catalog_snapshots_shop_sku` **(shop_id, sku)** INCLUDE(product_title, product_type, vendor).
- `idx_catalog_snapshots_shop_variant` **(shop_id, variant_id)** for join efficiency.
- Consider clustering on `(shop_id, csv_upload_id)` to keep run snapshots contiguous.

## `bundle_recommendations`
- `idx_br_shop_status_created` **(shop_id, is_approved, created_at DESC)** INCLUDE(objective, bundle_type).
- `idx_br_run_csv` **(csv_upload_id, shop_id)** to support after-run lookups.

## `embeddings` (if stored)
- Add a `VECTOR` column with ANN index (`ivfflat`/`hnsw`) on the embedding vector.
- Secondary index on `(shop_id, product_id)` linking back to catalog entries.

## Operational Notes
- Always analyze after adding indexes (`ANALYZE table_name`).
- Use `EXPLAIN (ANALYZE, BUFFERS)` on the hottest queries to confirm the optimizer picks them up.
- For CockroachDB/Postgres, set `fillfactor` 90 on write-heavy tables to reduce page splits.
- Monitor index bloat quarterly; consider `REINDEX` as needed.
