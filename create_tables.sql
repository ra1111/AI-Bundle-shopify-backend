-- Create all tables for AI Bundle Creator
-- CockroachDB Compatible SQL
-- Run this in CockroachDB console to create all tables at once

-- Drop existing tables (if you want to start fresh)
-- DROP TABLE IF EXISTS bundle_recommendations CASCADE;
-- DROP TABLE IF EXISTS association_rules CASCADE;
-- DROP TABLE IF EXISTS bundles CASCADE;
-- DROP TABLE IF EXISTS embedding_cache CASCADE;
-- DROP TABLE IF EXISTS catalog_snapshot CASCADE;
-- DROP TABLE IF EXISTS inventory_levels CASCADE;
-- DROP TABLE IF EXISTS variants CASCADE;
-- DROP TABLE IF EXISTS order_lines CASCADE;
-- DROP TABLE IF EXISTS orders CASCADE;
-- DROP TABLE IF EXISTS generation_progress CASCADE;
-- DROP TABLE IF EXISTS products CASCADE;
-- DROP TABLE IF EXISTS csv_uploads CASCADE;
-- DROP TABLE IF EXISTS shop_sync_status CASCADE;
-- DROP TABLE IF EXISTS users CASCADE;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
);

-- CSV Uploads table
CREATE TABLE IF NOT EXISTS csv_uploads (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    run_id VARCHAR,
    shop_id VARCHAR,
    filename TEXT NOT NULL,
    csv_type VARCHAR,
    total_rows INTEGER NOT NULL DEFAULT 0,
    processed_rows INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT,
    bundle_generation_metrics JSONB,
    code_version VARCHAR NOT NULL DEFAULT '1.0.0',
    processing_params JSONB,
    schema_version VARCHAR NOT NULL DEFAULT '1.0',
    time_window_start TIMESTAMP,
    time_window_end TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_csv_uploads_run_id ON csv_uploads(run_id);
CREATE INDEX IF NOT EXISTS ix_csv_uploads_shop_id ON csv_uploads(shop_id);

-- Generation Progress table
CREATE TABLE IF NOT EXISTS generation_progress (
    upload_id VARCHAR PRIMARY KEY,
    shop_domain TEXT NOT NULL,
    step TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT,
    metadata JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_generation_progress_range CHECK (progress BETWEEN 0 AND 100),
    CONSTRAINT ck_generation_progress_status CHECK (status IN ('in_progress','completed','failed'))
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR PRIMARY KEY,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id),
    customer_id VARCHAR,
    customer_email TEXT,
    customer_country TEXT,
    customer_currency TEXT,
    clv_band TEXT,
    created_at TIMESTAMP NOT NULL,
    channel TEXT,
    device TEXT,
    subtotal NUMERIC(10, 2),
    discount NUMERIC(10, 2),
    discount_code TEXT,
    shipping NUMERIC(10, 2),
    taxes NUMERIC(10, 2),
    total NUMERIC(10, 2),
    financial_status TEXT,
    fulfillment_status TEXT,
    returned BOOLEAN,
    basket_item_count INTEGER,
    basket_line_count INTEGER,
    basket_value NUMERIC(10, 2)
);

-- Order Lines table
CREATE TABLE IF NOT EXISTS order_lines (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id),
    order_id VARCHAR NOT NULL REFERENCES orders(order_id),
    sku TEXT,
    name TEXT,
    brand TEXT,
    category TEXT,
    subcategory TEXT,
    color TEXT,
    material TEXT,
    price NUMERIC(10, 2),
    cost NUMERIC(12, 3),
    weight_kg NUMERIC(10, 3),
    tags TEXT,
    quantity INTEGER,
    unit_price NUMERIC(10, 2),
    line_total NUMERIC(10, 2),
    hist_views INTEGER,
    hist_adds INTEGER,
    variant_id VARCHAR,
    line_item_id VARCHAR,
    line_discount NUMERIC(10, 2)
);

-- Unique indexes for order_lines (partial indexes)
CREATE UNIQUE INDEX IF NOT EXISTS uq_order_lines_with_line_item_id
    ON order_lines(order_id, line_item_id)
    WHERE line_item_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_order_lines_without_line_item_id
    ON order_lines(order_id, sku)
    WHERE line_item_id IS NULL;

-- Products table
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT,
    brand TEXT,
    category TEXT,
    subcategory TEXT,
    color TEXT,
    material TEXT,
    price NUMERIC(10, 2),
    cost NUMERIC(10, 2),
    weight_kg NUMERIC(8, 2),
    tags TEXT
);

-- Variants table
CREATE TABLE IF NOT EXISTS variants (
    variant_id VARCHAR PRIMARY KEY,
    product_id VARCHAR NOT NULL,
    sku TEXT,
    variant_title TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    compare_at_price NUMERIC(10, 2),
    inventory_item_id VARCHAR NOT NULL,
    inventory_item_created_at TIMESTAMP,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id)
);

-- Inventory Levels table
CREATE TABLE IF NOT EXISTS inventory_levels (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    inventory_item_id VARCHAR NOT NULL,
    location_id VARCHAR NOT NULL,
    available INTEGER NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_inventory_levels_item_location
    ON inventory_levels(inventory_item_id, location_id);

-- Catalog Snapshot table
CREATE TABLE IF NOT EXISTS catalog_snapshot (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    product_id VARCHAR NOT NULL,
    product_title TEXT NOT NULL,
    product_type TEXT,
    tags TEXT,
    product_status TEXT NOT NULL,
    vendor TEXT,
    product_created_at TIMESTAMP,
    product_published_at TIMESTAMP,
    variant_id VARCHAR NOT NULL,
    sku TEXT,
    variant_title TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    compare_at_price NUMERIC(10, 2),
    inventory_item_id VARCHAR NOT NULL,
    inventory_item_created_at TIMESTAMP,
    available_total INTEGER,
    last_inventory_update TIMESTAMP,
    is_slow_mover BOOLEAN NOT NULL DEFAULT FALSE,
    is_new_launch BOOLEAN NOT NULL DEFAULT FALSE,
    is_seasonal BOOLEAN NOT NULL DEFAULT FALSE,
    is_high_margin BOOLEAN NOT NULL DEFAULT FALSE,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_catalog_snapshots_upload_variant
    ON catalog_snapshot(csv_upload_id, variant_id);

-- Embedding Cache table
CREATE TABLE IF NOT EXISTS embedding_cache (
    key VARCHAR PRIMARY KEY,
    payload TEXT NOT NULL,
    expires_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

-- Association Rules table
CREATE TABLE IF NOT EXISTS association_rules (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id),
    antecedent JSONB NOT NULL,
    consequent JSONB NOT NULL,
    support NUMERIC(12, 6) NOT NULL,
    confidence NUMERIC(12, 6) NOT NULL,
    lift NUMERIC(12, 6) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_association_rules_upload ON association_rules(csv_upload_id);

-- Bundles table
CREATE TABLE IF NOT EXISTS bundles (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    name TEXT NOT NULL,
    description TEXT,
    bundle_type TEXT NOT NULL,
    products JSONB NOT NULL,
    pricing JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    shop_id VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bundles_shop_id ON bundles(shop_id);
CREATE INDEX IF NOT EXISTS ix_bundles_created_at ON bundles(created_at);
CREATE INDEX IF NOT EXISTS ix_bundles_type ON bundles(bundle_type);

-- Bundle Recommendations table
CREATE TABLE IF NOT EXISTS bundle_recommendations (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::VARCHAR,
    csv_upload_id VARCHAR REFERENCES csv_uploads(id),
    shop_id VARCHAR,
    bundle_type TEXT NOT NULL,
    objective TEXT NOT NULL,
    products JSONB NOT NULL,
    pricing JSONB NOT NULL,
    ai_copy JSONB NOT NULL,
    confidence NUMERIC(12, 6) NOT NULL,
    predicted_lift NUMERIC(12, 6) NOT NULL,
    support NUMERIC(12, 6),
    lift NUMERIC(12, 6),
    ranking_score NUMERIC(12, 6) NOT NULL,
    discount_reference TEXT,
    is_approved BOOLEAN NOT NULL DEFAULT FALSE,
    is_used BOOLEAN NOT NULL DEFAULT FALSE,
    rank_position INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bundle_recommendations_shop ON bundle_recommendations(shop_id);
CREATE INDEX IF NOT EXISTS ix_bundle_recommendations_upload ON bundle_recommendations(csv_upload_id);

-- Shop Sync Status table
CREATE TABLE IF NOT EXISTS shop_sync_status (
    shop_id VARCHAR PRIMARY KEY,
    initial_sync_completed BOOLEAN NOT NULL DEFAULT FALSE,
    last_sync_started_at TIMESTAMP,
    last_sync_completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

-- Success message
SELECT 'All tables created successfully!' AS status;
