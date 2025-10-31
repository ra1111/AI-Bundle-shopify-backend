-- Create generation_progress table for bundle generation progress tracking
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
