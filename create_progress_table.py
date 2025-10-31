#!/usr/bin/env python3
"""Create generation_progress table in CockroachDB"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from database import engine

async def create_table():
    """Create generation_progress table"""
    sql = """
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
    """

    async with engine.begin() as conn:
        await conn.execute(text(sql))
        print("✅ generation_progress table created successfully!")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_table())
