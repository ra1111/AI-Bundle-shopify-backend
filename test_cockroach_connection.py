#!/usr/bin/env python3
"""
Test CockroachDB connection and create tables
"""
import asyncio
import os
from dotenv import load_dotenv

# IMPORTANT: Load environment variables BEFORE importing database module
load_dotenv()

from database import engine, init_db, Base, probe_db_connection
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test CockroachDB connection and initialize database"""
    try:
        logger.info("Testing CockroachDB connection...")

        # Test basic connectivity
        await probe_db_connection()

        # Check CockroachDB version
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"Database version: {version}")

            # Check if we're connected to CockroachDB
            if "CockroachDB" in version:
                logger.info("✓ Successfully connected to CockroachDB!")
            else:
                logger.warning(f"Connected to: {version}")

        # Initialize database (create tables)
        logger.info("Creating database tables...")
        await init_db()

        # Verify tables were created
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"✓ Created {len(tables)} tables: {', '.join(tables)}")

        logger.info("✓ Database initialization completed successfully!")

    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
