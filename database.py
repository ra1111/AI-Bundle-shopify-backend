# --- models.py (or the models section of database.py) ---

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Text, Integer, Numeric, DateTime, Boolean,
    ForeignKey, func, Index, event, CheckConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator, Optional
import logging, os, uuid
import json

# Load environment variables early (before reading DATABASE_URL)
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------------
# Engine / Session (CockroachDB compatible)
# -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    # CockroachDB requires SSL; preserve sslmode parameter for asyncpg
    # asyncpg uses 'ssl' parameter, not 'sslmode', so we convert it
    if "sslmode=verify-full" in DATABASE_URL:
        # For CockroachDB, asyncpg needs ssl='require' or we handle SSL via connect_args
        # Keep the URL clean and handle SSL in connect_args if needed
        DATABASE_URL = DATABASE_URL.replace("?sslmode=verify-full", "")
        DATABASE_URL = DATABASE_URL.replace("&sslmode=verify-full", "")

    # CockroachDB-optimized connection settings
    # CockroachDB is PostgreSQL-compatible but has some differences
    # Most importantly: CockroachDB doesn't have the 'json' type, only 'jsonb'
    async def _setup_asyncpg_connection(conn):
        """Custom connection setup that avoids JSON codec issues with CockroachDB"""
        # Don't set up JSON codecs - CockroachDB doesn't have the 'json' type
        # We only use JSONB in our models anyway
        pass

    # OPTIMIZATION: Tuned connection pool for high-concurrency ML operations
    # Supports parallel candidate generation, embedding batches, and optimization engine
    engine = create_async_engine(
        DATABASE_URL,
        echo=os.getenv("NODE_ENV") == "development",
        pool_size=50,  # Increased to support 40+ parallel Phase 3 tasks
        max_overflow=20,  # Increased from 10 to 20 (Total max: 70 connections)
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_timeout=20,  # Increased from 15 to 20 seconds for busy periods
        # CockroachDB compatibility settings
        connect_args={
            "ssl": "require",  # CockroachDB requires SSL
            "server_settings": {
                "application_name": "ai_bundle_creator",
            },
            # OPTIMIZATION: asyncpg connection tuning
            "command_timeout": 60,  # Command timeout in seconds
            "timeout": 30,  # Connection timeout in seconds
        },
        use_insertmanyvalues=True,  # CockroachDB supports this
        # OPTIMIZATION: Enhanced query execution settings
        execution_options={
            "isolation_level": "READ COMMITTED",  # Optimal for high-concurrency reads
        },
    )

    # Monkey-patch the asyncpg dialect for CockroachDB compatibility
    from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
    from sqlalchemy.dialects.postgresql.base import PGDialect

    # Fix 1: JSON codec setup (CockroachDB doesn't have 'json' type)
    original_setup_asyncpg_json_codec = PGDialect_asyncpg.setup_asyncpg_json_codec

    async def patched_setup_asyncpg_json_codec(self, conn):
        """Patched version that skips JSON codec (CockroachDB doesn't have 'json' type)"""
        try:
            # Try to setup JSONB codec only (skip JSON)
            import asyncpg
            await conn.set_type_codec(
                'jsonb',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog',
                format='text',
            )
        except Exception:
            # If even JSONB fails, just skip codec setup entirely
            # SQLAlchemy will handle JSON encoding/decoding in Python
            pass

    PGDialect_asyncpg.setup_asyncpg_json_codec = patched_setup_asyncpg_json_codec

    # Fix 2: Version string parsing (CockroachDB has different version format)
    original_get_server_version_info = PGDialect._get_server_version_info

    def patched_get_server_version_info(self, connection):
        """Patched version that handles CockroachDB version strings"""
        import re
        version_str = connection.scalar(text("SELECT version()"))

        # Check if this is CockroachDB
        if "CockroachDB" in version_str:
            # Parse CockroachDB version: "CockroachDB CCL v25.2.6 ..."
            match = re.search(r'v(\d+)\.(\d+)\.(\d+)', version_str)
            if match:
                # CockroachDB is PostgreSQL-compatible
                # Return a PostgreSQL version tuple that SQLAlchemy can work with
                # CockroachDB v25.x is compatible with PostgreSQL 13+
                return (13, 0)  # Report as PostgreSQL 13

        # For regular PostgreSQL, use original logic
        return original_get_server_version_info(self, connection)

    PGDialect._get_server_version_info = patched_get_server_version_info
else:
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_NAME = os.getenv("DB_NAME", "bundles")
    SOCKET = os.getenv("INSTANCE_UNIX_SOCKET")
    if SOCKET:
        DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:@/{DB_NAME}?host={SOCKET}"
        engine = create_async_engine(
            DATABASE_URL,
            echo=os.getenv("NODE_ENV") == "development",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_timeout=15,
        )
    else:
        DATABASE_URL = "sqlite+aiosqlite:///:memory:"
        engine = create_async_engine(
            DATABASE_URL,
            echo=os.getenv("NODE_ENV") == "development",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

def _redact_db_url(url: str) -> str:
    try:
        if "@" in url and "://" in url:
            head, tail = url.split("://", 1)
            creds, hostpart = tail.split("@", 1)
            if ":" in creds:
                user, _pwd = creds.split(":", 1)
                return f"{head}://{user}:******@{hostpart}"
    except Exception:
        pass
    return "******"

logger = logging.getLogger(__name__)
logger.info(f"Creating SQL engine for { _redact_db_url(DATABASE_URL) }")

async def probe_db_connection():
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("DB connectivity probe: OK")
    except Exception as e:
        logger.exception(f"DB connectivity probe failed: {e}")

# -------------------------------------------------------------------
# Base
# -------------------------------------------------------------------
class Base(DeclarativeBase):
    pass

# -------------------------------------------------------------------
# MODELS (with correct nullable flags)
# -------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password: Mapped[str] = mapped_column(Text, nullable=False)


class CsvUpload(Base):
    __tablename__ = "csv_uploads"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # Can be absent on first upload; keep unique when present
    # Multiple uploads (orders, variants, inventory, catalog) belong to the same run_id
    # Remove unique=True so all four CSVs can share a single run_id
    run_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    shop_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)

    filename: Mapped[str] = mapped_column(Text, nullable=False)
    csv_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    bundle_generation_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    code_version: Mapped[str] = mapped_column(String, default="1.0.0", nullable=False)
    processing_params: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    schema_version: Mapped[str] = mapped_column(String, default="1.0", nullable=False)

    time_window_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    time_window_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    orders = relationship("Order", back_populates="csv_upload")
    order_lines = relationship("OrderLine", back_populates="csv_upload")
    association_rules = relationship("AssociationRule", back_populates="csv_upload")
    bundle_recommendations = relationship("BundleRecommendation", back_populates="csv_upload")


class GenerationProgress(Base):
    __tablename__ = "generation_progress"

    upload_id: Mapped[str] = mapped_column(String, primary_key=True)
    shop_domain: Mapped[str] = mapped_column(Text, nullable=False)
    step: Mapped[str] = mapped_column(Text, nullable=False)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("progress BETWEEN 0 AND 100", name="ck_generation_progress_range"),
        CheckConstraint(
            "status IN ('in_progress','completed','failed')",
            name="ck_generation_progress_status",
        ),
    )


class Order(Base):
    __tablename__ = "orders"

    # Always present in your CSVs
    order_id: Mapped[str] = mapped_column(String, primary_key=True)

    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)
    customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    customer_email: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    customer_country: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    customer_currency: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clv_band: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # present in CSVs
    channel: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    device: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    subtotal: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    discount: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    discount_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    shipping: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    taxes: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    total: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)

    financial_status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fulfillment_status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    returned: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    basket_item_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    basket_line_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    basket_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)

    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="orders")
    order_lines = relationship("OrderLine", back_populates="order")


class OrderLine(Base):
    __tablename__ = "order_lines"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    # required: must point to an existing order
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.order_id"), nullable=False)

    sku: Mapped[Optional[str]] = mapped_column(Text, nullable=True)        # can be blank in Shopify
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    brand: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subcategory: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    material: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 3), nullable=True)
    weight_kg: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 3), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    quantity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    unit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    line_total: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    hist_views: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hist_adds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    variant_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    line_item_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    line_discount: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)

    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="order_lines")
    order = relationship("Order", back_populates="order_lines")


class Product(Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    brand: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subcategory: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    material: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    weight_kg: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 2), nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Variant(Base):
    __tablename__ = "variants"

    variant_id: Mapped[str] = mapped_column(String, primary_key=True)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    sku: Mapped[Optional[str]] = mapped_column(Text, nullable=True)              # often missing
    variant_title: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    compare_at_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    inventory_item_id: Mapped[str] = mapped_column(String, nullable=False)
    inventory_item_created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    csv_upload = relationship("CsvUpload")


class InventoryLevel(Base):
    __tablename__ = "inventory_levels"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    inventory_item_id: Mapped[str] = mapped_column(String, nullable=False)
    location_id: Mapped[str] = mapped_column(String, nullable=False)
    available: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    csv_upload = relationship("CsvUpload")


class CatalogSnapshot(Base):
    __tablename__ = "catalog_snapshot"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    product_title: Mapped[str] = mapped_column(Text, nullable=False)
    product_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    product_status: Mapped[str] = mapped_column(Text, nullable=False)
    vendor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    product_created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    product_published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    variant_id: Mapped[str] = mapped_column(String, nullable=False)
    sku: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    variant_title: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    compare_at_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    inventory_item_id: Mapped[str] = mapped_column(String, nullable=False)
    inventory_item_created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    available_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_inventory_update: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    is_slow_mover: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_new_launch: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_seasonal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_high_margin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    csv_upload = relationship("CsvUpload")


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class AssociationRule(Base):
    __tablename__ = "association_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    antecedent: Mapped[dict] = mapped_column(JSONB, nullable=False)
    consequent: Mapped[dict] = mapped_column(JSONB, nullable=False)
    support: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    lift: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    csv_upload = relationship("CsvUpload", back_populates="association_rules")


class Bundle(Base):
    __tablename__ = "bundles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bundle_type: Mapped[str] = mapped_column(Text, nullable=False)
    products: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pricing: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    shop_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class BundleRecommendation(Base):
    __tablename__ = "bundle_recommendations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    csv_upload_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)
    shop_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)

    bundle_type: Mapped[str] = mapped_column(Text, nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)

    products: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pricing: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ai_copy: Mapped[dict] = mapped_column(JSONB, nullable=False)

    confidence: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    predicted_lift: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    support: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 6), nullable=True)
    lift: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 6), nullable=True)
    ranking_score: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)

    discount_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rank_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    csv_upload = relationship("CsvUpload", back_populates="bundle_recommendations")


class ShopSyncStatus(Base):
    __tablename__ = "shop_sync_status"

    shop_id: Mapped[str] = mapped_column(String, primary_key=True)
    initial_sync_completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_sync_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_sync_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

# -------------------------------------------------------------------
# Indexes (unchanged, still valid)
# -------------------------------------------------------------------
Index(
    'uq_order_lines_with_line_item_id',
    OrderLine.order_id, OrderLine.line_item_id,
    unique=True,
    postgresql_where=OrderLine.line_item_id.is_not(None)
)
Index(
    'uq_order_lines_without_line_item_id',
    OrderLine.order_id, OrderLine.sku,
    unique=True,
    postgresql_where=OrderLine.line_item_id.is_(None)
)
Index('uq_inventory_levels_item_location',
      InventoryLevel.inventory_item_id, InventoryLevel.location_id,
      unique=True)
Index('uq_catalog_snapshots_upload_variant',
      CatalogSnapshot.csv_upload_id, CatalogSnapshot.variant_id,
      unique=True)
Index('ix_bundle_recommendations_shop', BundleRecommendation.shop_id)
Index('ix_bundles_created_at', Bundle.created_at)
Index('ix_bundles_type', Bundle.bundle_type)
Index('ix_bundle_recommendations_upload', BundleRecommendation.csv_upload_id)
Index('ix_association_rules_upload', AssociationRule.csv_upload_id)
# -------------------------------------------------------------------
# DI + init helpers
# -------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Ensure tables exist."""
    await probe_db_connection()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB init complete (tables ensured).")
