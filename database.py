# --- models.py (or the models section of database.py) ---

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Text, Integer, Numeric, DateTime, Boolean,
    ForeignKey, func, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator
import logging, os, uuid

# -------------------------------------------------------------------
# Engine / Session (unchanged except for the log + probe helpers)
# -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    for needle in ("?sslmode=", "&sslmode="):
        if needle in DATABASE_URL:
            DATABASE_URL = DATABASE_URL.split(needle)[0]
    engine = create_async_engine(
        DATABASE_URL,
        echo=os.getenv("NODE_ENV") == "development",
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_timeout=30,
    )
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
            pool_recycle=3600,
            pool_timeout=30,
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
    run_id: Mapped[str | None] = mapped_column(String, unique=True, index=True, nullable=True)
    shop_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)

    filename: Mapped[str] = mapped_column(Text, nullable=False)
    csv_type: Mapped[str | None] = mapped_column(String, nullable=True)

    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    bundle_generation_metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    code_version: Mapped[str] = mapped_column(String, default="1.0.0", nullable=False)
    processing_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    schema_version: Mapped[str] = mapped_column(String, default="1.0", nullable=False)

    time_window_start: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    time_window_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    orders = relationship("Order", back_populates="csv_upload")
    order_lines = relationship("OrderLine", back_populates="csv_upload")
    association_rules = relationship("AssociationRule", back_populates="csv_upload")
    bundle_recommendations = relationship("BundleRecommendation", back_populates="csv_upload")


class Order(Base):
    __tablename__ = "orders"

    # Always present in your CSVs
    order_id: Mapped[str] = mapped_column(String, primary_key=True)

    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)
    customer_id: Mapped[str | None] = mapped_column(String, nullable=True)
    customer_email: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_country: Mapped[str | None] = mapped_column(Text, nullable=True)
    customer_currency: Mapped[str | None] = mapped_column(Text, nullable=True)
    clv_band: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # present in CSVs
    channel: Mapped[str | None] = mapped_column(Text, nullable=True)
    device: Mapped[str | None] = mapped_column(Text, nullable=True)

    subtotal: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    discount: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    discount_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    shipping: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    taxes: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    total: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    financial_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    fulfillment_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    returned: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    basket_item_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    basket_line_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    basket_value: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="orders")
    order_lines = relationship("OrderLine", back_populates="order")


class OrderLine(Base):
    __tablename__ = "order_lines"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    # required: must point to an existing order
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.order_id"), nullable=False)

    sku: Mapped[str | None] = mapped_column(Text, nullable=True)        # can be blank in Shopify
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    subcategory: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str | None] = mapped_column(Text, nullable=True)
    material: Mapped[str | None] = mapped_column(Text, nullable=True)

    price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    cost: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)

    quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    line_total: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    hist_views: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hist_adds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    variant_id: Mapped[str | None] = mapped_column(String, nullable=True)
    line_item_id: Mapped[str | None] = mapped_column(String, nullable=True)
    line_discount: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="order_lines")
    order = relationship("Order", back_populates="order_lines")


class Product(Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    subcategory: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str | None] = mapped_column(Text, nullable=True)
    material: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    cost: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    weight_kg: Mapped[Decimal | None] = mapped_column(Numeric(8, 2), nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)


class Variant(Base):
    __tablename__ = "variants"

    variant_id: Mapped[str] = mapped_column(String, primary_key=True)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    sku: Mapped[str | None] = mapped_column(Text, nullable=True)              # often missing
    variant_title: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    compare_at_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    inventory_item_id: Mapped[str] = mapped_column(String, nullable=False)
    inventory_item_created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    csv_upload = relationship("CsvUpload")


class InventoryLevel(Base):
    __tablename__ = "inventory_levels"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    inventory_item_id: Mapped[str] = mapped_column(String, nullable=False)
    location_id: Mapped[str] = mapped_column(String, nullable=False)
    available: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    csv_upload = relationship("CsvUpload")


class CatalogSnapshot(Base):
    __tablename__ = "catalog_snapshot"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    product_title: Mapped[str] = mapped_column(Text, nullable=False)
    product_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_status: Mapped[str] = mapped_column(Text, nullable=False)
    vendor: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    product_published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    variant_id: Mapped[str] = mapped_column(String, nullable=False)
    sku: Mapped[str | None] = mapped_column(Text, nullable=True)
    variant_title: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    compare_at_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    inventory_item_id: Mapped[str] = mapped_column(String, nullable=False)
    inventory_item_created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    available_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_inventory_update: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    csv_upload = relationship("CsvUpload")


class AssociationRule(Base):
    __tablename__ = "association_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    antecedent: Mapped[dict] = mapped_column(JSONB, nullable=False)
    consequent: Mapped[dict] = mapped_column(JSONB, nullable=False)
    support: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    lift: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    csv_upload = relationship("CsvUpload", back_populates="association_rules")


class Bundle(Base):
    __tablename__ = "bundles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    bundle_type: Mapped[str] = mapped_column(Text, nullable=False)
    products: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pricing: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class BundleRecommendation(Base):
    __tablename__ = "bundle_recommendations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    csv_upload_id: Mapped[str | None] = mapped_column(String, ForeignKey("csv_uploads.id"), nullable=True)

    bundle_type: Mapped[str] = mapped_column(Text, nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)

    products: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pricing: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ai_copy: Mapped[dict] = mapped_column(JSONB, nullable=False)

    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    predicted_lift: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    support: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    lift: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    ranking_score: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)

    discount_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rank_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    csv_upload = relationship("CsvUpload", back_populates="bundle_recommendations")

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
