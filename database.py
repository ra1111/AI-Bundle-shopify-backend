"""
Database Configuration and Models
SQLAlchemy async setup with PostgreSQL
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (
    String, Text, Integer, Numeric, DateTime, Boolean, 
    ForeignKey, func, Index
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
import os
from typing import AsyncGenerator
import uuid
from datetime import datetime
from decimal import Decimal

# Database URL with asyncpg driver
DATABASE_URL = os.getenv("DATABASE_URL", "")
if DATABASE_URL:
    # Convert psycopg2 style URL to asyncpg compatible URL
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    # Remove sslmode parameter which is not supported by asyncpg
    if "?sslmode=" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.split("?sslmode=")[0]
    if "&sslmode=" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.split("&sslmode=")[0]
else:
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/bundledb"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("NODE_ENV") == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Validate connections before use
    pool_recycle=3600,   # Recycle connections every hour
    pool_timeout=30,     # Timeout for getting connection from pool
)

# Create sessionmaker
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Base model class
class Base(DeclarativeBase):
    pass

# Database models for 4-CSV ingestion: Orders, Products/Variants, Inventory, Catalog Joined

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password: Mapped[str] = mapped_column(Text, nullable=False)

class CsvUpload(Base):
    __tablename__ = "csv_uploads"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column("run_id", String, unique=True, index=True)  # Hash-based idempotency key
    shop_id: Mapped[str] = mapped_column("shop_id", String, index=True)  # For per-shop processing
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    csv_type: Mapped[str | None] = mapped_column("csv_type", String)  # orders, products_variants, inventory_levels, catalog_joined
    total_rows: Mapped[int] = mapped_column("total_rows", Integer, nullable=False)
    processed_rows: Mapped[int] = mapped_column("processed_rows", Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)  # 'processing', 'completed', 'failed', 'generating_bundles', 'bundle_generation_completed', 'bundle_generation_failed'
    error_message: Mapped[str | None] = mapped_column("error_message", Text)
    bundle_generation_metrics: Mapped[dict | None] = mapped_column("bundle_generation_metrics", JSONB)  # Store bundle generation metrics
    code_version: Mapped[str] = mapped_column("code_version", String, default="1.0.0")  # Track code version for schema evolution
    processing_params: Mapped[dict | None] = mapped_column("processing_params", JSONB)  # Store processing parameters
    schema_version: Mapped[str] = mapped_column("schema_version", String, default="1.0")  # CSV schema version
    time_window_start: Mapped[datetime | None] = mapped_column("time_window_start", DateTime)  # For delta processing
    time_window_end: Mapped[datetime | None] = mapped_column("time_window_end", DateTime)
    created_at: Mapped[datetime] = mapped_column("created_at", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updated_at", DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    orders = relationship("Order", back_populates="csv_upload")
    order_lines = relationship("OrderLine", back_populates="csv_upload")
    association_rules = relationship("AssociationRule", back_populates="csv_upload")
    bundle_recommendations = relationship("BundleRecommendation", back_populates="csv_upload")

class Order(Base):
    __tablename__ = "orders"
    
    # Use original Shopify order_id as primary key
    order_id: Mapped[str] = mapped_column("order_id", String, primary_key=True)
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    customer_id: Mapped[str | None] = mapped_column("customer_id", String)
    customer_email: Mapped[str | None] = mapped_column("customer_email", Text)
    customer_country: Mapped[str | None] = mapped_column("customer_country", Text)
    customer_currency: Mapped[str | None] = mapped_column("customer_currency", Text)
    clv_band: Mapped[str | None] = mapped_column("clv_band", Text)
    created_at: Mapped[datetime] = mapped_column("created_at", DateTime, nullable=False)
    channel: Mapped[str | None] = mapped_column("channel", Text)
    device: Mapped[str | None] = mapped_column("device", Text)
    subtotal: Mapped[Decimal | None] = mapped_column("subtotal", Numeric(10, 2))
    discount: Mapped[Decimal | None] = mapped_column("discount", Numeric(10, 2))
    discount_code: Mapped[str | None] = mapped_column("discount_code", Text)
    shipping: Mapped[Decimal | None] = mapped_column("shipping", Numeric(10, 2))
    taxes: Mapped[Decimal | None] = mapped_column("taxes", Numeric(10, 2))
    total: Mapped[Decimal | None] = mapped_column("total", Numeric(10, 2))
    financial_status: Mapped[str | None] = mapped_column("financial_status", Text)
    fulfillment_status: Mapped[str | None] = mapped_column("fulfillment_status", Text)
    returned: Mapped[bool | None] = mapped_column("returned", Boolean)
    basket_item_count: Mapped[int | None] = mapped_column("basket_item_count", Integer)
    basket_line_count: Mapped[int | None] = mapped_column("basket_line_count", Integer)
    basket_value: Mapped[Decimal | None] = mapped_column("basket_value", Numeric(10, 2))
    
    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="orders")
    order_lines = relationship("OrderLine", back_populates="order")

class OrderLine(Base):
    __tablename__ = "order_lines"
    
    id: Mapped[str] = mapped_column("id", String, primary_key=True)
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    order_id: Mapped[str] = mapped_column("order_id", String, ForeignKey("orders.order_id"), nullable=False)
    sku: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str | None] = mapped_column(Text)
    brand: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    subcategory: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(Text)
    material: Mapped[str | None] = mapped_column(Text)
    price: Mapped[Decimal | None] = mapped_column("price", Numeric(10, 2))
    cost: Mapped[Decimal | None] = mapped_column("cost", Numeric(10, 2))
    weight_kg: Mapped[Decimal | None] = mapped_column("weight_kg", Numeric(10, 3))
    tags: Mapped[str | None] = mapped_column(Text)
    quantity: Mapped[int | None] = mapped_column("quantity", Integer)
    unit_price: Mapped[Decimal | None] = mapped_column("unit_price", Numeric(10, 2))
    line_total: Mapped[Decimal | None] = mapped_column("line_total", Numeric(10, 2))
    hist_views: Mapped[int | None] = mapped_column("hist_views", Integer)
    hist_adds: Mapped[int | None] = mapped_column("hist_adds", Integer)
    # New fields for enhanced Shopify data
    variant_id: Mapped[str | None] = mapped_column("variant_id", String)
    line_item_id: Mapped[str | None] = mapped_column("line_item_id", String)
    line_discount: Mapped[Decimal | None] = mapped_column("line_discount", Numeric(10, 2))
    
    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="order_lines")
    order = relationship("Order", back_populates="order_lines")

class Product(Base):
    __tablename__ = "products"
    
    # Use sku as primary key to match actual database schema
    sku: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    brand: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    subcategory: Mapped[str | None] = mapped_column(Text)
    color: Mapped[str | None] = mapped_column(Text)
    material: Mapped[str | None] = mapped_column(Text)
    price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    cost: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    weight_kg: Mapped[Decimal | None] = mapped_column("weight_kg", Numeric(8, 2))
    tags: Mapped[str | None] = mapped_column(Text)

class Variant(Base):
    __tablename__ = "variants"
    
    variant_id: Mapped[str] = mapped_column("variant_id", String, primary_key=True)
    product_id: Mapped[str] = mapped_column("product_id", String, nullable=False)
    sku: Mapped[str | None] = mapped_column(Text, nullable=True)
    variant_title: Mapped[str] = mapped_column("variant_title", Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    compare_at_price: Mapped[Decimal | None] = mapped_column("compare_at_price", Numeric(10, 2))
    inventory_item_id: Mapped[str] = mapped_column("inventory_item_id", String, nullable=False)
    inventory_item_created_at: Mapped[datetime | None] = mapped_column("inventory_item_created_at", DateTime)
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    
    # Relationships
    csv_upload = relationship("CsvUpload")

class InventoryLevel(Base):
    __tablename__ = "inventory_levels"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    inventory_item_id: Mapped[str] = mapped_column("inventory_item_id", String, nullable=False)
    location_id: Mapped[str] = mapped_column("location_id", String, nullable=False)
    available: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column("updated_at", DateTime, nullable=False)
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    
    # Relationships
    csv_upload = relationship("CsvUpload")

class CatalogSnapshot(Base):
    __tablename__ = "catalog_snapshot"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id: Mapped[str] = mapped_column("product_id", String, nullable=False)
    product_title: Mapped[str] = mapped_column("product_title", Text, nullable=False)
    product_type: Mapped[str | None] = mapped_column("product_type", Text)
    tags: Mapped[str | None] = mapped_column(Text)
    product_status: Mapped[str] = mapped_column("product_status", Text, nullable=False)
    vendor: Mapped[str | None] = mapped_column(Text)
    product_created_at: Mapped[datetime | None] = mapped_column("product_created_at", DateTime)
    product_published_at: Mapped[datetime | None] = mapped_column("product_published_at", DateTime)
    variant_id: Mapped[str] = mapped_column("variant_id", String, nullable=False)
    sku: Mapped[str | None] = mapped_column(Text)
    variant_title: Mapped[str] = mapped_column("variant_title", Text, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    compare_at_price: Mapped[Decimal | None] = mapped_column("compare_at_price", Numeric(10, 2))
    inventory_item_id: Mapped[str] = mapped_column("inventory_item_id", String, nullable=False)
    inventory_item_created_at: Mapped[datetime | None] = mapped_column("inventory_item_created_at", DateTime)
    available_total: Mapped[int | None] = mapped_column("available_total", Integer)
    last_inventory_update: Mapped[datetime | None] = mapped_column("last_inventory_update", DateTime)
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    
    # Computed fields for objectives (only fields that exist in database)
    is_slow_mover: Mapped[bool] = mapped_column("is_slow_mover", Boolean, default=False)
    is_new_launch: Mapped[bool] = mapped_column("is_new_launch", Boolean, default=False)
    is_seasonal: Mapped[bool] = mapped_column("is_seasonal", Boolean, default=False)
    is_high_margin: Mapped[bool] = mapped_column("is_high_margin", Boolean, default=False)
    # Note: avg_discount_pct and objective_flags removed - don't exist in actual database
    
    # Relationships
    csv_upload = relationship("CsvUpload")

class AssociationRule(Base):
    __tablename__ = "association_rules"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    antecedent: Mapped[dict] = mapped_column(JSONB, nullable=False)  # Array of SKUs
    consequent: Mapped[dict] = mapped_column(JSONB, nullable=False)  # Array of SKUs
    support: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    lift: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    created_at: Mapped[datetime] = mapped_column("created_at", DateTime, default=func.now())
    
    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="association_rules")

class Bundle(Base):
    __tablename__ = "bundles"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    bundle_type: Mapped[str] = mapped_column("bundle_type", Text, nullable=False)  # 'FBT', 'FIXED', 'MIX_MATCH', 'VOLUME_DISCOUNT', 'BXGY'
    products: Mapped[dict] = mapped_column(JSONB, nullable=False)  # Array of product configurations
    pricing: Mapped[dict] = mapped_column(JSONB, nullable=False)  # Pricing rules
    is_active: Mapped[bool] = mapped_column("is_active", Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column("created_at", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updated_at", DateTime, default=func.now())

class BundleRecommendation(Base):
    __tablename__ = "bundle_recommendations"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    csv_upload_id: Mapped[str | None] = mapped_column("csv_upload_id", String, ForeignKey("csv_uploads.id"))
    bundle_type: Mapped[str] = mapped_column("bundle_type", Text, nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)  # NEW: objective function
    products: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pricing: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ai_copy: Mapped[dict] = mapped_column("ai_copy", JSONB, nullable=False)  # Generated title, description, value props
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    predicted_lift: Mapped[Decimal] = mapped_column("predicted_lift", Numeric(5, 4), nullable=False)
    support: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    lift: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    ranking_score: Mapped[Decimal] = mapped_column("ranking_score", Numeric(8, 4), nullable=False)  # NEW: confidence * lift
    discount_reference: Mapped[str | None] = mapped_column("discount_reference", Text)  # NEW: avg_discount vs default
    is_approved: Mapped[bool] = mapped_column("is_approved", Boolean, nullable=False, default=False)
    is_used: Mapped[bool] = mapped_column("is_used", Boolean, nullable=False, default=False)  # NEW: tracking usage
    rank_position: Mapped[int | None] = mapped_column("rank_position", Integer)  # NEW: rank within bundle type
    created_at: Mapped[datetime] = mapped_column("created_at", DateTime, default=func.now())
    
    # Relationships
    csv_upload = relationship("CsvUpload", back_populates="bundle_recommendations")

# Create unique indexes for UPSERT operations
# These indexes are required for ON CONFLICT clauses in PostgreSQL UPSERT operations

# Partial unique indexes for OrderLine to handle nullable line_item_id
# Two separate indexes to ensure uniqueness when line_item_id is NULL vs NOT NULL
Index('uq_order_lines_with_line_item_id', OrderLine.order_id, OrderLine.line_item_id, unique=True, 
      postgresql_where=OrderLine.line_item_id.is_not(None))
Index('uq_order_lines_without_line_item_id', OrderLine.order_id, OrderLine.sku, unique=True,
      postgresql_where=OrderLine.line_item_id.is_(None))

Index('uq_inventory_levels_item_location', InventoryLevel.inventory_item_id, InventoryLevel.location_id, unique=True)
Index('uq_catalog_snapshots_upload_variant', CatalogSnapshot.csv_upload_id, CatalogSnapshot.variant_id, unique=True)

# Dependency to get database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Initialize database
async def init_db():
    """Initialize database - create tables if they don't exist"""
    async with engine.begin() as conn:
        # Create/update tables to add new versioning columns
        await conn.run_sync(Base.metadata.create_all)