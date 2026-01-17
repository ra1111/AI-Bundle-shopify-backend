"""
Storage Service Layer
Provides database operations matching the TypeScript storage interface
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, desc, and_, or_, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from typing import List, Optional, Dict, Any, Tuple
from types import SimpleNamespace
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy.exc import ProgrammingError

from database import (
    AsyncSessionLocal, User, CsvUpload, Order, OrderLine, Product,
    Variant, InventoryLevel, CatalogSnapshot, EmbeddingCache,
    AssociationRule, Bundle, BundleRecommendation, ShopSyncStatus
)
from settings import resolve_shop_id, sanitize_shop_id, DEFAULT_SHOP_ID
from utils import retry_async, is_transient_error

# Import bundle validation for data quality checks
try:
    from schemas import validate_bundle
except ImportError:
    # Fallback if schemas module not available
    validate_bundle = None

logger = logging.getLogger(__name__)


class BatchConfig:
    """Auto-configure batching strategy based on dataset size for optimal performance"""

    @staticmethod
    def get_config(row_count: int) -> dict:
        """
        Returns adaptive batch configuration based on dataset size.

        Tiers:
        - SMALL (< 500): Single batch, minimal overhead
        - MEDIUM (500-5K): Multi-batch, single transaction
        - LARGE (5K-50K): Multi-batch, progressive commits (prevents timeouts)
        - HUGE (50K+): Parallel processing (future)
        """
        if row_count < 500:
            return {
                'tier': 'SMALL',
                'batch_size': row_count,  # Single batch
                'commit_per_batch': False,
                'description': 'Single batch for minimal overhead'
            }
        elif row_count < 5000:
            return {
                'tier': 'MEDIUM',
                'batch_size': 200,
                'commit_per_batch': False,  # Single transaction at end
                'description': 'Multi-batch with single commit'
            }
        elif row_count < 50000:
            return {
                'tier': 'LARGE',
                'batch_size': 500,
                'commit_per_batch': True,  # Progressive commits prevent timeouts
                'description': 'Progressive commits for large datasets'
            }
        else:
            return {
                'tier': 'HUGE',
                'batch_size': 1000,
                'commit_per_batch': True,
                'description': 'Chunked processing for huge datasets'
            }


class StorageService:
    """Storage service providing database operations"""

    # ---------- NEW: helpers & defaults ----------

    ORDER_LINE_TEXT_DEFAULT = "unspecified"

    # Retry configuration for transient database errors
    RETRY_MAX_ATTEMPTS = 3
    RETRY_BASE_DELAY = 0.5  # seconds

    @staticmethod
    def _convert_decimals_to_float(obj):
        """Recursively convert Decimal values to float for JSON serialization."""
        from decimal import Decimal

        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: StorageService._convert_decimals_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [StorageService._convert_decimals_to_float(item) for item in obj]
        else:
            return obj

    async def _with_retry(self, operation, *args, **kwargs):
        """
        Execute a database operation with automatic retry on transient failures.
        Implements exponential backoff with jitter.
        """
        import asyncio
        import random

        last_exception = None
        for attempt in range(self.RETRY_MAX_ATTEMPTS):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not is_transient_error(e) or attempt >= self.RETRY_MAX_ATTEMPTS - 1:
                    raise

                delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), 10.0)
                delay = delay * (0.5 + random.random())  # Add jitter
                logger.warning(
                    f"Retry {attempt + 1}/{self.RETRY_MAX_ATTEMPTS} for {operation.__name__} "
                    f"after {delay:.2f}s due to: {type(e).__name__}: {str(e)[:100]}"
                )
                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception

    # Pre-flight check cache: {csv_upload_id: (result, timestamp)}
    # Caches results for 60 seconds to handle rapid retries
    _preflight_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
    _PREFLIGHT_CACHE_TTL = 60  # seconds

    def _table_column_names(self, table):
        return {c.name for c in table.columns}

    def _filter_columns(self, table, rows: list[dict]) -> list[dict]:
        """Drop keys that don't exist on the SQLAlchemy table (prevents invalid kw errors)."""
        allowed = self._table_column_names(table)
        return [{k: v for k, v in row.items() if k in allowed} for row in rows]

    def _as_decimal(self, v, default="0"):
        from decimal import Decimal
        if v in (None, "", "NULL", "null"):
            return Decimal(default)
        try:
            return Decimal(str(v))
        except Exception:
            return Decimal(default)

    def _as_int(self, v, default=0):
        if v in (None, "", "NULL", "null"):
            return default
        try:
            return int(float(v))
        except Exception:
            return default

    def _sanitize_order(self, row: dict) -> dict:
        """Be defensive; ensure non-nullables have values."""
        # created_at must not be NULL in your model
        if not row.get("created_at"):
            from datetime import datetime
            row["created_at"] = datetime.now()
        # Optional: normalize empties for text fields you want non-NULL at DB level
        for k in ("customer_id","customer_email","customer_country","customer_currency",
                  "clv_band","channel","device","discount_code","financial_status","fulfillment_status"):
            if row.get(k) is None:
                row[k] = ""
        # Numerics
        for k in ("subtotal","discount","shipping","taxes","total","basket_value"):
            if k in row:
                row[k] = self._as_decimal(row.get(k), "0")
        for k in ("basket_item_count","basket_line_count"):
            if k in row:
                row[k] = self._as_int(row.get(k), 0)
        if row.get("returned") is None:
            row["returned"] = False
        return row

    def _sanitize_order_line(self, line: dict) -> Optional[dict]:
        """
        Fill defaults for NOT NULL columns. Skip if we truly can't make a valid line.
        DB is complaining about NULLs in: sku, subcategory, color, material, etc.
        """
        # Must have an order_id to relate the line; otherwise skip
        if not line.get("order_id"):
            return None

        # Normalize blank IDs to None so they fall in the (order_id, sku) bucket
        liid = (line.get("line_item_id") or "").strip()
        line["line_item_id"] = liid or None

        vid = (line.get("variant_id") or "").strip()
        line["variant_id"] = vid or None

        # Stable fallback SKU if missing (DB has NOT NULL on sku in your instance)
        sku = (line.get("sku") or "").strip()
        if not sku:
            # Compose a deterministic pseudo-sku from variant/line ids
            seed = line.get("variant_id") or line.get("line_item_id") or line.get("id")
            if not seed:
                # can't build a stable key; skip this line
                return None
            sku = f"no-sku-{str(seed)[-10:]}"
            # Clamp very long fallback SKUs if DB has length limit
            if len(sku) > 100:  # Common SKU field limit
                sku = sku[:100]
        line["sku"] = sku

        # Friendly defaults for text columns that appear NOT NULL in your DB
        for k in ("brand", "category", "subcategory", "color", "material", "tags", "name"):
            if not line.get(k):
                # name gets a more human default
                line[k] = "Unknown" if k == "name" else self.ORDER_LINE_TEXT_DEFAULT

        # Numerics (NOT NULL in your DB per errors)
        for k in ("price", "cost", "weight_kg", "unit_price", "line_total", "line_discount"):
            line[k] = self._as_decimal(line.get(k), "0")
        for k in ("quantity", "hist_views", "hist_adds"):
            line[k] = self._as_int(line.get(k), 0)

        return line
    
    def get_session(self):
        """Get database session context manager"""
        return AsyncSessionLocal()

    async def get(self, key: str) -> Optional[str]:
        """Generic key-value getter used by embedding cache."""
        async with self.get_session() as session:
            try:
                return await self._embedding_cache_get(session, key)
            except Exception:
                logger.exception("Failed to read embedding cache key=%s", key)
                return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Generic key-value setter used by embedding cache."""
        expires_at = None
        if ttl:
            try:
                expires_at = datetime.utcnow() + timedelta(seconds=int(ttl))
            except Exception:
                expires_at = None

        async with self.get_session() as session:
            try:
                await self._embedding_cache_set(session, key, value, expires_at)
            except Exception:
                logger.exception("Failed to write embedding cache key=%s", key)
                await session.rollback()

    async def _ensure_embedding_cache_table(self, session: AsyncSession) -> None:
        async def _create(sync_session):
            EmbeddingCache.__table__.create(
                bind=sync_session.bind, checkfirst=True
            )

        await session.run_sync(_create)

    async def _embedding_cache_get(
        self, session: AsyncSession, key: str, retried: bool = False
    ) -> Optional[str]:
        try:
            record = await session.get(EmbeddingCache, key)
            if not record:
                return None
            if record.expires_at and record.expires_at < datetime.utcnow():
                await session.delete(record)
                await session.commit()
                return None
            logger.debug("Embedding cache hit | key=%s", key)
            return record.payload
        except ProgrammingError as exc:
            if not retried and "embedding_cache" in str(exc).lower():
                logger.info("Embedding cache table missing; creating now.")
                await session.rollback()
                await self._ensure_embedding_cache_table(session)
                await session.commit()
                return await self._embedding_cache_get(session, key, retried=True)
            raise

    async def _embedding_cache_set(
        self,
        session: AsyncSession,
        key: str,
        value: str,
        expires_at: Optional[datetime],
        retried: bool = False,
    ) -> None:
        try:
            record = await session.get(EmbeddingCache, key)
            if record:
                record.payload = value
                record.expires_at = expires_at
                record.updated_at = datetime.utcnow()
            else:
                record = EmbeddingCache(
                    key=key,
                    payload=value,
                    expires_at=expires_at,
                )
                session.add(record)
            await session.commit()
            ttl_hint = (
                max(0, int((expires_at - datetime.utcnow()).total_seconds()))
                if expires_at
                else None
            )
            logger.debug(
                "Embedding cache write | key=%s ttl=%s",
                key,
                ttl_hint,
            )
        except ProgrammingError as exc:
            if not retried and "embedding_cache" in str(exc).lower():
                logger.info("Embedding cache table missing on write; creating now.")
                await session.rollback()
                await self._ensure_embedding_cache_table(session)
                await session.commit()
                await self._embedding_cache_set(session, key, value, expires_at, retried=True)
            else:
                raise
    # ---------------- Run helpers ----------------
    async def get_run_id_for_upload(self, csv_upload_id: str) -> Optional[str]:
        try:
            async with self.get_session() as session:
                from database import CsvUpload
                cu = await session.get(CsvUpload, csv_upload_id)
                return cu.run_id if cu else None
        except Exception:
            return None
    
    async def get_shop_id_for_upload(self, csv_upload_id: str) -> str:
        """Resolve the shop identifier for a CSV upload (defaults applied)."""
        async with self.get_session() as session:
            result = await session.execute(
                select(CsvUpload.shop_id).where(CsvUpload.id == csv_upload_id)
            )
            value = result.scalar()
            # Return sanitized shop_id or None - don't fallback to DEFAULT
            return sanitize_shop_id(value)
    
    def _overwrite_all_columns(self, table, stmt):
        """Helper to update all columns except primary key(s) in UPSERT operations"""
        pks = {c.name for c in table.primary_key.columns}
        return {
            c.name: getattr(stmt.excluded, c.name)
            for c in table.columns
            if c.name not in pks and c.name != 'updated_at'  # Include created_at for hard overwrite, exclude only audit timestamps
        }
    
    # CSV Upload operations
    async def create_csv_upload(self, upload_data: Dict[str, Any]) -> CsvUpload:
        """Create new CSV upload record"""
        async with self.get_session() as session:
            payload = dict(upload_data)
            # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
            shop_id = sanitize_shop_id(payload.get("shop_id"))
            if not shop_id:
                raise ValueError("shop_id is required for CSV upload")
            payload["shop_id"] = shop_id
            upload = CsvUpload(**payload)
            session.add(upload)
            await session.commit()
            await session.refresh(upload)
            return upload
    
    async def get_csv_upload(self, upload_id: str) -> Optional[CsvUpload]:
        """Get CSV upload by ID"""
        async with self.get_session() as session:
            return await session.get(CsvUpload, upload_id)
    
    async def get_run_uploads(self, run_id: str) -> List[CsvUpload]:
        """Return all CSV uploads belonging to a run."""
        if not run_id:
            return []
        async with self.get_session() as session:
            query = select(CsvUpload).where(CsvUpload.run_id == run_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_latest_orders_upload_for_run(self, run_id: str) -> Optional[CsvUpload]:
        """Return the most recent orders upload for a given run."""
        if not run_id:
            return None
        async with self.get_session() as session:
            query = (
                select(CsvUpload)
                .where(CsvUpload.run_id == run_id, CsvUpload.csv_type == "orders")
                .order_by(desc(CsvUpload.created_at))
                .limit(1)
            )
            result = await session.execute(query)
            return result.scalars().first()

    async def get_latest_upload_for_run(self, run_id: str, csv_type: str) -> Optional[CsvUpload]:
        """Return the most recent upload of a given type for a run."""
        if not run_id or not csv_type:
            return None
        async with self.get_session() as session:
            query = (
                select(CsvUpload)
                .where(CsvUpload.run_id == run_id, CsvUpload.csv_type == csv_type)
                .order_by(desc(CsvUpload.created_at))
                .limit(1)
            )
            result = await session.execute(query)
            return result.scalars().first()
    
    async def update_csv_upload(self, upload_id: str, updates: Dict[str, Any]) -> Optional[CsvUpload]:
        """Update CSV upload record"""
        async with self.get_session() as session:
            upload = await session.get(CsvUpload, upload_id)
            if upload:
                change_set = dict(updates)
                if "shop_id" in change_set:
                    # Use new shop_id if valid, else keep existing - NEVER fallback to DEFAULT
                    new_shop_id = sanitize_shop_id(change_set.get("shop_id"))
                    existing_shop_id = sanitize_shop_id(upload.shop_id)
                    final_shop_id = new_shop_id or existing_shop_id
                    if not final_shop_id:
                        raise ValueError("Cannot update to empty shop_id")
                    change_set["shop_id"] = final_shop_id
                for key, value in change_set.items():
                    setattr(upload, key, value)
                await session.commit()
                await session.refresh(upload)
            return upload

    async def update_csv_upload_status(
        self,
        csv_upload_id: str,
        status: str,
        error_message: Optional[str] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[CsvUpload]:
        """Update upload status with optional error and metrics payload."""
        if not csv_upload_id:
            return None
        updates: Dict[str, Any] = {"status": status}
        if error_message is not None:
            updates["error_message"] = error_message
        if extra_metrics is not None:
            updates["bundle_generation_metrics"] = extra_metrics
        return await self.update_csv_upload(csv_upload_id, updates)
    
    async def get_recent_uploads(self, limit: int = 10) -> List[CsvUpload]:
        """Get recent CSV uploads"""
        async with self.get_session() as session:
            query = select(CsvUpload).order_by(desc(CsvUpload.created_at)).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def safe_mark_upload_completed(self, upload_id: str) -> bool:
        """Atomically transition an upload to 'bundle_generation_completed' if not already terminal."""
        if not upload_id:
            return False

        terminal_statuses = (
            "bundle_generation_failed",
            "bundle_generation_timed_out",
            "bundle_generation_cancelled",
        )

        async with self.get_session() as session:
            stmt = (
                update(CsvUpload)
                .where(
                    CsvUpload.id == upload_id,
                    CsvUpload.status.notin_(terminal_statuses),
                )
                .values(status="bundle_generation_completed", updated_at=func.now())
            )
            result = await session.execute(stmt)
            if result.rowcount:
                await session.commit()
                return True

            await session.rollback()
            return False
    
    # Order operations
    async def create_orders(self, orders_data: List[Dict[str, Any]]) -> List[Order]:
        """Bulk upsert orders - overwrite existing orders with same order_id"""
        if not orders_data:
            return []
        # NEW: sanitize + filter
        orders_data = [self._sanitize_order(o) for o in orders_data]
        orders_data = self._filter_columns(Order.__table__, orders_data)

        async with self.get_session() as session:
            stmt = pg_insert(Order).values(orders_data)
            upsert = stmt.on_conflict_do_update(
                index_elements=[Order.order_id],
                set_=self._overwrite_all_columns(Order.__table__, stmt)
            )
            await session.execute(upsert)
            await session.commit()
            return []
    
    async def hard_overwrite_orders_with_lines(self, orders_data: List[Dict[str, Any]], order_lines_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Transactional hard overwrite of orders with their lines - truly 'rewrite' orders as requested
        
        This method implements the user's requirement: 'if the order already exist in DB just rewrite it'
        by deleting existing OrderLine rows before inserting new ones, ensuring no stale data remains.
        """
        if not orders_data:
            return {"orders_upserted": 0, "order_lines_upserted": 0, "order_lines_skipped": 0}

        # NEW: sanitize + filter orders
        orders_data = [self._sanitize_order(o) for o in orders_data]
        orders_data = self._filter_columns(Order.__table__, orders_data)

        # NEW: sanitize lines, drop invalid
        sanitized_lines = []
        skipped = 0
        for ln in (order_lines_data or []):
            ln_s = self._sanitize_order_line(dict(ln))
            if ln_s is None:
                skipped += 1
                continue
            sanitized_lines.append(ln_s)
        order_lines_data = self._filter_columns(OrderLine.__table__, sanitized_lines)
            
        async with self.get_session() as session:
            try:
                # Extract unique order_ids from incoming data
                incoming_order_ids = list({order.get("order_id") for order in orders_data if order.get("order_id")})
                
                if incoming_order_ids:
                    # Step 1: Delete ALL existing OrderLine rows for these orders (true hard overwrite)
                    # Remove csv_upload_id filter to ensure we delete ALL lines for these orders,
                    # not just lines from the current upload, implementing true "rewrite" semantics
                    delete_lines_stmt = delete(OrderLine).where(OrderLine.order_id.in_(incoming_order_ids))
                    
                    await session.execute(delete_lines_stmt)
                
                # Step 2: Upsert Order headers
                orders_stmt = pg_insert(Order).values(orders_data)
                orders_upsert = orders_stmt.on_conflict_do_update(
                    index_elements=[Order.order_id],
                    set_=self._overwrite_all_columns(Order.__table__, orders_stmt)
                )
                await session.execute(orders_upsert)
                
                # Step 3: Insert OrderLines (no conflicts since we deleted all lines for these orders)
                order_lines_count = 0
                if order_lines_data:
                    # Use plain INSERT since we deleted all existing lines in step 1
                    lines_stmt = pg_insert(OrderLine).values(order_lines_data)
                    await session.execute(lines_stmt)
                    order_lines_count = len(order_lines_data)
                
                # Commit the entire transaction
                await session.commit()
                
                return {
                    "orders_upserted": len(orders_data),
                    "order_lines_upserted": order_lines_count,
                    "order_lines_skipped": skipped
                }
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error in hard_overwrite_orders_with_lines: {e}")
                raise
    
    async def create_order_lines(self, order_lines_data: List[Dict[str, Any]]) -> List[OrderLine]:
        """
        Adaptive bulk upsert for order lines.
        Auto-scales from 100 rows to 100,000+ rows with optimal batching strategy.
        Handles both NULL and NOT NULL line_item_id cases with partial indexes.
        """
        if not order_lines_data:
            return []

        t0 = time.time()

        # Sanitize + filter + drop invalid
        sanitized = []
        skipped = 0
        for ln in order_lines_data:
            ln_s = self._sanitize_order_line(dict(ln))
            if ln_s is not None:
                sanitized.append(ln_s)
            else:
                skipped += 1
        order_lines_data = self._filter_columns(OrderLine.__table__, sanitized)
        if not order_lines_data:
            if skipped > 0:
                logger.debug(f"create_order_lines: All {skipped} lines were invalid and skipped")
            return []

        if skipped > 0:
            logger.debug(f"create_order_lines: Processed {len(order_lines_data)} lines, skipped {skipped} invalid lines")

        # Auto-configure batching based on dataset size
        config = BatchConfig.get_config(len(order_lines_data))

        logger.info(
            f"Storage: create_order_lines rows={len(order_lines_data)} "
            f"tier={config['tier']} batch_size={config['batch_size']}"
        )

        # Separate data by line_item_id NULL vs NOT NULL cases for different conflict targets
        lines_with_item_id = [line for line in order_lines_data if line.get('line_item_id') is not None]
        lines_without_item_id = [line for line in order_lines_data if line.get('line_item_id') is None]

        batch_size = config['batch_size']
        commit_per_batch = config['commit_per_batch']
        batches_processed = 0

        async with self.get_session() as session:
            # Handle lines with line_item_id (NOT NULL case)
            if lines_with_item_id:
                for i in range(0, len(lines_with_item_id), batch_size):
                    batch = lines_with_item_id[i:i + batch_size]

                    stmt = pg_insert(OrderLine).values(batch)
                    # Use partial index with WHERE predicate for ON CONFLICT
                    upsert = stmt.on_conflict_do_update(
                        index_elements=[OrderLine.order_id, OrderLine.line_item_id],
                        index_where=OrderLine.line_item_id.is_not(None),
                        set_=self._overwrite_all_columns(OrderLine.__table__, stmt)
                    )
                    await session.execute(upsert)

                    if commit_per_batch:
                        await session.commit()
                        batches_processed += 1
                        if batches_processed % 5 == 0:
                            logger.info(
                                f"Order lines (with item_id) progress: batch {batches_processed}, "
                                f"{i + len(batch)}/{len(lines_with_item_id)} rows"
                            )

            # Handle lines without line_item_id (NULL case)
            # IMPORTANT: Deduplicate by (order_id, sku) to avoid "cannot affect row a second time" error
            # Keep the row with highest quantity when duplicates exist
            if lines_without_item_id:
                original_count = len(lines_without_item_id)
                deduped_lines: Dict[tuple, Dict[str, Any]] = {}
                for line in lines_without_item_id:
                    key = (line.get('order_id'), line.get('sku'))
                    existing = deduped_lines.get(key)
                    if existing is None:
                        deduped_lines[key] = line
                    else:
                        # Keep the one with higher quantity, or merge quantities
                        existing_qty = existing.get('quantity', 0) or 0
                        new_qty = line.get('quantity', 0) or 0
                        if new_qty > existing_qty:
                            deduped_lines[key] = line

                lines_without_item_id = list(deduped_lines.values())
                duplicates_removed = original_count - len(lines_without_item_id)
                if duplicates_removed > 0:
                    logger.info(f"Deduplicated {duplicates_removed} duplicate order lines by (order_id, sku)")

                for i in range(0, len(lines_without_item_id), batch_size):
                    batch = lines_without_item_id[i:i + batch_size]

                    stmt = pg_insert(OrderLine).values(batch)
                    # Use partial index with WHERE predicate for ON CONFLICT
                    upsert = stmt.on_conflict_do_update(
                        index_elements=[OrderLine.order_id, OrderLine.sku],
                        index_where=OrderLine.line_item_id.is_(None),
                        set_=self._overwrite_all_columns(OrderLine.__table__, stmt)
                    )
                    await session.execute(upsert)

                    if commit_per_batch:
                        await session.commit()
                        batches_processed += 1
                        if batches_processed % 5 == 0:
                            logger.info(
                                f"Order lines (without item_id) progress: batch {batches_processed}, "
                                f"{i + len(batch)}/{len(lines_without_item_id)} rows"
                            )

            if not commit_per_batch:
                await session.commit()

        duration_ms = (time.time() - t0) * 1000
        ms_per_row = duration_ms / len(order_lines_data) if order_lines_data else 0
        logger.info(
            f"✅ Order lines ingestion complete: {len(order_lines_data)} rows in {duration_ms:.0f}ms "
            f"({ms_per_row:.1f}ms per row, tier={config['tier']})"
        )
        return []
    
    async def get_orders(self, csv_upload_id: str, limit: Optional[int] = None) -> List[Order]:
        """Get orders for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(Order).where(Order.csv_upload_id == csv_upload_id)
            if limit:
                query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    @retry_async(max_retries=3, base_delay=1.0, max_delay=10.0)
    async def get_order_lines(self, csv_upload_id: str) -> List[OrderLine]:
        """Get order lines for a specific CSV upload.

        Retries on transient database errors (connection issues, timeouts).
        """
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(OrderLine).where(OrderLine.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Product operations
    async def create_products(self, products_data: List[Dict[str, Any]]) -> List[Product]:
        """Bulk upsert products - overwrite existing products with same sku"""
        if not products_data:
            return []
        async with self.get_session() as session:
            stmt = pg_insert(Product).values(products_data)
            upsert = stmt.on_conflict_do_update(
                index_elements=[Product.sku],
                set_=self._overwrite_all_columns(Product.__table__, stmt)
            )
            await session.execute(upsert)
            await session.commit()
            return []
    
    async def get_products(self) -> List[Product]:
        """Get all products"""
        async with self.get_session() as session:
            query = select(Product)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Association Rules operations
    async def create_association_rules(self, rules_data: List[Dict[str, Any]]) -> List[AssociationRule]:
        """Bulk create association rules"""
        async with self.get_session() as session:
            rules = [AssociationRule(**rule_data) for rule_data in rules_data]
            session.add_all(rules)
            await session.commit()
            return rules
    
    async def get_association_rules(self, csv_upload_id: str, limit: int = 100) -> List[AssociationRule]:
        """Get association rules for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(AssociationRule).where(AssociationRule.csv_upload_id == csv_upload_id)
            query = query.order_by(desc(AssociationRule.lift)).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_association_rules_by_run(self, run_id: str, limit: int = 100) -> List[AssociationRule]:
        """Get association rules for a whole run (any upload in the run)."""
        if not run_id:
            return []
        async with self.get_session() as session:
            query = (
                select(AssociationRule)
                .join(CsvUpload, AssociationRule.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
                .order_by(desc(AssociationRule.lift))
                .limit(limit)
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def clear_association_rules(self, csv_upload_id: str) -> None:
        """Clear association rules for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = delete(AssociationRule).where(AssociationRule.csv_upload_id == csv_upload_id)
            await session.execute(query)
            await session.commit()
    
    # Bundle Recommendations operations
    async def create_bundle_recommendations(self, recommendations_data: List[Dict[str, Any]]) -> List[BundleRecommendation]:
        """
        Adaptive bulk create for bundle recommendations.
        Auto-scales from 10 bundles to 10,000+ bundles with optimal batching strategy.
        """
        if not recommendations_data:
            return []

        t0 = time.time()

        # Auto-configure batching based on dataset size
        config = BatchConfig.get_config(len(recommendations_data))

        logger.info(
            f"Storage: create_bundle_recommendations bundles={len(recommendations_data)} "
            f"tier={config['tier']} batch_size={config['batch_size']}"
        )

        async with self.get_session() as session:
            # Ensure each recommendation carries the owning shop_id
            upload_ids = {
                rec.get("csv_upload_id")
                for rec in recommendations_data
                if rec.get("csv_upload_id")
            }
            shop_by_upload: Dict[str, Optional[str]] = {}
            if upload_ids:
                result = await session.execute(
                    select(CsvUpload.id, CsvUpload.shop_id).where(CsvUpload.id.in_(upload_ids))
                )
                shop_by_upload = {row.id: row.shop_id for row in result.all()}

            normalized_rows: List[Dict[str, Any]] = []
            missing_shop_ids: set[str] = set()
            missing_upload_refs: set[str] = set()
            for rec in recommendations_data:
                row = rec.copy()
                upload_id = row.get("csv_upload_id")
                explicit_shop = sanitize_shop_id(row.get("shop_id"))
                fallback_shop = sanitize_shop_id(shop_by_upload.get(upload_id)) if upload_id else None

                # Use explicit shop_id first, then fallback to upload's shop_id
                resolved_shop = explicit_shop or fallback_shop

                # CRITICAL: Never fallback to DEFAULT_SHOP_ID - fail instead
                if not resolved_shop:
                    rec_id = row.get("id", "unknown")
                    missing_shop_ids.add(f"rec_id={rec_id}, upload_id={upload_id}")
                    logger.error(
                        "Bundle recommendation missing shop_id: rec_id=%s, upload_id=%s",
                        rec_id, upload_id
                    )
                    continue  # Skip this bundle - don't save it with wrong shop_id

                if not upload_id:
                    missing_upload_refs.add(str(row.get("id", "")))
                row["shop_id"] = resolved_shop
                normalized_rows.append(row)

            if missing_upload_refs:
                logger.warning(
                    "Bundle recommendations missing csv_upload_id; rec_ids=%s",
                    sorted(r for r in missing_upload_refs if r)
                )
            if missing_shop_ids:
                logger.error(
                    "Skipped bundle recommendations with no shop_id: %s",
                    sorted(missing_shop_ids)
                )

            filtered_rows = self._filter_columns(BundleRecommendation.__table__, normalized_rows)

            # Convert Decimal values to float in JSON fields to prevent serialization errors
            for row in filtered_rows:
                for json_field in ['products', 'pricing', 'ai_copy']:
                    if json_field in row and row[json_field]:
                        row[json_field] = self._convert_decimals_to_float(row[json_field])

            # Validate bundles and log warnings for data quality issues
            if validate_bundle:
                validation_warnings = 0
                for row in filtered_rows:
                    is_valid, errors = validate_bundle(row)
                    if not is_valid:
                        validation_warnings += 1
                        if validation_warnings <= 3:  # Log first 3 warnings in detail
                            logger.warning(
                                f"Bundle validation warning (id={row.get('id', 'unknown')}): {errors[:3]}"
                            )
                if validation_warnings > 0:
                    logger.warning(
                        f"Bundle validation: {validation_warnings}/{len(filtered_rows)} bundles have data quality issues"
                    )

            # Remove any existing recommendations with matching IDs to support upsert-like behaviour
            existing_ids = [row.get("id") for row in filtered_rows if row.get("id")]
            if existing_ids:
                await session.execute(
                    delete(BundleRecommendation).where(BundleRecommendation.id.in_(existing_ids))
                )

            # Adaptive batched save
            batch_size = config['batch_size']
            commit_per_batch = config['commit_per_batch']
            batches_processed = 0

            for i in range(0, len(filtered_rows), batch_size):
                batch = filtered_rows[i:i + batch_size]
                recommendations = [BundleRecommendation(**row) for row in batch]
                session.add_all(recommendations)

                if commit_per_batch:
                    await session.commit()
                    batches_processed += 1
                    if batches_processed % 5 == 0:
                        logger.info(
                            f"Bundle save progress: batch {batches_processed}, "
                            f"{i + len(batch)}/{len(filtered_rows)} bundles saved"
                        )

            if not commit_per_batch:
                await session.commit()

            # ❌ REMOVED: Individual refresh() loop - was 93% of save time
            # Each refresh = 1 separate SELECT query = 500ms network latency per bundle
            # 40 bundles × 500ms = 20+ seconds wasted
            # ✅ Caller doesn't use return value anyway - return empty list

        duration_ms = (time.time() - t0) * 1000
        ms_per_bundle = duration_ms / len(recommendations_data) if recommendations_data else 0
        logger.info(
            f"✅ Bundle recommendations save complete: {len(recommendations_data)} bundles in {duration_ms:.0f}ms "
            f"({ms_per_bundle:.1f}ms per bundle, tier={config['tier']})"
        )
        return []

    async def delete_partial_bundle_recommendations(self, csv_upload_id: str) -> None:
        """Remove any partially persisted recommendations for an upload."""
        if not csv_upload_id:
            return
        async with self.get_session() as session:
            await session.execute(
                delete(BundleRecommendation).where(
                    BundleRecommendation.csv_upload_id == csv_upload_id,
                    BundleRecommendation.discount_reference.isnot(None),
                    BundleRecommendation.discount_reference.like("__partial__%"),
                )
            )
            await session.commit()

    async def get_partial_bundle_recommendations(self, csv_upload_id: str) -> List[BundleRecommendation]:
        """Fetch partially persisted recommendations used for resume checkpoints."""
        if not csv_upload_id:
            return []
        async with self.get_session() as session:
            query = (
                select(BundleRecommendation)
                .where(
                    BundleRecommendation.csv_upload_id == csv_upload_id,
                    BundleRecommendation.discount_reference.isnot(None),
                    BundleRecommendation.discount_reference.like("__partial__%"),
                )
                .order_by(
                    BundleRecommendation.rank_position.asc().nulls_last(),
                    desc(BundleRecommendation.ranking_score),
                    desc(BundleRecommendation.confidence),
                    desc(BundleRecommendation.created_at),
                )
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_bundle_recommendations_by_upload(
        self,
        csv_upload_id: str,
        shop_id: Optional[str] = None,
        limit: int = 50
    ) -> List[BundleRecommendation]:
        """Get bundle recommendations for a specific CSV upload with shop scoping."""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")

        async with self.get_session() as session:
            query = select(BundleRecommendation).where(BundleRecommendation.csv_upload_id == csv_upload_id)

            shop_filter = sanitize_shop_id(shop_id)
            if shop_filter:
                query = query.where(BundleRecommendation.shop_id == shop_filter)
            else:
                query = query.join(CsvUpload, BundleRecommendation.csv_upload_id == CsvUpload.id)
                query = query.where(BundleRecommendation.shop_id == CsvUpload.shop_id)

            query = query.order_by(
                BundleRecommendation.rank_position.asc().nulls_last(),
                desc(BundleRecommendation.ranking_score),
                desc(BundleRecommendation.confidence),
                desc(BundleRecommendation.created_at)
            ).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_bundle_recommendations(
        self,
        shop_id: str,
        csv_upload_id: Optional[str] = None,
        limit: int = 50
    ) -> List[BundleRecommendation]:
        """Get bundle recommendations scoped to a shop, optionally filtered by upload."""
        # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
        normalized_shop_id = sanitize_shop_id(shop_id)
        if not normalized_shop_id:
            raise ValueError("shop_id is required")

        async with self.get_session() as session:
            query = select(BundleRecommendation).where(BundleRecommendation.shop_id == normalized_shop_id)
            if csv_upload_id:
                query = query.where(BundleRecommendation.csv_upload_id == csv_upload_id)
                query = query.order_by(
                    BundleRecommendation.rank_position.asc().nulls_last(),
                    desc(BundleRecommendation.ranking_score),
                    desc(BundleRecommendation.confidence),
                    desc(BundleRecommendation.created_at)
                )
            else:
                query = query.order_by(
                    desc(BundleRecommendation.ranking_score),
                    desc(BundleRecommendation.confidence),
                    desc(BundleRecommendation.created_at)
                )

            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def update_bundle_recommendation(self, rec_id: str, updates: Dict[str, Any]) -> Optional[BundleRecommendation]:
        """Update bundle recommendation"""
        async with self.get_session() as session:
            recommendation = await session.get(BundleRecommendation, rec_id)
            if recommendation:
                for key, value in updates.items():
                    setattr(recommendation, key, value)
                await session.commit()
                await session.refresh(recommendation)
            return recommendation
    
    async def clear_bundle_recommendations(self, csv_upload_id: str, shop_id: Optional[str] = None) -> None:
        """Clear bundle recommendations for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = delete(BundleRecommendation).where(BundleRecommendation.csv_upload_id == csv_upload_id)
            resolved_shop = sanitize_shop_id(shop_id)
            if resolved_shop:
                query = query.where(BundleRecommendation.shop_id == resolved_shop)
            await session.execute(query)
            await session.commit()

    async def get_bundle_count_for_upload(self, csv_upload_id: str) -> int:
        """Get the actual count of bundle recommendations from DB for an upload.

        This is used for accurate progress reporting after saving bundles,
        rather than relying on in-memory counts which may differ due to
        deduplication, filtering, or save failures.
        """
        if not csv_upload_id:
            return 0

        async with self.get_session() as session:
            query = (
                select(func.count())
                .select_from(BundleRecommendation)
                .where(BundleRecommendation.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            count = result.scalar_one_or_none() or 0
            return int(count)

    # Bundle operations
    async def create_bundle(self, bundle_data: Dict[str, Any]) -> Bundle:
        """Create new bundle"""
        async with self.get_session() as session:
            bundle = Bundle(**bundle_data)
            session.add(bundle)
            await session.commit()
            await session.refresh(bundle)
            return bundle
    
    async def get_bundles(self) -> List[Bundle]:
        """Get all bundles"""
        async with self.get_session() as session:
            query = select(Bundle).order_by(desc(Bundle.created_at))
            result = await session.execute(query)
            return list(result.scalars().all())

    # Shop sync status operations
    async def get_shop_sync_status(self, shop_id: str) -> Optional[ShopSyncStatus]:
        """Retrieve sync status for a shop."""
        # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
        normalized_shop_id = sanitize_shop_id(shop_id)
        if not normalized_shop_id:
            raise ValueError("shop_id is required")
        async with self.get_session() as session:
            return await session.get(ShopSyncStatus, normalized_shop_id)

    async def mark_shop_sync_started(self, shop_id: str) -> ShopSyncStatus:
        """Upsert sync status when ingestion begins."""
        # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
        normalized_shop_id = sanitize_shop_id(shop_id)
        if not normalized_shop_id:
            raise ValueError("shop_id is required")
        async with self.get_session() as session:
            status = await session.get(ShopSyncStatus, normalized_shop_id)
            now = datetime.utcnow()
            if not status:
                status = ShopSyncStatus(
                    shop_id=normalized_shop_id,
                    initial_sync_completed=False,
                    last_sync_started_at=now,
                    last_sync_completed_at=None,
                )
                session.add(status)
            else:
                status.last_sync_started_at = now
                status.initial_sync_completed = False
            await session.commit()
            await session.refresh(status)
            return status

    async def mark_shop_sync_completed(self, shop_id: str) -> ShopSyncStatus:
        """Mark the initial ingestion as complete for a shop."""
        # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
        normalized_shop_id = sanitize_shop_id(shop_id)
        if not normalized_shop_id:
            raise ValueError("shop_id is required")
        async with self.get_session() as session:
            status = await session.get(ShopSyncStatus, normalized_shop_id)
            now = datetime.utcnow()
            if not status:
                status = ShopSyncStatus(
                    shop_id=normalized_shop_id,
                    initial_sync_completed=True,
                    last_sync_started_at=None,
                    last_sync_completed_at=now,
                )
                session.add(status)
            else:
                status.initial_sync_completed = True
                status.last_sync_completed_at = now
            await session.commit()
            await session.refresh(status)
            return status

    async def is_first_time_install(self, shop_id: str) -> bool:
        """Check if this is a first-time installation for a shop.

        Returns True if:
        - No sync status exists for the shop
        - Sync status exists but initial_sync_completed is False

        Returns False if initial_sync_completed is True (regular user)
        """
        # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
        normalized_shop_id = sanitize_shop_id(shop_id)
        if not normalized_shop_id:
            raise ValueError("shop_id is required")
        async with self.get_session() as session:
            status = await session.get(ShopSyncStatus, normalized_shop_id)
            if not status:
                # No status record = first time
                return True
            # Check if initial sync has been completed
            return not status.initial_sync_completed

    async def get_quick_start_preflight_info(self, csv_upload_id: str, shop_id: str) -> Dict[str, Any]:
        """Consolidated pre-flight check for quick-start bundle generation.

        Performs a single optimized query to gather all information needed for quick-start decision:
        - is_first_time_install: Whether this is a first-time installation
        - has_existing_quick_start: Whether quick-start bundles already exist for this upload
        - csv_upload_status: Current status of the CSV upload

        This reduces database round-trips from 3+ queries to 1 query.
        Results are cached for 60 seconds to handle rapid retries.

        Returns:
            Dict with keys:
            - is_first_time_install (bool)
            - has_existing_quick_start (bool)
            - quick_start_bundle_count (int)
            - csv_upload_status (str or None)
        """
        import traceback
        step_start = time.time()
        logger.info(f"[{csv_upload_id}] 🔍 Pre-flight check STARTING | shop_id={shop_id}")
        
        try:
            # Check cache first
            now = time.time()
            if csv_upload_id in self._preflight_cache:
                cached_result, cached_time = self._preflight_cache[csv_upload_id]
                if now - cached_time < self._PREFLIGHT_CACHE_TTL:
                    logger.info(
                        f"[{csv_upload_id}] ✅ Pre-flight check cache HIT "
                        f"(age: {now - cached_time:.1f}s) | result={cached_result}"
                    )
                    return cached_result
                else:
                    # Cache expired, remove it
                    del self._preflight_cache[csv_upload_id]
                    logger.info(f"[{csv_upload_id}] Pre-flight cache expired, fetching fresh data")

            # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
            normalized_shop_id = sanitize_shop_id(shop_id)
            if not normalized_shop_id:
                logger.error(f"[{csv_upload_id}] Pre-flight failed: invalid shop_id")
                raise ValueError("shop_id is required")
            logger.info(f"[{csv_upload_id}] Pre-flight: normalized_shop_id={normalized_shop_id}")

            async with self.get_session() as session:
                logger.info(f"[{csv_upload_id}] Pre-flight: got session, executing query...")
                query_start = time.time()
                
                # Single query to get all pre-flight info using a SQL query with JOINs
                query = text("""
                    SELECT
                        COALESCE(sss.initial_sync_completed, FALSE) as sync_completed,
                        cu.status as upload_status,
                        COUNT(br.id) FILTER (WHERE br.discount_reference LIKE :quick_start_pattern) as quick_start_count
                    FROM csv_uploads cu
                    LEFT JOIN shop_sync_status sss ON sss.shop_id = cu.shop_id
                    LEFT JOIN bundle_recommendations br ON br.csv_upload_id = cu.id
                    WHERE cu.id = :csv_upload_id
                    GROUP BY sss.initial_sync_completed, cu.status
                """)

                result = await session.execute(
                    query,
                    {
                        "csv_upload_id": csv_upload_id,
                        "quick_start_pattern": "__quick_start_%"
                    }
                )
                query_duration = (time.time() - query_start) * 1000
                logger.info(f"[{csv_upload_id}] Pre-flight: query executed in {query_duration:.1f}ms")
                
                row = result.fetchone()
                logger.info(f"[{csv_upload_id}] Pre-flight: query result row={row}")

                if not row:
                    # CSV upload not found - assume first install, no existing bundles
                    logger.warning(f"[{csv_upload_id}] Pre-flight: NO ROW FOUND - CSV upload missing from database!")
                    return {
                        "is_first_time_install": True,
                        "has_existing_quick_start": False,
                        "quick_start_bundle_count": 0,
                        "csv_upload_status": None,
                    }

                sync_completed = row[0] if row[0] is not None else False
                upload_status = row[1]
                quick_start_count = row[2] if row[2] is not None else 0
                
                logger.info(
                    f"[{csv_upload_id}] Pre-flight: parsed values | "
                    f"sync_completed={sync_completed}, upload_status={upload_status}, quick_start_count={quick_start_count}"
                )

                result_dict = {
                    "is_first_time_install": not sync_completed,
                    "has_existing_quick_start": quick_start_count > 0,
                    "quick_start_bundle_count": int(quick_start_count),
                    "csv_upload_status": upload_status,
                }

                # Store in cache
                self._preflight_cache[csv_upload_id] = (result_dict, time.time())
                total_duration = (time.time() - step_start) * 1000
                logger.info(
                    f"[{csv_upload_id}] ✅ Pre-flight check COMPLETE in {total_duration:.1f}ms | result={result_dict}"
                )

                return result_dict
                
        except Exception as e:
            total_duration = (time.time() - step_start) * 1000
            logger.error(
                f"[{csv_upload_id}] ❌ Pre-flight check FAILED after {total_duration:.1f}ms!\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

    async def backfill_bundle_recommendation_shop_ids(self) -> int:
        """Populate missing bundle_recommendations.shop_id values from their CSV uploads."""
        async with self.get_session() as session:
            result = await session.execute(
                text(
                    """
                    UPDATE bundle_recommendations br
                    SET shop_id = cu.shop_id
                    FROM csv_uploads cu
                    WHERE br.csv_upload_id = cu.id
                      AND br.shop_id IS NULL
                      AND cu.shop_id IS NOT NULL
                    """
                )
            )
            await session.commit()
            return result.rowcount or 0
    
    # Variant operations
    async def create_variants(self, variants_data: List[Dict[str, Any]]) -> List[Variant]:
        """Bulk upsert variants - overwrite existing variants with same variant_id"""
        if not variants_data:
            return []
        async with self.get_session() as session:
            stmt = pg_insert(Variant).values(variants_data)
            upsert = stmt.on_conflict_do_update(
                index_elements=[Variant.variant_id],
                set_=self._overwrite_all_columns(Variant.__table__, stmt)
            )
            await session.execute(upsert)
            await session.commit()
            return []
    
    async def get_variants(self, csv_upload_id: str) -> List[Variant]:
        """Get variants for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(Variant).where(Variant.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def count_variant_and_catalog_records(
        self, csv_upload_id: Optional[str], run_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Count variants and catalog snapshots for an upload or run."""
        if not csv_upload_id and not run_id:
            raise ValueError("csv_upload_id or run_id is required")

        async with self.get_session() as session:
            if run_id:
                variant_query = (
                    select(func.count())
                    .select_from(Variant)
                    .join(CsvUpload, Variant.csv_upload_id == CsvUpload.id)
                    .where(CsvUpload.run_id == run_id)
                )
                catalog_query = (
                    select(func.count())
                    .select_from(CatalogSnapshot)
                    .join(CsvUpload, CatalogSnapshot.csv_upload_id == CsvUpload.id)
                    .where(CsvUpload.run_id == run_id)
                )
            else:
                variant_query = select(func.count()).select_from(Variant).where(Variant.csv_upload_id == csv_upload_id)
                catalog_query = (
                    select(func.count()).select_from(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
                )

            variant_count = (await session.execute(variant_query)).scalar_one() or 0
            catalog_count = (await session.execute(catalog_query)).scalar_one() or 0
            return {"variant_count": int(variant_count), "catalog_count": int(catalog_count)}

    async def _count_orders_and_lines(
        self, csv_upload_id: Optional[str], run_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Count orders and order_lines scoped to an upload or run."""
        if not csv_upload_id and not run_id:
            raise ValueError("csv_upload_id or run_id is required")

        async with self.get_session() as session:
            if run_id:
                orders_query = (
                    select(func.count())
                    .select_from(Order)
                    .join(CsvUpload, Order.csv_upload_id == CsvUpload.id)
                    .where(CsvUpload.run_id == run_id)
                )
                order_lines_query = (
                    select(func.count())
                    .select_from(OrderLine)
                    .join(CsvUpload, OrderLine.csv_upload_id == CsvUpload.id)
                    .where(CsvUpload.run_id == run_id)
                )
            else:
                orders_query = select(func.count()).select_from(Order).where(Order.csv_upload_id == csv_upload_id)
                order_lines_query = (
                    select(func.count())
                    .select_from(OrderLine)
                    .where(OrderLine.csv_upload_id == csv_upload_id)
                )

            order_count = (await session.execute(orders_query)).scalar_one() or 0
            order_line_count = (await session.execute(order_lines_query)).scalar_one() or 0
            return {"order_count": int(order_count), "order_line_count": int(order_line_count)}

    async def summarize_upload_coverage(
        self, csv_upload_id: str, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize ingestion coverage for an upload/run: orders, order_lines, variants, catalog.
        """
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")

        resolved_run_id = run_id or await self.get_run_id_for_upload(csv_upload_id)
        orders = await self._count_orders_and_lines(csv_upload_id, resolved_run_id)
        variants = await self.count_variant_and_catalog_records(csv_upload_id, resolved_run_id)

        summary = {
            "csv_upload_id": csv_upload_id,
            "run_id": resolved_run_id,
            "orders": orders.get("order_count", 0),
            "order_lines": orders.get("order_line_count", 0),
            "variants": variants.get("variant_count", 0),
            "catalog": variants.get("catalog_count", 0),
        }
        if summary["variants"] > 0:
            summary["missing_catalog_variants"] = max(summary["variants"] - summary["catalog"], 0)
            summary["catalog_coverage_ratio"] = (
                summary["catalog"] / summary["variants"] if summary["variants"] else 0.0
            )
        else:
            summary["missing_catalog_variants"] = 0
            summary["catalog_coverage_ratio"] = 0.0
        logger.info(
            "[%s] Coverage summary | run=%s orders=%d order_lines=%d variants=%d catalog=%d",
            csv_upload_id,
            resolved_run_id,
            summary["orders"],
            summary["order_lines"],
            summary["variants"],
            summary["catalog"],
        )
        return summary

    async def backfill_order_line_skus_from_variants(
        self, csv_upload_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> int:
        """Populate missing order_lines.sku from variants when variant_id matches."""
        if not csv_upload_id and not run_id:
            raise ValueError("csv_upload_id or run_id is required")

        async with self.get_session() as session:
            if run_id:
                stmt = text(
                    """
                    UPDATE order_lines ol
                    SET sku = v.sku
                    FROM variants v
                    JOIN csv_uploads cu_v ON cu_v.id = v.csv_upload_id
                    JOIN csv_uploads cu_ol ON cu_ol.id = ol.csv_upload_id
                    WHERE ol.variant_id = v.variant_id
                      AND cu_v.run_id = :run_id
                      AND cu_ol.run_id = :run_id
                      AND (ol.sku IS NULL OR TRIM(ol.sku) = '')
                      AND v.sku IS NOT NULL
                    """
                )
                params = {"run_id": run_id}
            else:
                stmt = text(
                    """
                    UPDATE order_lines ol
                    SET sku = v.sku
                    FROM variants v
                    WHERE ol.variant_id = v.variant_id
                      AND ol.csv_upload_id = :csv_upload_id
                      AND v.csv_upload_id = :csv_upload_id
                      AND (ol.sku IS NULL OR TRIM(ol.sku) = '')
                      AND v.sku IS NOT NULL
                    """
                )
                params = {"csv_upload_id": csv_upload_id}

            result = await session.execute(stmt, params)
            await session.commit()
            updated = result.rowcount or 0
            if updated:
                logger.info(
                    "Backfilled %d order_lines.sku from variants (upload=%s run=%s)",
                    updated,
                    csv_upload_id,
                    run_id,
                )
            return int(updated)

    async def get_variant_maps(self, csv_upload_id: str) -> Tuple[Dict[str, Variant], Dict[str, Variant]]:
        """Return variants keyed by SKU and variant_id for a CSV upload."""
        variants = await self.get_variants(csv_upload_id)
        by_sku: Dict[str, Variant] = {}
        by_id: Dict[str, Variant] = {}
        for variant in variants:
            sku = getattr(variant, "sku", None)
            if sku:
                by_sku[sku] = variant
            variant_id = getattr(variant, "variant_id", None)
            if variant_id:
                by_id[variant_id] = variant
        return by_sku, by_id

    async def get_variants_for_run(self, run_id: str) -> List[Variant]:
        """Get variants associated with all uploads in a run."""
        if not run_id:
            return []
        async with self.get_session() as session:
            query = (
                select(Variant)
                .join(CsvUpload, Variant.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_variant_maps_by_run(self, run_id: str) -> Tuple[Dict[str, Variant], Dict[str, Variant]]:
        """Return variants keyed by SKU and variant_id for a run."""
        variants = await self.get_variants_for_run(run_id)
        by_sku: Dict[str, Variant] = {}
        by_id: Dict[str, Variant] = {}
        for variant in variants:
            sku = getattr(variant, "sku", None)
            if sku:
                by_sku[sku] = variant
            variant_id = getattr(variant, "variant_id", None)
            if variant_id:
                by_id[variant_id] = variant
        return by_sku, by_id

    async def get_variant_by_sku_run(self, sku: str, run_id: str) -> Optional[Variant]:
        if not sku or not run_id:
            return None
        async with self.get_session() as session:
            query = (
                select(Variant)
                .join(CsvUpload, Variant.csv_upload_id == CsvUpload.id)
                .where(and_(Variant.sku == sku, CsvUpload.run_id == run_id))
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def get_variant_by_id_run(self, variant_id: str, run_id: str) -> Optional[Variant]:
        if not variant_id or not run_id:
            return None
        async with self.get_session() as session:
            query = (
                select(Variant)
                .join(CsvUpload, Variant.csv_upload_id == CsvUpload.id)
                .where(and_(Variant.variant_id == variant_id, CsvUpload.run_id == run_id))
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    # Inventory Level operations
    async def create_inventory_levels(self, inventory_data: List[Dict[str, Any]]) -> List[InventoryLevel]:
        """
        Adaptive bulk upsert for inventory levels.
        Auto-scales from 100 rows to 100,000+ rows with optimal batching strategy.
        """
        if not inventory_data:
            return []

        t0 = time.time()

        # Auto-configure batching based on dataset size
        config = BatchConfig.get_config(len(inventory_data))

        logger.info(
            f"Storage: create_inventory_levels rows={len(inventory_data)} "
            f"tier={config['tier']} batch_size={config['batch_size']}"
        )

        # Adaptive batched ingestion
        batch_size = config['batch_size']
        commit_per_batch = config['commit_per_batch']
        batches_processed = 0

        async with self.get_session() as session:
            for i in range(0, len(inventory_data), batch_size):
                batch = inventory_data[i:i + batch_size]

                stmt = pg_insert(InventoryLevel).values(batch)
                upsert = stmt.on_conflict_do_update(
                    index_elements=[InventoryLevel.inventory_item_id, InventoryLevel.location_id],
                    set_=self._overwrite_all_columns(InventoryLevel.__table__, stmt)
                )
                await session.execute(upsert)

                if commit_per_batch:
                    await session.commit()
                    batches_processed += 1
                    if batches_processed % 5 == 0:  # Log every 5 batches
                        logger.info(
                            f"Inventory ingestion progress: batch {batches_processed}, "
                            f"{i + len(batch)}/{len(inventory_data)} rows processed"
                        )

            if not commit_per_batch:
                await session.commit()

        duration_ms = (time.time() - t0) * 1000
        ms_per_row = duration_ms / len(inventory_data) if inventory_data else 0
        logger.info(
            f"✅ Inventory ingestion complete: {len(inventory_data)} rows in {duration_ms:.0f}ms "
            f"({ms_per_row:.1f}ms per row, tier={config['tier']})"
        )
        return []
    
    async def get_inventory_levels(self, csv_upload_id: str) -> List[InventoryLevel]:
        """Get inventory levels for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(InventoryLevel).where(InventoryLevel.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_inventory_levels_map(self, csv_upload_id: str) -> Dict[str, List[InventoryLevel]]:
        """Return inventory levels keyed by inventory_item_id for a CSV upload."""
        levels = await self.get_inventory_levels(csv_upload_id)
        inventory_map: Dict[str, List[InventoryLevel]] = {}
        for level in levels:
            item_id = getattr(level, "inventory_item_id", None)
            if not item_id:
                continue
            inventory_map.setdefault(item_id, []).append(level)
        return inventory_map

    async def get_inventory_levels_for_run(self, run_id: str) -> List[InventoryLevel]:
        """Get inventory levels associated with all uploads in a run."""
        if not run_id:
            return []
        async with self.get_session() as session:
            query = (
                select(InventoryLevel)
                .join(CsvUpload, InventoryLevel.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_inventory_levels_map_by_run(self, run_id: str) -> Dict[str, List[InventoryLevel]]:
        """Return inventory levels keyed by inventory_item_id for a run."""
        levels = await self.get_inventory_levels_for_run(run_id)
        inventory_map: Dict[str, List[InventoryLevel]] = {}
        for level in levels:
            item_id = getattr(level, "inventory_item_id", None)
            if not item_id:
                continue
            inventory_map.setdefault(item_id, []).append(level)
        return inventory_map

    async def get_inventory_levels_by_item_id_run(self, inventory_item_id: str, run_id: str) -> List[InventoryLevel]:
        if not inventory_item_id or not run_id:
            return []
        async with self.get_session() as session:
            query = (
                select(InventoryLevel)
                .join(CsvUpload, InventoryLevel.csv_upload_id == CsvUpload.id)
                .where(and_(InventoryLevel.inventory_item_id == inventory_item_id, CsvUpload.run_id == run_id))
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Catalog Snapshot operations
    async def create_catalog_snapshots(self, catalog_data: List[Dict[str, Any]]) -> List[CatalogSnapshot]:
        """
        Adaptive bulk upsert for catalog snapshots.
        Auto-scales from 100 rows to 100,000+ rows with optimal batching strategy.
        """
        if not catalog_data:
            return []

        t0 = time.time()

        # Filter to valid table columns to avoid 'unconsumed column names' errors
        catalog_data = self._filter_columns(CatalogSnapshot.__table__, catalog_data)

        # Deduplicate within the same batch to avoid ON CONFLICT affecting the same row twice
        # Key = (csv_upload_id, variant_id)
        dedup: Dict[tuple, Dict[str, Any]] = {}
        dropped_missing_key = 0
        for row in catalog_data:
            key = (row.get("csv_upload_id"), row.get("variant_id"))
            if not key[0] or not key[1]:
                dropped_missing_key += 1
                continue
            dedup[key] = row  # keep last occurrence
        catalog_data = list(dedup.values())

        try:
            run_id = await self.get_run_id_for_upload(catalog_data[0]["csv_upload_id"]) if catalog_data else None
        except Exception:
            run_id = None

        # Auto-configure batching based on dataset size
        config = BatchConfig.get_config(len(catalog_data))

        logger.info(
            f"Storage: create_catalog_snapshots upload_id={catalog_data[0]['csv_upload_id'] if catalog_data else 'n/a'} "
            f"run_id={run_id} rows_filtered={len(catalog_data)} dropped_missing_key={dropped_missing_key} "
            f"tier={config['tier']} batch_size={config['batch_size']} strategy={config['description']}"
        )

        # Adaptive batched ingestion
        batch_size = config['batch_size']
        commit_per_batch = config['commit_per_batch']
        batches_processed = 0

        async with self.get_session() as session:
            for i in range(0, len(catalog_data), batch_size):
                batch = catalog_data[i:i + batch_size]

                stmt = pg_insert(CatalogSnapshot).values(batch)
                upsert = stmt.on_conflict_do_update(
                    index_elements=[CatalogSnapshot.csv_upload_id, CatalogSnapshot.variant_id],
                    set_=self._overwrite_all_columns(CatalogSnapshot.__table__, stmt)
                )
                await session.execute(upsert)

                if commit_per_batch:
                    await session.commit()
                    batches_processed += 1
                    if batches_processed % 5 == 0:  # Log every 5 batches
                        logger.info(
                            f"Catalog ingestion progress: batch {batches_processed}, "
                            f"{i + len(batch)}/{len(catalog_data)} rows processed"
                        )

            if not commit_per_batch:
                await session.commit()

        duration_ms = (time.time() - t0) * 1000
        ms_per_row = duration_ms / len(catalog_data) if catalog_data else 0
        logger.info(
            f"✅ Catalog ingestion complete: {len(catalog_data)} rows in {duration_ms:.0f}ms "
            f"({ms_per_row:.1f}ms per row, tier={config['tier']})"
        )
        return []
    
    async def get_catalog_snapshots(self, csv_upload_id: str) -> List[CatalogSnapshot]:
        """Get catalog snapshots for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_catalog_snapshots_by_run(self, run_id: str) -> List[CatalogSnapshot]:
        if not run_id:
            return []
        async with self.get_session() as session:
            query = (
                select(CatalogSnapshot)
                .join(CsvUpload, CatalogSnapshot.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def recompute_catalog_objectives(self, csv_upload_id: str) -> None:
        """Recompute is_slow_mover flags for a specific CSV upload"""
        async with self.get_session() as session:
            logger.info(f"Recomputing catalog objectives for upload {csv_upload_id}")
            
            # Get all catalog snapshots for this upload
            catalog_query = select(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
            catalog_result = await session.execute(catalog_query)
            catalog_snapshots = list(catalog_result.scalars().all())
            
            if not catalog_snapshots:
                logger.warning(f"No catalog snapshots found for upload {csv_upload_id}")
                return
            
            # Create variant_id -> catalog mapping for fast lookups
            variant_to_catalog = {snapshot.variant_id: snapshot for snapshot in catalog_snapshots}
            
            # Build SKU -> variant_id fallback mapping
            sku_to_variant = {snapshot.sku: snapshot.variant_id for snapshot in catalog_snapshots if getattr(snapshot, 'sku', None)}
            
            # 90-day lookback window for slow-mover detection
            cutoff_date = datetime.now() - timedelta(days=90)
            
            # Query order_lines joined with orders for the lookback period
            # Note: OrderLine doesn't have variant_id column, only sku
            order_lines_query = (
                select(OrderLine.sku, OrderLine.quantity, OrderLine.unit_price, Order.created_at)
                .join(Order, OrderLine.order_id == Order.order_id)
                .where(
                    and_(
                        Order.csv_upload_id == csv_upload_id,
                        Order.created_at >= cutoff_date
                    )
                )
            )
            order_lines_result = await session.execute(order_lines_query)
            order_lines = order_lines_result.fetchall()
            
            # Aggregate sales data by variant_id
            variant_metrics = {}
            for line in order_lines:
                # Map SKU to variant_id using catalog mapping
                line_sku = getattr(line, 'sku', None)
                variant_id = None
                if line_sku and line_sku in sku_to_variant:
                    variant_id = sku_to_variant[line_sku]
                
                if not variant_id or variant_id not in variant_to_catalog:
                    continue
                
                if variant_id not in variant_metrics:
                    variant_metrics[variant_id] = {
                        'total_quantity': 0,
                        'total_revenue': Decimal('0'),
                        'discount_sum': Decimal('0'),
                        'discount_count': 0
                    }
                
                quantity = getattr(line, 'quantity', 0) or 0
                unit_price = getattr(line, 'unit_price', Decimal('0')) or Decimal('0')
                variant_metrics[variant_id]['total_quantity'] += quantity
                variant_metrics[variant_id]['total_revenue'] += unit_price * quantity
                
                # Calculate discount percentage vs catalog price
                catalog_snapshot = variant_to_catalog[variant_id]
                catalog_price = getattr(catalog_snapshot, 'price', Decimal('0')) or Decimal('0')
                if catalog_price > Decimal('0') and unit_price > Decimal('0'):
                    discount_pct = max(Decimal('0'), (catalog_price - unit_price) / catalog_price) * Decimal('100')
                    # Cap at 100% discount
                    discount_pct = min(discount_pct, Decimal('100'))
                    variant_metrics[variant_id]['discount_sum'] += discount_pct
                    variant_metrics[variant_id]['discount_count'] += 1
            
            # Update catalog snapshots with computed flags
            updates_count = 0
            for snapshot in catalog_snapshots:
                variant_id = snapshot.variant_id
                metrics = variant_metrics.get(variant_id, {'total_quantity': 0, 'discount_sum': Decimal('0'), 'discount_count': 0})
                
                # Slow-mover detection: 0 units sold OR ≤ 2 units with stock
                available_total = getattr(snapshot, 'available_total', 0) or 0
                is_slow_mover = (
                    metrics['total_quantity'] == 0 or 
                    (metrics['total_quantity'] <= 2 and available_total > 0)
                )
                
                # Note: avg_discount_pct column was removed from database, no longer setting it
                
                # Update the snapshot with only fields that exist on the model
                setattr(snapshot, 'is_slow_mover', is_slow_mover)
                updates_count += 1
            
            await session.commit()
            logger.info(f"Updated {updates_count} catalog snapshots with objective flags for upload {csv_upload_id}")

    # New methods for enhanced v2 services
    
    # Data Mapper methods
    async def get_variant_by_sku(self, sku: str, csv_upload_id: str) -> Optional[Variant]:
        """Get variant by SKU"""
        async with self.get_session() as session:
            query = select(Variant).where(
                and_(Variant.sku == sku, Variant.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_variant_by_id(self, variant_id: str, csv_upload_id: str) -> Optional[Variant]:
        """Get variant by ID"""
        async with self.get_session() as session:
            query = select(Variant).where(
                and_(Variant.variant_id == variant_id, Variant.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_product_by_sku(self, sku: str) -> Optional[Product]:
        """Get product by SKU"""
        async with self.get_session() as session:
            query = select(Product).where(Product.sku == sku)
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_inventory_levels_by_item_id(self, inventory_item_id: str, csv_upload_id: str) -> List[InventoryLevel]:
        """Get inventory levels by inventory item ID"""
        async with self.get_session() as session:
            query = select(InventoryLevel).where(
                and_(InventoryLevel.inventory_item_id == inventory_item_id, 
                     InventoryLevel.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_order_lines_by_upload(self, csv_upload_id: str) -> List[OrderLine]:
        """Get all order lines for an upload"""
        async with self.get_session() as session:
            query = select(OrderLine).where(OrderLine.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_order_lines_by_sku(self, identifier: str, csv_upload_id: str) -> List[OrderLine]:
        """Get order lines for a specific SKU or variant_id in a CSV upload.

        IMPORTANT: The identifier may be either a SKU or variant_id (when SKU is missing).
        We query by both to ensure products without SKUs are found.
        """
        async with self.get_session() as session:
            query = select(OrderLine).where(
                and_(
                    or_(OrderLine.sku == identifier, OrderLine.variant_id == identifier),
                    OrderLine.csv_upload_id == csv_upload_id
                )
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_variant_sales_data(self, identifier: str, csv_upload_id: str, days: int = 60) -> List[OrderLine]:
        """Get sales data for a variant by SKU or variant_id.

        IMPORTANT: The identifier may be either a SKU or variant_id (when SKU is missing).
        We query by both to ensure products without SKUs are found.
        """
        async with self.get_session() as session:
            query = select(OrderLine).where(
                and_(
                    or_(OrderLine.sku == identifier, OrderLine.variant_id == identifier),
                    OrderLine.csv_upload_id == csv_upload_id
                )
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Objective Scorer methods
    @retry_async(max_retries=3, base_delay=1.0, max_delay=10.0)
    async def get_catalog_snapshots_by_upload(self, csv_upload_id: str) -> List[CatalogSnapshot]:
        """Get all catalog snapshots for an upload.

        Retries on transient database errors (connection issues, timeouts).
        """
        try:
            async with self.get_session() as session:
                query = select(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
                result = await session.execute(query)
                return list(result.scalars().all())
        except Exception as e:
            logger.warning(f"Error fetching catalog snapshots for upload {csv_upload_id}: {e}")
            return []
    
    async def get_catalog_snapshots_map(self, csv_upload_id: str) -> Dict[str, CatalogSnapshot]:
        """
        ARCHITECTURE: Returns catalog with DUAL KEYS for backward compatibility.

        Keys:
        - variant_id (primary) - Always exists, immutable, unique
        - sku (fallback) - For legacy ML pipeline compatibility

        This allows:
        - Quick-start path: Uses variant_id
        - Full generation path: Uses SKU
        - Gradual migration to variant_id across entire codebase
        """
        try:
            snapshots = await self.get_catalog_snapshots_by_upload(csv_upload_id)
            result = {}
            variant_id_count = 0
            sku_count = 0

            for snapshot in snapshots:
                # Primary key: variant_id
                if snapshot.variant_id:
                    result[snapshot.variant_id] = snapshot
                    variant_id_count += 1

                # Fallback key: SKU (for backward compatibility with ML pipeline)
                # Skip 'no-sku-*' placeholders to avoid polluting catalog
                if snapshot.sku and not snapshot.sku.startswith('no-sku-'):
                    result[snapshot.sku] = snapshot
                    sku_count += 1

            logger.info(
                f"[{csv_upload_id}] Catalog map built: {variant_id_count} variant_ids + {sku_count} SKUs = {len(result)} total keys"
            )
            return result
        except Exception as e:
            logger.error(f"Error building catalog map for {csv_upload_id}: {e}")
            return {}
    
    async def get_catalog_snapshots_by_variants(self, variant_ids: List[str], csv_upload_id: str) -> List[CatalogSnapshot]:
        """Get catalog snapshots by variant IDs"""
        async with self.get_session() as session:
            query = select(CatalogSnapshot).where(
                and_(CatalogSnapshot.variant_id.in_(variant_ids), 
                     CatalogSnapshot.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_catalog_snapshots_by_skus(self, skus: List[str], csv_upload_id: str) -> List[CatalogSnapshot]:
        """DEPRECATED: Use get_catalog_snapshots_map for safer operations"""
        try:
            async with self.get_session() as session:
                query = select(CatalogSnapshot).where(
                    and_(CatalogSnapshot.sku.in_(skus), 
                         CatalogSnapshot.csv_upload_id == csv_upload_id)
                )
                result = await session.execute(query)
                return list(result.scalars().all())
        except Exception as e:
            logger.warning(f"Error fetching catalog snapshots by SKUs: {e}")
            return []
    
    async def get_catalog_snapshot_by_sku(self, sku: str, csv_upload_id: str) -> Optional[CatalogSnapshot]:
        """DEPRECATED: Use get_catalog_snapshots_map for safer operations"""
        try:
            async with self.get_session() as session:
                query = select(CatalogSnapshot).where(
                    and_(CatalogSnapshot.sku == sku, 
                         CatalogSnapshot.csv_upload_id == csv_upload_id)
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.warning(f"Error fetching catalog snapshot for SKU {sku}: {e}")
            return None
    
    async def update_catalog_snapshots_with_flags(self, updated_items: List[Dict[str, Any]]) -> None:
        """Update catalog snapshots with objective flags"""
        async with self.get_session() as session:
            for item in updated_items:
                variant_id = item["variant_id"]
                csv_upload_id = item["csv_upload_id"]
                
                query = select(CatalogSnapshot).where(
                    and_(CatalogSnapshot.variant_id == variant_id,
                         CatalogSnapshot.csv_upload_id == csv_upload_id)
                )
                result = await session.execute(query)
                snapshot = result.scalar_one_or_none()
                
                if snapshot:
                    # Update with new flags
                    if "objective_flags" in item:
                        setattr(snapshot, "objective_flags", item["objective_flags"])
                    if "velocity" in item:
                        setattr(snapshot, "velocity", item["velocity"])
                    # Skip avg_discount_pct as it doesn't exist in the model anymore
            
            await session.commit()
    
    # ML Candidate Generator methods
    async def get_orders_with_lines(self, csv_upload_id: str) -> List[Order]:
        """Get orders with their order lines for ML training - optimized to prevent N+1 queries"""
        async with self.get_session() as session:
            # Get all orders for this CSV upload
            orders_query = select(Order).where(Order.csv_upload_id == csv_upload_id)
            orders_result = await session.execute(orders_query)
            orders = list(orders_result.scalars().all())
            
            if not orders:
                return orders
            
            # Get all order IDs 
            order_ids = [order.order_id for order in orders]
            
            # Single query to get all order lines - prevents N+1 query problem
            lines_query = select(OrderLine).where(OrderLine.order_id.in_(order_ids))
            lines_result = await session.execute(lines_query)
            all_lines = list(lines_result.scalars().all())
            
            # Group order lines by order_id
            lines_by_order = {}
            for line in all_lines:
                if line.order_id not in lines_by_order:
                    lines_by_order[line.order_id] = []
                lines_by_order[line.order_id].append(line)
            
            # Build lightweight order representations to avoid lazy-loading outside the session
            result_orders: List[SimpleNamespace] = []
            for order in orders:
                order_lines = [
                    SimpleNamespace(
                        order_id=line.order_id,
                        sku=line.sku,
                        variant_id=line.variant_id,
                    )
                    for line in lines_by_order.get(order.order_id, [])
                ]
                result_orders.append(
                    SimpleNamespace(
                        order_id=order.order_id,
                        csv_upload_id=order.csv_upload_id,
                        order_lines=order_lines,
                    )
                )
            
            return result_orders

    async def get_orders_with_lines_by_run(self, run_id: str) -> List[Order]:
        """Get all orders and their lines for a run (join via CsvUpload.run_id)."""
        if not run_id:
            return []
        async with self.get_session() as session:
            orders_query = (
                select(Order)
                .join(CsvUpload, Order.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            orders_result = await session.execute(orders_query)
            orders = list(orders_result.scalars().all())
            if not orders:
                return orders
            order_ids = [o.order_id for o in orders]
            lines_query = select(OrderLine).where(OrderLine.order_id.in_(order_ids))
            lines_result = await session.execute(lines_query)
            all_lines = list(lines_result.scalars().all())
            lines_by_order = {}
            for line in all_lines:
                lines_by_order.setdefault(line.order_id, []).append(line)
            result_orders: List[SimpleNamespace] = []
            for order in orders:
                order_lines = [
                    SimpleNamespace(
                        order_id=line.order_id,
                        sku=line.sku,
                        variant_id=line.variant_id,
                    )
                    for line in lines_by_order.get(order.order_id, [])
                ]
                result_orders.append(
                    SimpleNamespace(
                        order_id=order.order_id,
                        csv_upload_id=order.csv_upload_id,
                        order_lines=order_lines,
                    )
                )
            return result_orders
    
    async def get_variant_embeddings(self, csv_upload_id: str) -> Optional[str]:
        """Get stored variant embeddings (mock implementation)"""
        # This would be stored in a separate embeddings table in a real implementation
        return None
    
    async def store_variant_embeddings(self, csv_upload_id: str, embeddings_data: str) -> None:
        """Store variant embeddings (mock implementation)"""
        # This would store in a separate embeddings table in a real implementation
        logger.info(f"Embeddings stored for upload {csv_upload_id}")
    
    # Pricing Engine methods
    async def get_all_sales_data(self, csv_upload_id: str, days: int = 90) -> List[OrderLine]:
        """Get all sales data for pricing analysis"""
        async with self.get_session() as session:
            query = select(OrderLine).where(OrderLine.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_catalog_snapshots_map_by_run(self, run_id: str) -> Dict[str, CatalogSnapshot]:
        """Get catalog snapshots with DUAL KEYS (variant_id + SKU) for a run."""
        async with self.get_session() as session:
            query = (
                select(CatalogSnapshot)
                .join(CsvUpload, CatalogSnapshot.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            snapshots = list(result.scalars().all())

            # Build dual-key map for backward compatibility
            catalog_map = {}
            for snapshot in snapshots:
                # Primary key: variant_id
                if snapshot.variant_id:
                    catalog_map[snapshot.variant_id] = snapshot
                # Fallback key: SKU (for backward compatibility)
                if snapshot.sku and not snapshot.sku.startswith('no-sku-'):
                    catalog_map[snapshot.sku] = snapshot
            return catalog_map

    async def get_catalog_snapshots_map_by_variant(self, csv_upload_id: str) -> Dict[str, CatalogSnapshot]:
        """Get catalog snapshots as a map keyed by variant_id."""
        async with self.get_session() as session:
            query = select(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            snapshots = list(result.scalars().all())
            return {snapshot.variant_id: snapshot for snapshot in snapshots if snapshot.variant_id}

    async def get_catalog_snapshots_map_by_variant_and_run(self, run_id: str) -> Dict[str, CatalogSnapshot]:
        """Get catalog snapshots keyed by variant_id for a run."""
        async with self.get_session() as session:
            query = (
                select(CatalogSnapshot)
                .join(CsvUpload, CatalogSnapshot.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            snapshots = list(result.scalars().all())
            return {snapshot.variant_id: snapshot for snapshot in snapshots if snapshot.variant_id}

    async def get_order_lines_by_run(self, run_id: str) -> List[OrderLine]:
        if not run_id:
            return []
        async with self.get_session() as session:
            # Join OrderLine -> Order -> CsvUpload(run_id)
            query = (
                select(OrderLine)
                .join(Order, OrderLine.order_id == Order.order_id)
                .join(CsvUpload, Order.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_variant_sales_data_run(self, variant_id: str, run_id: str, days: int = 60) -> List[OrderLine]:
        """Get sales data for a variant by variant_id (or fallback to SKU lookup).

        IMPORTANT: Products may not have SKUs. We query by both SKU and variant_id
        to ensure products without SKUs are found.
        """
        if not variant_id or not run_id:
            return []
        async with self.get_session() as session:
            # Find the variant's SKU for this run (may be None)
            vq = (
                select(Variant.sku)
                .join(CsvUpload, Variant.csv_upload_id == CsvUpload.id)
                .where(and_(Variant.variant_id == variant_id, CsvUpload.run_id == run_id))
            )
            vres = await session.execute(vq)
            sku = vres.scalar_one_or_none()

            cutoff_date = datetime.now() - timedelta(days=days)

            # Build query to match by EITHER SKU or variant_id
            if sku:
                # If SKU exists, query by both SKU and variant_id
                identifier_condition = or_(OrderLine.sku == sku, OrderLine.variant_id == variant_id)
            else:
                # If no SKU, query by variant_id only
                identifier_condition = OrderLine.variant_id == variant_id

            query = (
                select(OrderLine)
                .join(Order, OrderLine.order_id == Order.order_id)
                .join(CsvUpload, Order.csv_upload_id == CsvUpload.id)
                .where(and_(CsvUpload.run_id == run_id, Order.created_at >= cutoff_date, identifier_condition))
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Deduplication methods
    async def get_bundle_recommendations_hashes(
        self,
        csv_upload_id: str,
        shop_id: Optional[str] = None
    ) -> List[BundleRecommendation]:
        """Get bundle recommendation hashes for deduplication"""
        async with self.get_session() as session:
            query = select(BundleRecommendation).where(
                BundleRecommendation.csv_upload_id == csv_upload_id
            )
            shop_filter = sanitize_shop_id(shop_id)
            if shop_filter:
                query = query.where(BundleRecommendation.shop_id == shop_filter)
            else:
                query = query.join(CsvUpload, BundleRecommendation.csv_upload_id == CsvUpload.id)
                query = query.where(BundleRecommendation.shop_id == CsvUpload.shop_id)
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def store_bundle_hashes(self, hash_records: List[Dict[str, Any]]) -> None:
        """Store bundle hashes for deduplication (mock implementation)"""
        # This would store in a separate bundle_hashes table in a real implementation
        logger.info(f"Stored {len(hash_records)} bundle hashes")

    @retry_async(max_retries=3, base_delay=1.0)
    async def cleanup_intermediate_data(self, run_id: str) -> Dict[str, int]:
        """
        Delete intermediate data after bundle generation completes.

        Removes:
        - order_lines: Line items from orders (only needed during ML analysis)
        - catalog_snapshot: Denormalized product catalog (redundant)
        - variants: Product variants (only needed during generation)
        - inventory_levels: Stock levels (only needed during generation)
        - orders: Order records (only needed during generation)

        Keeps:
        - csv_uploads: Metadata and metrics about the upload
        - bundle_recommendations: The generated bundles
        - association_rules: Can be kept for debugging or cleared separately

        Args:
            run_id: The run_id that groups all uploads from a single sync

        Returns:
            Dict with counts of deleted rows per table
        """
        if not run_id:
            raise ValueError("run_id is required for cleanup")

        deleted_counts = {
            "order_lines": 0,
            "catalog_snapshot": 0,
            "variants": 0,
            "inventory_levels": 0,
            "orders": 0,
        }

        logger.info(f"[cleanup] Starting intermediate data cleanup for run_id={run_id}")

        async with self.get_session() as session:
            try:
                # Get all csv_upload_ids for this run
                stmt = select(CsvUpload.id).where(
                    CsvUpload.processing_params["run_id"].astext == run_id
                )
                result = await session.execute(stmt)
                upload_ids = [row[0] for row in result.all()]

                if not upload_ids:
                    logger.warning(f"[cleanup] No uploads found for run_id={run_id}")
                    return deleted_counts

                logger.info(f"[cleanup] Found {len(upload_ids)} uploads for run_id={run_id}")

                # Delete in batches to avoid long-running transactions
                BATCH_SIZE = 1000

                # Delete order_lines in batches
                total_order_lines = 0
                while True:
                    result = await session.execute(
                        delete(OrderLine)
                        .where(OrderLine.csv_upload_id.in_(upload_ids))
                        .execution_options(synchronize_session=False)
                    )
                    if result.rowcount == 0:
                        break
                    total_order_lines += result.rowcount
                    await session.commit()
                    if result.rowcount < BATCH_SIZE:
                        break
                deleted_counts["order_lines"] = total_order_lines
                logger.info(f"[cleanup] Deleted {total_order_lines} order_lines")

                # Delete catalog_snapshot
                result = await session.execute(
                    delete(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id.in_(upload_ids))
                )
                deleted_counts["catalog_snapshot"] = result.rowcount
                logger.info(f"[cleanup] Deleted {result.rowcount} catalog_snapshot rows")
                await session.commit()

                # Delete variants
                result = await session.execute(
                    delete(Variant).where(Variant.csv_upload_id.in_(upload_ids))
                )
                deleted_counts["variants"] = result.rowcount
                logger.info(f"[cleanup] Deleted {result.rowcount} variants")
                await session.commit()

                # Delete inventory_levels
                result = await session.execute(
                    delete(InventoryLevel).where(InventoryLevel.csv_upload_id.in_(upload_ids))
                )
                deleted_counts["inventory_levels"] = result.rowcount
                logger.info(f"[cleanup] Deleted {result.rowcount} inventory_levels")
                await session.commit()

                # Delete orders
                result = await session.execute(
                    delete(Order).where(Order.csv_upload_id.in_(upload_ids))
                )
                deleted_counts["orders"] = result.rowcount
                logger.info(f"[cleanup] Deleted {result.rowcount} orders")
                await session.commit()

                total_deleted = sum(deleted_counts.values())
                logger.info(
                    f"[cleanup] ✅ Cleanup complete for run_id={run_id}: "
                    f"total {total_deleted} rows deleted across {len(deleted_counts)} tables"
                )

                return deleted_counts

            except Exception as e:
                logger.exception(f"[cleanup] ❌ Cleanup failed for run_id={run_id}: {e}")
                await session.rollback()
                raise

# Global storage instance
storage = StorageService()
storage.client = storage


# Convenience wrappers for async callers that import functions directly
async def update_csv_upload_status(
    csv_upload_id: str,
    status: str,
    error_message: Optional[str] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[CsvUpload]:
    return await storage.update_csv_upload_status(
        csv_upload_id,
        status,
        error_message=error_message,
        extra_metrics=extra_metrics,
    )


async def summarize_upload_coverage(csv_upload_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    return await storage.summarize_upload_coverage(csv_upload_id, run_id)
