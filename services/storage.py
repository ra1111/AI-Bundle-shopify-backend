"""
Storage Service Layer
Provides database operations matching the TypeScript storage interface
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, desc, and_, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from database import (
    AsyncSessionLocal, User, CsvUpload, Order, OrderLine, Product, 
    Variant, InventoryLevel, CatalogSnapshot,
    AssociationRule, Bundle, BundleRecommendation
)

logger = logging.getLogger(__name__)

class StorageService:
    """Storage service providing database operations"""

    # ---------- NEW: helpers & defaults ----------

    ORDER_LINE_TEXT_DEFAULT = "unspecified"

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

    def _sanitize_order_line(self, line: dict) -> dict | None:
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

    # ---------------- Run helpers ----------------
    async def get_run_id_for_upload(self, csv_upload_id: str) -> Optional[str]:
        try:
            async with self.get_session() as session:
                from database import CsvUpload
                cu = await session.get(CsvUpload, csv_upload_id)
                return cu.run_id if cu else None
        except Exception:
            return None
    
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
            upload = CsvUpload(**upload_data)
            session.add(upload)
            await session.commit()
            await session.refresh(upload)
            return upload
    
    async def get_csv_upload(self, upload_id: str) -> Optional[CsvUpload]:
        """Get CSV upload by ID"""
        async with self.get_session() as session:
            return await session.get(CsvUpload, upload_id)
    
    async def update_csv_upload(self, upload_id: str, updates: Dict[str, Any]) -> Optional[CsvUpload]:
        """Update CSV upload record"""
        async with self.get_session() as session:
            upload = await session.get(CsvUpload, upload_id)
            if upload:
                for key, value in updates.items():
                    setattr(upload, key, value)
                await session.commit()
                await session.refresh(upload)
            return upload
    
    async def get_recent_uploads(self, limit: int = 10) -> List[CsvUpload]:
        """Get recent CSV uploads"""
        async with self.get_session() as session:
            query = select(CsvUpload).order_by(desc(CsvUpload.created_at)).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())
    
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
        """Bulk upsert order lines - handle both NULL and NOT NULL line_item_id cases with partial indexes"""
        if not order_lines_data:
            return []

        # NEW: sanitize + filter + drop invalid
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
        
        async with self.get_session() as session:
            # Separate data by line_item_id NULL vs NOT NULL cases for different conflict targets
            lines_with_item_id = [line for line in order_lines_data if line.get('line_item_id') is not None]
            lines_without_item_id = [line for line in order_lines_data if line.get('line_item_id') is None]
            
            # Handle lines with line_item_id (NOT NULL case)
            if lines_with_item_id:
                stmt = pg_insert(OrderLine).values(lines_with_item_id)
                upsert = stmt.on_conflict_do_update(
                    index_elements=[OrderLine.order_id, OrderLine.line_item_id],
                    index_where=OrderLine.line_item_id.is_not(None),
                    set_=self._overwrite_all_columns(OrderLine.__table__, stmt)
                )
                await session.execute(upsert)
            
            # Handle lines without line_item_id (NULL case)  
            if lines_without_item_id:
                stmt = pg_insert(OrderLine).values(lines_without_item_id)
                upsert = stmt.on_conflict_do_update(
                    index_elements=[OrderLine.order_id, OrderLine.sku],
                    index_where=OrderLine.line_item_id.is_(None),
                    set_=self._overwrite_all_columns(OrderLine.__table__, stmt)
                )
                await session.execute(upsert)
            
            await session.commit()
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
    
    async def get_order_lines(self, csv_upload_id: str) -> List[OrderLine]:
        """Get order lines for a specific CSV upload"""
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
        """Bulk create bundle recommendations"""
        async with self.get_session() as session:
            recommendations = [BundleRecommendation(**rec_data) for rec_data in recommendations_data]
            session.add_all(recommendations)
            await session.commit()
            return recommendations
    
    async def get_bundle_recommendations(self, csv_upload_id: str) -> List[BundleRecommendation]:
        """Get bundle recommendations for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(BundleRecommendation).where(BundleRecommendation.csv_upload_id == csv_upload_id)
            
            # Per-upload scoped query: Use persistent rank_position with NULLS LAST
            query = query.order_by(
                BundleRecommendation.rank_position.asc().nulls_last(),  # Persistent ranking for this upload
                desc(BundleRecommendation.ranking_score),  # Fallback for any records without rank_position
                desc(BundleRecommendation.confidence),
                desc(BundleRecommendation.created_at)
            )
            
            query = query.limit(50)
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
    
    async def clear_bundle_recommendations(self, csv_upload_id: str) -> None:
        """Clear bundle recommendations for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = delete(BundleRecommendation).where(BundleRecommendation.csv_upload_id == csv_upload_id)
            await session.execute(query)
            await session.commit()
    
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
        """Bulk upsert inventory levels - overwrite existing levels with same (inventory_item_id, location_id)"""
        if not inventory_data:
            return []
        async with self.get_session() as session:
            stmt = pg_insert(InventoryLevel).values(inventory_data)
            upsert = stmt.on_conflict_do_update(
                index_elements=[InventoryLevel.inventory_item_id, InventoryLevel.location_id],
                set_=self._overwrite_all_columns(InventoryLevel.__table__, stmt)
            )
            await session.execute(upsert)
            await session.commit()
            return []
    
    async def get_inventory_levels(self, csv_upload_id: str) -> List[InventoryLevel]:
        """Get inventory levels for a specific CSV upload"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        async with self.get_session() as session:
            query = select(InventoryLevel).where(InventoryLevel.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            return list(result.scalars().all())

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
        """Bulk upsert catalog snapshots - overwrite existing snapshots with same (csv_upload_id, variant_id)"""
        if not catalog_data:
            return []
        async with self.get_session() as session:
            stmt = pg_insert(CatalogSnapshot).values(catalog_data)
            upsert = stmt.on_conflict_do_update(
                index_elements=[CatalogSnapshot.csv_upload_id, CatalogSnapshot.variant_id],
                set_=self._overwrite_all_columns(CatalogSnapshot.__table__, stmt)
            )
            await session.execute(upsert)
            await session.commit()
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
    
    async def get_order_lines_by_sku(self, sku: str, csv_upload_id: str) -> List[OrderLine]:
        """Get order lines for a specific SKU in a CSV upload"""
        async with self.get_session() as session:
            query = select(OrderLine).where(
                and_(OrderLine.sku == sku, OrderLine.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def get_variant_sales_data(self, variant_id: str, csv_upload_id: str, days: int = 60) -> List[OrderLine]:
        """Get sales data for a variant"""
        async with self.get_session() as session:
            # For simplicity, return order lines by SKU since variant mapping might not be complete
            query = select(OrderLine).where(
                and_(OrderLine.sku == variant_id, OrderLine.csv_upload_id == csv_upload_id)
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Objective Scorer methods
    async def get_catalog_snapshots_by_upload(self, csv_upload_id: str) -> List[CatalogSnapshot]:
        """Get all catalog snapshots for an upload"""
        try:
            async with self.get_session() as session:
                query = select(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
                result = await session.execute(query)
                return list(result.scalars().all())
        except Exception as e:
            logger.warning(f"Error fetching catalog snapshots for upload {csv_upload_id}: {e}")
            return []
    
    async def get_catalog_snapshots_map(self, csv_upload_id: str) -> Dict[str, CatalogSnapshot]:
        """ARCHITECT FIX: Safe method that returns catalog as SKU->CatalogSnapshot dict"""
        try:
            snapshots = await self.get_catalog_snapshots_by_upload(csv_upload_id)
            return {snapshot.sku: snapshot for snapshot in snapshots if snapshot.sku}
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
            
            # Attach order lines to their respective orders
            for order in orders:
                order.order_lines = lines_by_order.get(order.order_id, [])
            
            return orders
    
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
    
    async def get_catalog_snapshots_map(self, csv_upload_id: str) -> Dict[str, CatalogSnapshot]:
        """Get catalog snapshots as a map keyed by SKU"""
        async with self.get_session() as session:
            query = select(CatalogSnapshot).where(CatalogSnapshot.csv_upload_id == csv_upload_id)
            result = await session.execute(query)
            snapshots = list(result.scalars().all())
            return {snapshot.sku: snapshot for snapshot in snapshots if snapshot.sku is not None}

    async def get_catalog_snapshots_map_by_run(self, run_id: str) -> Dict[str, CatalogSnapshot]:
        async with self.get_session() as session:
            query = (
                select(CatalogSnapshot)
                .join(CsvUpload, CatalogSnapshot.csv_upload_id == CsvUpload.id)
                .where(CsvUpload.run_id == run_id)
            )
            result = await session.execute(query)
            snapshots = list(result.scalars().all())
            return {snapshot.sku: snapshot for snapshot in snapshots if snapshot.sku}

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
        if not variant_id or not run_id:
            return []
        async with self.get_session() as session:
            # Find the variant's SKU for this run
            vq = (
                select(Variant.sku)
                .join(CsvUpload, Variant.csv_upload_id == CsvUpload.id)
                .where(and_(Variant.variant_id == variant_id, CsvUpload.run_id == run_id))
            )
            vres = await session.execute(vq)
            sku = vres.scalar_one_or_none()
            if not sku:
                return []
            cutoff_date = datetime.now() - timedelta(days=days)
            query = (
                select(OrderLine)
                .join(Order, OrderLine.order_id == Order.order_id)
                .join(CsvUpload, Order.csv_upload_id == CsvUpload.id)
                .where(and_(CsvUpload.run_id == run_id, Order.created_at >= cutoff_date, OrderLine.sku == sku))
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    # Deduplication methods
    async def get_bundle_recommendations_hashes(self, csv_upload_id: str) -> List[BundleRecommendation]:
        """Get bundle recommendation hashes for deduplication"""
        async with self.get_session() as session:
            query = select(BundleRecommendation).where(
                BundleRecommendation.csv_upload_id == csv_upload_id
            )
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def store_bundle_hashes(self, hash_records: List[Dict[str, Any]]) -> None:
        """Store bundle hashes for deduplication (mock implementation)"""
        # This would store in a separate bundle_hashes table in a real implementation
        logger.info(f"Stored {len(hash_records)} bundle hashes")

# Global storage instance
storage = StorageService()
