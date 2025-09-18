"""
CSV Processor Service
Handles parsing and processing of uploaded CSV files
"""
import csv
import io
import re
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
from decimal import Decimal

from .storage import storage
from database import AsyncSessionLocal  # kept for parity; storage handles inserts
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

# ---------- Defaults used to satisfy NOT NULLs on orders/lines ----------
ORDER_NOT_NULL_DEFAULTS = {
    'customer_id':        lambda r, oid: r.get('customerId') or f'customer_{oid}',
    'customer_email':     lambda r, oid: r.get('customerEmail') or f'customer_{oid}@example.com',
    'customer_country':   lambda r, oid: r.get('shippingCountryCode') or 'US',
    'customer_currency':  lambda r, oid: r.get('currencyCode') or 'USD',
    'clv_band':           lambda r, oid: r.get('clvBand') or 'medium',
    'channel':            lambda r, oid: r.get('salesChannel') or 'online',
    'device':             lambda r, oid: r.get('deviceType') or 'desktop',
    'discount':           lambda r, oid: Decimal(str(r.get('discount', '0') or '0')),
    'discount_code':      lambda r, oid: r.get('discountCode') or '',
    'returned':           lambda r, oid: False,
    'basket_item_count':  lambda r, oid: 1,
    'basket_line_count':  lambda r, oid: 1,
    'basket_value':       lambda r, oid: Decimal(str(r.get('basketValue', '0') or '0')),
    'financial_status':   lambda r, oid: r.get('displayFinancialStatus') or '',
    'fulfillment_status': lambda r, oid: r.get('displayFulfillmentStatus') or '',
}

LINE_NOT_NULL_DEFAULTS = {
    'brand':      lambda r: r.get('vendor') or 'unknown',
    'category':   lambda r: r.get('productType') or 'general',
    'subcategory':lambda r: r.get('subcategory') or r.get('productSubcategory') or r.get('productType') or 'general',
    'color':      lambda r: r.get('color') or 'unspecified',
    'material':   lambda r: r.get('material') or 'unspecified',
    'price':      lambda r: Decimal(str(r.get('price') or '0')),
    'cost':       lambda r: Decimal('0'),
    'weight_kg':  lambda r: Decimal('0'),
    'tags':       lambda r: r.get('tags', ''),
    'hist_views': lambda r: 0,
    'hist_adds':  lambda r: 0,
}

class CSVProcessor:
    """CSV processing service for 4-CSV ingestion model."""

    def __init__(self):
        self.schema_version = "v2.0"
        self.valid_types = {"orders", "variants", "inventory_levels", "catalog_joined"}

        # Required columns per canonical type (kept minimal to avoid false negatives)
        self.required_columns = {
            'orders': {'createdAt', 'lineItemQuantity', 'originalUnitPrice'},
            'variants': {'product_id', 'variant_id', 'variant_title', 'price', 'inventory_item_id'},
            'inventory_levels': {'inventory_item_id', 'location_id', 'available', 'updated_at'},
            'catalog_joined': {
                'product_id', 'variant_id', 'product_title',
                'price', 'inventory_item_id', 'product_status'
            },
        }

        # One-of groups (at least one field from each group must be present)
        self.required_one_of = {
            'orders': [
                {'order_id', 'orderId', 'name'},
                {'variantId', 'variant_id', 'sku'}
            ],
            # variants & catalog_joined require variant_id explicitly (no one-of group here)
        }

        self.numeric_fields = {
            'orders': {'lineItemQuantity', 'originalUnitPrice', 'subtotalPrice',
                       'totalShippingPrice', 'totalTax', 'totalPrice',
                       'discountedTotal', 'lineItemTotalDiscount'},
            'variants': {'price', 'compare_at_price'},
            'inventory_levels': {'available'},
            'catalog_joined': {'price', 'compare_at_price', 'available_total'},
        }

        self.datetime_fields = {
            'orders': {'createdAt'},
            'variants': {'inventory_item_created_at'},
            'inventory_levels': {'updated_at'},
            'catalog_joined': {'product_created_at', 'product_published_at',
                               'inventory_item_created_at', 'last_inventory_update'},
        }

        # Header exemplars (used only for readability/docs)
        self.orders_headers = {
            'order_id', 'name', 'createdAt', 'currencyCode', 'displayFinancialStatus',
            'displayFulfillmentStatus', 'subtotalPrice', 'totalShippingPrice', 'totalTax',
            'totalPrice', 'customerEmail', 'shippingCountryCode', 'shippingProvince',
            'shippingCity', 'lineItemId', 'lineItemName', 'lineItemQuantity', 'sku',
            'vendor', 'variantTitle', 'productId', 'productTitle', 'productType',
            'variantId', 'originalUnitPrice', 'discountedTotal', 'lineItemTotalDiscount',
            'requiresShipping', 'isGiftCard', 'taxable', 'imageUrl'
        }

        self.variants_headers = {
            'product_id', 'variant_id', 'sku', 'variant_title',
            'price', 'compare_at_price', 'inventory_item_id', 'inventory_item_created_at'
        }

        self.inventory_levels_headers = {'inventory_item_id', 'location_id', 'available', 'updated_at'}

        self.catalog_joined_headers = {
            'product_id', 'product_title', 'product_type', 'tags', 'product_status',
            'vendor', 'product_created_at', 'product_published_at', 'variant_id',
            'sku', 'variant_title', 'price', 'compare_at_price', 'inventory_item_id',
            'inventory_item_created_at', 'available_total', 'last_inventory_update'
        }

    # ---------- Main entry ----------

    async def process_csv(self, csv_content: str, upload_id: str, csv_type: str = "auto") -> None:
        """Process CSV content into DB tables based on canonical type."""
        try:
            logger.info(f"Starting CSV processing for upload {upload_id}, type: {csv_type}")

            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(csv_reader)
            if not rows:
                raise ValueError("CSV file is empty")

            # Track row counts early
            await storage.update_csv_upload(upload_id, {
                "total_rows": len(rows),
                "processed_rows": 0,
                "schema_version": self.schema_version
            })

            # Normalize incoming aliases to canonical types
            alias_map = {
                "products": "catalog_joined",
                "catalog": "catalog_joined",
                "catalog_joined": "catalog_joined",
                "variants": "variants",
                "orders": "orders",
                "inventory_levels": "inventory_levels",
                None: "auto",
            }
            csv_type = alias_map.get(csv_type, csv_type)

            # Detect when requested
            if csv_type == "auto":
                csv_type = self.detect_csv_type(list(rows[0].keys()))

            # Persist detected type for observability
            await storage.update_csv_upload(upload_id, {"csv_type": csv_type})
            logger.info(f"Processing CSV type: {csv_type}")

            # Validate schema + sample datatypes
            await self.validate_csv_schema(csv_type, list(rows[0].keys()), upload_id)
            await self.validate_sample_rows(csv_type, rows[:5], upload_id)

            # Dispatch
            if csv_type == "orders":
                await self.process_orders_csv(rows, upload_id)
            elif csv_type == "variants":
                await self.process_variants_csv(rows, upload_id)
            elif csv_type == "inventory_levels":
                await self.process_inventory_levels_csv(rows, upload_id)
            elif csv_type == "catalog_joined":
                await self.process_catalog_joined_csv(rows, upload_id)
            else:
                await self.process_legacy_format(rows, upload_id)

            # Done
            await storage.update_csv_upload(upload_id, {
                "status": "completed",
                "processed_rows": len(rows)
            })
            logger.info(f"CSV processing completed for upload {upload_id}")

        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            await storage.update_csv_upload(upload_id, {"status": "failed", "error_message": str(e)})
            raise

    # ---------- Detection & validation ----------

    def detect_csv_type(self, headers: List[str]) -> str:
        h = {x.strip().lower() for x in headers}
        if {"order_id"} <= h and ({"line_item_id"} & h or {"sku"} & h):
            return "orders"
        if {"inventory_item_id", "location_id"} <= h:
            return "inventory_levels"
        if {"variant_id", "product_id"} <= h and {"inventory_item_id"} <= h:
            return "variants"
        if {"variant_id", "product_id"} <= h and {"product_title"} & h:
            return "catalog_joined"
        raise ValueError("Could not detect CSV type from headers")

    async def validate_csv_schema(self, csv_type: str, headers: List[str], upload_id: str) -> None:
        if csv_type not in self.valid_types:
            msg = (f"Unrecognized CSV type '{csv_type}'. Expected one of: "
                   f"{', '.join(sorted(self.valid_types))}. Found headers: {', '.join(headers[:10])}")
            await self._fail_upload_with_error(upload_id, msg)
            raise ValueError(msg)

        required_cols = self.required_columns.get(csv_type, set())
        headers_set = {h.strip() for h in headers}
        missing_cols = required_cols - headers_set

        one_of_failures = []
        for grp in self.required_one_of.get(csv_type, []):
            if headers_set.isdisjoint(grp):
                one_of_failures.append(grp)

        if missing_cols or one_of_failures:
            parts = []
            if missing_cols:
                parts.append(f"Missing required: {', '.join(sorted(missing_cols))}")
            for grp in one_of_failures:
                parts.append(f"Need at least one of: {', '.join(sorted(grp))}")
            msg = f"{csv_type}.csv schema error → " + " | ".join(parts)
            await self._fail_upload_with_error(upload_id, msg)
            raise ValueError(msg)

        logger.info(f"Schema validation passed for {csv_type} with {len(headers)} columns")

    async def validate_sample_rows(self, csv_type: str, sample_rows: List[Dict[str, str]], upload_id: str) -> None:
        numeric_fields = self.numeric_fields.get(csv_type, set())
        datetime_fields = self.datetime_fields.get(csv_type, set())

        for i, row in enumerate(sample_rows, 1):
            for field in numeric_fields:
                if field in row:
                    raw = (row[field] or '').strip()
                    if raw and raw.upper() != 'NULL':
                        try:
                            float(raw)
                        except ValueError:
                            msg = (f"Invalid numeric value in {csv_type} CSV, row {i}, "
                                   f"column '{field}': '{raw}'. Expected a number.")
                            await self._fail_upload_with_error(upload_id, msg)
                            raise ValueError(msg)

            for field in datetime_fields:
                if field in row:
                    raw = (row[field] or '').strip()
                    if raw and raw.upper() not in ['NULL', 'NONE', '']:
                        if not self._is_valid_datetime_format(raw):
                            msg = (f"Invalid datetime format in {csv_type} CSV, row {i}, "
                                   f"column '{field}': '{raw}'. Expected format like YYYY-MM-DD or ISO-8601.")
                            await self._fail_upload_with_error(upload_id, msg)
                            raise ValueError(msg)

        logger.info(f"Sample data validation passed for {csv_type}")

    def _is_valid_datetime_format(self, date_str: str) -> bool:
        if not date_str:
            return True
        bad = ['0000-00-00', '1900-01-01', 'N/A', 'NULL', 'null']
        if date_str.lower() in [x.lower() for x in bad]:
            return False
        pattern = r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{1,2}(?::\d{1,2})?(?:Z|[+-]\d{2}:?\d{2})?)?$'
        return bool(re.match(pattern, date_str.strip()))

    async def _fail_upload_with_error(self, upload_id: str, error_message: str) -> None:
        await storage.update_csv_upload(upload_id, {"status": "failed", "error_message": error_message})
        logger.error(f"CSV upload {upload_id} failed: {error_message}")

    # ---------- Orders (unchanged, but with sanitizers) ----------

    async def process_orders_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        orders_map: Dict[str, Dict[str, Any]] = {}
        order_lines: List[Dict[str, Any]] = []

        for row in rows:
            try:
                order_id = (row.get('order_id') or row.get('orderId') or row.get('name') or '').strip()
                if not order_id:
                    continue
                if order_id not in orders_map:
                    orders_map[order_id] = self.extract_order_data(row, upload_id)

                line_data = self.extract_order_line_data(row, upload_id, order_id)
                if line_data:
                    line_data = self.sanitize_line(line_data, row)
                    order_lines.append(self._filter_line_fields(line_data))
            except Exception as e:
                logger.warning(f"Error processing order row: {e}")
                continue

        if orders_map:
            await storage.create_orders([self._filter_order_fields(o) for o in orders_map.values()])
            logger.info(f"Created {len(orders_map)} orders")
        if order_lines:
            await storage.create_order_lines(order_lines)
            logger.info(f"Created {len(order_lines)} order lines")

    # ---------- Variants ----------

    async def process_variants_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Insert rows into `variants`."""
        variants: List[Dict[str, Any]] = []
        for r in rows:
            try:
                vid = (r.get("variant_id") or "").strip()
                pid = (r.get("product_id") or "").strip()
                if not vid or not pid:
                    continue

                price = Decimal(str(r.get("price", "0") or "0"))
                compare_at_price = Decimal(str(r.get("compare_at_price", "0") or "0"))

                variants.append({
                    "variant_id": vid,
                    "product_id": pid,
                    "sku": (r.get("sku") or r.get("variant_sku") or "").strip() or None,
                    "variant_title": (r.get("variant_title") or "").strip() or "Variant",
                    "price": price,
                    "compare_at_price": compare_at_price if compare_at_price != 0 else None,
                    "inventory_item_id": (r.get("inventory_item_id") or "").strip(),
                    "inventory_item_created_at": self.parse_datetime(r.get("inventory_item_created_at")),
                    "csv_upload_id": upload_id,
                })
            except Exception as e:
                logger.warning(f"Error processing variants row: {e}")
                continue

        if variants:
            await storage.create_variants(variants)
            logger.info(f"Created {len(variants)} variants")

    # ---------- Inventory ----------

    async def process_inventory_levels_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        inventory_levels = []
        for row in rows:
            try:
                raw_available = (row.get('available', '') or '').strip()
                try:
                    available = int(float(raw_available)) if raw_available != '' else 0
                except (ValueError, TypeError):
                    available = 0

                location_id = (row.get('location_id') or '').strip()
                if not location_id:
                    logger.warning(f"Skipping inventory row with missing location_id: {row.get('inventory_item_id','?')}")
                    continue

                inventory_levels.append({
                    "inventory_item_id": (row.get('inventory_item_id') or '').strip(),
                    "csv_upload_id": upload_id,
                    "location_id": location_id,
                    "available": available,
                    "updated_at": self.parse_datetime(row.get('updated_at')),
                })
            except Exception as e:
                logger.warning(f"Error processing inventory row: {e}")
                continue

        if inventory_levels:
            await storage.create_inventory_levels(inventory_levels)
            logger.info(f"Created {len(inventory_levels)} inventory level records")

    # ---------- Catalog snapshot ----------

    async def process_catalog_joined_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Insert rows into `catalog_snapshot` (wide product+variant view)."""
        snaps: List[Dict[str, Any]] = []

        for row in rows:
            try:
                pid = (row.get('product_id') or '').strip()
                vid = (row.get('variant_id') or '').strip()
                if not pid or not vid:
                    continue

                product_status = (row.get('product_status') or '').upper()
                if product_status in {'ARCHIVED', 'DRAFT'}:
                    continue

                price = Decimal(str(row.get('price', '0') or '0'))
                compare_at_price = Decimal(str(row.get('compare_at_price', '0') or '0'))

                available_total = row.get('available_total', '')
                if available_total == '' or available_total is None:
                    available_total_int = None
                else:
                    try:
                        available_total_int = int(float(available_total))
                        if available_total_int == -1:
                            available_total_int = None
                    except (ValueError, TypeError):
                        available_total_int = None

                snap = {
                    "product_id": pid,
                    "csv_upload_id": upload_id,
                    "product_title": row.get('product_title', 'Unknown Product'),
                    "product_type": row.get('product_type', ''),
                    "tags": row.get('tags', ''),
                    "product_status": product_status,
                    "vendor": row.get('vendor', ''),
                    "product_created_at": self.parse_datetime(row.get('product_created_at')),
                    "product_published_at": self.parse_datetime(row.get('product_published_at')),
                    "variant_id": vid,
                    "sku": row.get('sku', ''),
                    "variant_title": row.get('variant_title', 'Default Title'),
                    "price": price,
                    "compare_at_price": compare_at_price if compare_at_price != 0 else None,
                    "inventory_item_id": (row.get('inventory_item_id') or '').strip(),
                    "inventory_item_created_at": self.parse_datetime(row.get('inventory_item_created_at')),
                    "available_total": available_total_int,
                    "last_inventory_update": self.parse_datetime(row.get('last_inventory_update')),
                }

                snaps.append(self._filter_snapshot_fields(snap))

            except Exception as e:
                logger.warning(f"Error processing catalog row: {e}")
                continue

        if snaps:
            await storage.create_catalog_snapshots(snaps)
            logger.info(f"Created {len(snaps)} catalog snapshot records")
            # Best-effort post-computation; non-fatal if it fails
            try:
                await storage.recompute_catalog_objectives(upload_id)
            except Exception as e:
                logger.error(f"Objective recompute failed for {upload_id}: {e}")

    # ---------- Legacy fallback ----------

    async def process_legacy_format(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        orders_map: Dict[str, Dict[str, Any]] = {}
        order_lines: List[Dict[str, Any]] = []

        for i, row in enumerate(rows):
            try:
                order_id = row.get('order_id') or f"custom_order_{i}_{upload_id[:8]}"
                if order_id not in orders_map:
                    orders_map[order_id] = self.create_default_order(row, upload_id, order_id)
                order_lines.append(self.create_default_order_line(row, upload_id, order_id, i))
            except Exception as e:
                logger.warning(f"Error processing custom row: {e}")
                continue

        if orders_map:
            await storage.create_orders([self._filter_order_fields(o) for o in orders_map.values()])
        if order_lines:
            await storage.create_order_lines(order_lines)

    # ---------- Helpers (orders/lines) ----------

    def sanitize_order(self, order_dict: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
        oid = order_dict.get("order_id", "unknown")
        for field, default_func in ORDER_NOT_NULL_DEFAULTS.items():
            if order_dict.get(field) is None:
                order_dict[field] = default_func(row, oid)
        return order_dict

    def sanitize_line(self, line_dict: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
        def empty(v): 
            return v is None or (isinstance(v, str) and v.strip() == "")
        out = dict(line_dict)
        for field, default_func in LINE_NOT_NULL_DEFAULTS.items():
            if field not in out or empty(out[field]):
                out[field] = default_func(row)
        for k in ("price", "cost", "weight_kg", "unit_price", "line_total", "line_discount"):
            if k in out and out[k] is None:
                out[k] = Decimal("0")
        for k in ("quantity", "hist_views", "hist_adds"):
            if k in out and out[k] is None:
                out[k] = 0
        return out

    def _filter_order_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "order_id","csv_upload_id","customer_id","customer_email","customer_country",
            "customer_currency","clv_band","created_at","channel","device","subtotal",
            "discount","discount_code","shipping","taxes","total","financial_status",
            "fulfillment_status","returned","basket_item_count","basket_line_count","basket_value"
        }
        return {k: v for k, v in d.items() if k in allowed}

    def extract_order_data(self, row: Dict[str, str], upload_id: str) -> Dict[str, Any]:
        dec = lambda x: Decimal(str(x or '0'))
        order_id = row.get('order_id') or row.get('orderId') or row.get('name') or str(uuid.uuid4())
        order_dict = {
            "order_id": str(order_id).strip(),
            "csv_upload_id": upload_id,
            "customer_id": row.get('customerId'),
            "customer_email": row.get('customerEmail'),
            "customer_country": row.get('shippingCountryCode'),
            "customer_currency": row.get('currencyCode'),
            "created_at": self.parse_datetime(row.get('createdAt')),
            "subtotal": dec(row.get('subtotalPrice')),
            "shipping": dec(row.get('totalShippingPrice')),
            "taxes": dec(row.get('totalTax')),
            "total": dec(row.get('totalPrice')),
            "financial_status": row.get('displayFinancialStatus'),
            "fulfillment_status": row.get('displayFulfillmentStatus'),
        }
        return self._filter_order_fields(self.sanitize_order(order_dict, row))

    def extract_order_line_data(self, row: Dict[str, str], upload_id: str, order_id: str) -> Optional[Dict[str, Any]]:
        variant_id = (row.get('variantId') or row.get('variant_id') or '').strip()
        sku = (row.get('sku') or '').strip()
        if not (variant_id or sku):
            return None

        def as_int(v: str, default=0) -> int:
            try:
                return int(float(v))
            except Exception:
                return default
        dec = lambda x: Decimal(str(x or '0'))

        discounted = row.get('discountedTotal')
        line_qty = as_int(row.get('lineItemQuantity', '1'), 1)
        line_total = dec(discounted) if discounted not in (None, '', 'NULL') else dec(row.get('originalUnitPrice')) * line_qty

        return {
            "id": str(uuid.uuid4()),
            "csv_upload_id": upload_id,
            "order_id": order_id,
            "line_item_id": row.get('lineItemId'),
            "variant_id": variant_id or None,
            "sku": sku or None,
            "name": row.get('lineItemName') or row.get('productTitle') or 'Unknown',
            "brand": row.get('vendor'),
            "category": row.get('productType'),
            "quantity": line_qty,
            "unit_price": dec(row.get('originalUnitPrice')),
            "line_total": line_total,
            "line_discount": dec(row.get('lineItemTotalDiscount')),
        }

    def _filter_line_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "id","csv_upload_id","order_id","sku","name","brand","category",
            "subcategory","color","material","price","cost","weight_kg","tags",
            "quantity","unit_price","line_total","hist_views","hist_adds",
            "variant_id","line_item_id","line_discount"
        }
        return {k:v for k,v in d.items() if k in allowed}

    # ---------- Helpers (catalog) ----------

    def _filter_snapshot_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only columns that exist on CatalogSnapshot model."""
        allowed = {
            "id", "product_id", "product_title", "product_type", "tags", "product_status",
            "vendor", "product_created_at", "product_published_at", "variant_id", "sku",
            "variant_title", "price", "compare_at_price", "inventory_item_id",
            "inventory_item_created_at", "available_total", "last_inventory_update",
            "csv_upload_id"
        }
        return {k: v for k, v in d.items() if k in allowed}

    # ---------- Misc helpers ----------

    def create_default_order(self, row: Dict[str, str], upload_id: str, order_id: str) -> Dict[str, Any]:
        dec = lambda x: Decimal(str(x or '100'))
        order_dict = {
            "order_id": order_id,
            "csv_upload_id": upload_id,
            "customer_id": row.get('customer_id'),
            "customer_email": row.get('customer_email'),
            "customer_country": row.get('country'),
            "customer_currency": row.get('currency'),
            "clv_band": 'medium',
            "created_at": self.parse_datetime(row.get('date') or row.get('created_at')),
            "channel": 'online',
            "device": 'desktop',
            "subtotal": dec(row.get('subtotal')),
            "discount": Decimal('0'),
            "discount_code": None,
            "shipping": Decimal('0'),
            "taxes": Decimal('0'),
            "total": dec(row.get('total')),
            "financial_status": row.get('financial_status'),
            "fulfillment_status": row.get('fulfillment_status'),
            "returned": False,
            "basket_item_count": 1,
            "basket_line_count": 1,
            "basket_value": dec(row.get('total')),
        }
        return self._filter_order_fields(self.sanitize_order(order_dict, row))

    def create_default_order_line(self, row: Dict[str, str], upload_id: str, order_id: str, index: int) -> Dict[str, Any]:
        sku = row.get('sku') or row.get('product_id') or f"product_{index}"
        price = Decimal(str(row.get('price', '100') or '100'))
        quantity = int(row.get('quantity', '1') or '1')
        return {
            "id": str(uuid.uuid4()),
            "csv_upload_id": upload_id,
            "order_id": order_id,
            "sku": sku,
            "name": row.get('product_name', row.get('name', f'Product {index}')),
            "brand": row.get('brand', 'Unknown'),
            "category": row.get('category', 'General'),
            "subcategory": row.get('subcategory', 'General'),
            "color": row.get('color', 'Unknown'),
            "material": row.get('material', 'Unknown'),
            "price": price,
            "cost": price * Decimal('0.7'),
            "weight_kg": Decimal('1.0'),
            "tags": row.get('tags', ''),
            "quantity": quantity,
            "unit_price": price,
            "line_total": price * quantity,
            "hist_views": 0,
            "hist_adds": 0,
        }

    def parse_datetime(self, date_str: Optional[str]) -> datetime:
        if not date_str or not date_str.strip():
            return datetime.now()
        ds = date_str.strip()
        bad = ['0000-00-00', '1900-01-01', 'N/A', 'NULL', 'null', '', 'nan']
        if ds.lower() in [x.lower() for x in bad]:
            return datetime.now()

        fmts = [
            '%Y-%m-%dT%H:%M:%S.%fZ','%Y-%m-%dT%H:%M:%SZ','%Y-%m-%dT%H:%M:%S.%f%z','%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%f','%Y-%m-%dT%H:%M:%S','%Y-%m-%dT%H:%M','%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M','%Y-%m-%d','%m/%d/%Y %H:%M:%S','%m/%d/%Y',
            '%d/%m/%Y %H:%M:%S','%d/%m/%Y',
        ]
        for f in fmts:
            try:
                dt = datetime.strptime(ds, f)
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                pass

        if '+' in ds or ds.endswith('Z'):
            try:
                clean = re.sub(r'[+-]\d{2}:?\d{2}$|Z$', '', ds)
                return self.parse_datetime(clean)
            except Exception:
                pass

        logger.warning(f"Could not parse datetime: {ds}")
        return datetime.now()
