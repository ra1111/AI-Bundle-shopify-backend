"""
CSV Processor Service
Handles parsing and processing of uploaded CSV files
"""
import csv
import io
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from services.storage import storage

logger = logging.getLogger(__name__)

# Default values for NOT NULL database constraints (schema drift fix)
ORDER_NOT_NULL_DEFAULTS = {
    'customer_id': lambda r, oid: r.get('customerId') or f'customer_{oid}',
    'customer_email': lambda r, oid: r.get('customerEmail') or f'customer_{oid}@example.com',
    'customer_country': lambda r, oid: r.get('shippingCountryCode') or 'US',
    'customer_currency': lambda r, oid: r.get('currencyCode') or 'USD',
    'clv_band': lambda r, oid: r.get('clvBand') or 'medium',
    'channel': lambda r, oid: r.get('salesChannel') or 'online',
    'device': lambda r, oid: r.get('deviceType') or 'desktop',
    'discount': lambda r, oid: Decimal(str(r.get('discount', '0') or '0')),
    'discount_code': lambda r, oid: r.get('discountCode') or '',
    'returned': lambda r, oid: False,
    'basket_item_count': lambda r, oid: 1,
    'basket_line_count': lambda r, oid: 1,
    'basket_value': lambda r, oid: Decimal(str(r.get('basketValue', '0') or '0')),
    'financial_status': lambda r, oid: r.get('displayFinancialStatus') or '',
    'fulfillment_status': lambda r, oid: r.get('displayFulfillmentStatus') or '',
}

# Default values for OrderLine table NOT NULL constraints
LINE_NOT_NULL_DEFAULTS = {
    'brand': lambda r: r.get('vendor') or 'unknown',
    'category': lambda r: r.get('productType') or 'general',
    'subcategory': lambda r: r.get('subcategory') or r.get('productSubcategory') or r.get('productType') or 'general',
    'color': lambda r: r.get('color') or 'unspecified',
    'material': lambda r: r.get('material') or 'unspecified',
    'price': lambda r: Decimal(str(r.get('price') or '0')),
    'cost': lambda r: Decimal('0'),
    'weight_kg': lambda r: Decimal('0'),
    'tags': lambda r: r.get('tags', ''),
    'hist_views': lambda r: 0,
    'hist_adds': lambda r: 0,
}

class CSVProcessor:
    """CSV processing service for 4-CSV ingestion model"""
    
    def __init__(self):
        # PR-1: strict column requirements for 4-CSV model
        self.required_columns = {
            # Orders = line-item grain; must include time, qty, price (identifier handled via one-of groups)
            'orders': {'createdAt', 'lineItemQuantity', 'originalUnitPrice'},
            # Products + Variants (one row per variant)
            'products_variants': {'product_id', 'variant_id', 'product_title', 'product_type',
                                  'product_status', 'product_created_at', 'price', 'inventory_item_id'},
            # Raw inventory (per location)
            # NOTE: location_id is required because DB column is NOT NULL
            'inventory_levels': {'inventory_item_id', 'location_id', 'available', 'updated_at'},
            # Catalog joined snapshot (variant + total inventory)
            'catalog_joined': {'product_id', 'variant_id', 'product_title', 'product_type',
                               'product_status', 'product_created_at', 'price', 'inventory_item_id',
                               'available_total', 'last_inventory_update'},
        }
        
        # Some CSVs allow alternative IDs (variantId vs sku). Enforce "one-of" groups.
        self.required_one_of = {
            'orders': [ 
                {'order_id', 'orderId', 'name'},  # Order identifier - Shopify uses various field names
                {'variantId', 'variant_id', 'sku'}  # Variant identifier - handle different field name variations
            ],
            'products_variants': [ {'variant_id', 'sku'} ],
            'catalog_joined': [ {'variant_id', 'sku'} ],
            # inventory_levels has no alt key group
        }
        
        # Numeric fields per type (money as decimals, qty as ints is ok)
        self.numeric_fields = {
            'orders': {'lineItemQuantity', 'originalUnitPrice', 'subtotalPrice',
                       'totalShippingPrice', 'totalTax', 'totalPrice',
                       'discountedTotal', 'lineItemTotalDiscount'},
            'products_variants': {'price', 'compare_at_price'},
            'inventory_levels': {'available'},
            'catalog_joined': {'price', 'compare_at_price', 'available_total'},
        }
        
        # Datetime fields per type
        self.datetime_fields = {
            'orders': {'createdAt'},
            'products_variants': {'product_created_at', 'product_published_at', 'inventory_item_created_at'},
            'inventory_levels': {'updated_at'},
            'catalog_joined': {'product_created_at', 'product_published_at',
                               'inventory_item_created_at', 'last_inventory_update'},
        }
        
        # Schema version tracking
        self.schema_version = 'v2.0'
        
        # Legacy header sets for backward compatibility
        self.orders_headers = {
            'order_id', 'name', 'createdAt', 'currencyCode', 'displayFinancialStatus',
            'displayFulfillmentStatus', 'subtotalPrice', 'totalShippingPrice', 'totalTax',
            'totalPrice', 'customerEmail', 'shippingCountryCode', 'shippingProvince',
            'shippingCity', 'lineItemId', 'lineItemName', 'lineItemQuantity', 'sku',
            'vendor', 'variantTitle', 'productId', 'productTitle', 'productType',
            'variantId', 'originalUnitPrice', 'discountedTotal', 'lineItemTotalDiscount',
            'requiresShipping', 'isGiftCard', 'taxable', 'imageUrl'
        }
        
        self.products_variants_headers = {
            'product_id', 'product_title', 'product_type', 'tags', 'product_status',
            'vendor', 'product_created_at', 'product_published_at', 'variant_id',
            'sku', 'variant_title', 'price', 'compare_at_price', 'inventory_item_id',
            'inventory_item_created_at'
        }
        
        self.inventory_levels_headers = {
            'inventory_item_id', 'location_id', 'available', 'updated_at'
        }
        
        self.catalog_joined_headers = {
            'product_id', 'product_title', 'product_type', 'tags', 'product_status',
            'vendor', 'product_created_at', 'product_published_at', 'variant_id',
            'sku', 'variant_title', 'price', 'compare_at_price', 'inventory_item_id',
            'inventory_item_created_at', 'available_total', 'last_inventory_update'
        }
    
    async def process_csv(self, csv_content: str, upload_id: str, csv_type: str = "auto") -> None:
        """Process CSV content based on 4-CSV model"""
        try:
            logger.info(f"Starting CSV processing for upload {upload_id}, type: {csv_type}")
            
            # Parse CSV content
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(csv_reader)
            
            if not rows:
                raise ValueError("CSV file is empty")
            
            # Update upload record with total rows and schema version
            await storage.update_csv_upload(upload_id, {
                "total_rows": len(rows),
                "processed_rows": 0,
                "schema_version": self.schema_version
            })
            
            # Detect format if auto-detection is enabled
            if csv_type == "auto":
                csv_type = self.detect_csv_type(list(rows[0].keys()))
            
            # Store detected CSV type for observability
            await storage.update_csv_upload(upload_id, {"csv_type": csv_type})
            
            logger.info(f"Processing CSV type: {csv_type}")
            
            # Validate CSV schema strictly before processing
            await self.validate_csv_schema(csv_type, list(rows[0].keys()), upload_id)
            
            # Sample validation on first few rows for data types
            await self.validate_sample_rows(csv_type, rows[:5], upload_id)
            
            # Process based on detected type
            if csv_type == "orders":
                await self.process_orders_csv(rows, upload_id)
            elif csv_type == "products_variants":
                await self.process_products_variants_csv(rows, upload_id)
            elif csv_type == "inventory_levels":
                await self.process_inventory_levels_csv(rows, upload_id)
            elif csv_type == "catalog_joined":
                await self.process_catalog_joined_csv(rows, upload_id)
            else:
                # Fallback to legacy processing
                await self.process_legacy_format(rows, upload_id)
            
            # Mark upload as completed
            await storage.update_csv_upload(upload_id, {
                "status": "completed",
                "processed_rows": len(rows)
            })
            
            logger.info(f"CSV processing completed for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            await storage.update_csv_upload(upload_id, {
                "status": "failed",
                "error_message": str(e)
            })
            raise
    
    def detect_csv_type(self, headers: List[str]) -> str:
        """Detect CSV type from headers for 4-CSV model"""
        headers_set = set(headers)
        
        # Check orders CSV (has both order and line item data)
        orders_matches = len(headers_set.intersection(self.orders_headers))
        if orders_matches >= 15:  # Strong match for orders
            return "orders"
        
        # Check products_variants CSV
        products_matches = len(headers_set.intersection(self.products_variants_headers))
        if products_matches >= 10:
            return "products_variants"
        
        # Check inventory_levels CSV
        inventory_matches = len(headers_set.intersection(self.inventory_levels_headers))
        if inventory_matches >= 3:
            return "inventory_levels"
        
        # Check catalog_joined CSV
        catalog_matches = len(headers_set.intersection(self.catalog_joined_headers))
        if catalog_matches >= 12:
            return "catalog_joined"
        
        return "unknown"
    
    async def validate_csv_schema(self, csv_type: str, headers: List[str], upload_id: str) -> None:
        """PR-1: Enhanced CSV schema validation with strict column requirements and one-of constraints"""
        if csv_type == "unknown":
            error_msg = f"Unrecognized CSV format. Expected one of: orders, products_variants, inventory_levels, catalog_joined. Found headers: {', '.join(headers[:10])}"
            await self._fail_upload_with_error(upload_id, error_msg)
            raise ValueError(error_msg)
        
        required_cols = self.required_columns.get(csv_type, set())
        headers_set = set(headers)
        missing_cols = required_cols - headers_set
        
        # One-of constraints
        one_of_groups = self.required_one_of.get(csv_type, [])
        one_of_failures = []
        for grp in one_of_groups:
            if headers_set.isdisjoint(grp):  # none of the alternatives present
                one_of_failures.append(grp)
        
        if missing_cols or one_of_failures:
            parts = []
            if missing_cols:
                parts.append(f"Missing required: {', '.join(sorted(missing_cols))}")
            for grp in one_of_failures:
                parts.append(f"Need at least one of: {', '.join(sorted(grp))}")
            error_msg = f"{csv_type}.csv schema error → " + " | ".join(parts)
            await self._fail_upload_with_error(upload_id, error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Schema validation passed for {csv_type} CSV with {len(headers)} columns")
    
    async def validate_sample_rows(self, csv_type: str, sample_rows: List[Dict[str, str]], upload_id: str) -> None:
        """Validate sample rows for data type compliance including datetime fields"""
        numeric_fields = self.numeric_fields.get(csv_type, set())
        datetime_fields = self.datetime_fields.get(csv_type, set())
        
        for i, row in enumerate(sample_rows, 1):
            # Validate numeric fields
            for field in numeric_fields:
                if field in row:
                    raw = (row[field] or '').strip()
                    if raw and raw.upper() != 'NULL':
                        try:
                            # Allow integer-ish floats like "1.0" for quantities
                            float(raw)  # accept any numeric; stricter casting happens later
                        except ValueError:
                            error_msg = (f"Invalid numeric value in {csv_type} CSV, row {i}, "
                                         f"column '{field}': '{raw}'. Expected a number.")
                            await self._fail_upload_with_error(upload_id, error_msg)
                            raise ValueError(error_msg)
            
            # Validate datetime fields
            for field in datetime_fields:
                if field in row:
                    raw = (row[field] or '').strip()
                    if raw and raw.upper() not in ['NULL', 'NONE', '']:
                        # Try strict datetime parsing for obviously malformed dates
                        if not self._is_valid_datetime_format(raw):
                            error_msg = (f"Invalid datetime format in {csv_type} CSV, row {i}, "
                                         f"column '{field}': '{raw}'. Expected format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
                            await self._fail_upload_with_error(upload_id, error_msg)
                            raise ValueError(error_msg)
        
        logger.info(f"Sample data validation passed for {csv_type} CSV")
    
    def _is_valid_datetime_format(self, date_str: str) -> bool:
        """Quick validation for obviously invalid datetime formats"""
        if not date_str:
            return True  # Empty is acceptable
        
        # Quick checks for obviously invalid formats
        common_invalid = ['0000-00-00', '1900-01-01', 'N/A', 'NULL', 'null']
        if date_str.lower() in [x.lower() for x in common_invalid]:
            return False
            
        # Basic format check - should contain numbers and separators
        # Match basic datetime patterns (flexible but not permissive)
        datetime_pattern = r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{1,2}(?::\d{1,2})?(?:Z|[+-]\d{2}:?\d{2})?)?$'
        return bool(re.match(datetime_pattern, date_str.strip()))
    
    async def _fail_upload_with_error(self, upload_id: str, error_message: str) -> None:
        """Helper to mark upload as failed with detailed error message"""
        await storage.update_csv_upload(upload_id, {
            "status": "failed",
            "error_message": error_message
        })
        logger.error(f"CSV upload {upload_id} failed: {error_message}")
    
    async def process_orders_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Process orders.csv with order and line item data"""
        orders_map: Dict[str, Dict[str, Any]] = {}
        order_lines: List[Dict[str, Any]] = []
        
        for row in rows:
            try:
                # Process order data - derive order_id from multiple possible field names
                order_id = (row.get('order_id') or row.get('orderId') or row.get('name') or '').strip()
                if not order_id:
                    continue
                
                if order_id not in orders_map:
                    orders_map[order_id] = self.extract_order_data(row, upload_id)
                
                # Process order line data
                line_data = self.extract_order_line_data(row, upload_id, order_id)
                if line_data:
                    line_data = self.sanitize_line(line_data, row)  # Apply line sanitization
                    order_lines.append(self._filter_line_fields(line_data))
                    
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
        
        # Save to database
        if orders_map:
            await storage.create_orders([self._filter_order_fields(o) for o in orders_map.values()])
            logger.info(f"Created {len(orders_map)} orders")
        
        if order_lines:
            await storage.create_order_lines(order_lines)
            logger.info(f"Created {len(order_lines)} order lines")
    
    async def process_legacy_format(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Process legacy/custom format CSV"""
        # For legacy format, we'll create simplified data structure
        # This can be extended based on specific legacy format requirements
        
        orders_map: Dict[str, Dict[str, Any]] = {}
        order_lines: List[Dict[str, Any]] = []
        
        for i, row in enumerate(rows):
            try:
                # Generate order ID if not present
                order_id = row.get('order_id') or f"custom_order_{i}_{upload_id[:8]}"
                
                if order_id not in orders_map:
                    orders_map[order_id] = self.create_default_order(row, upload_id, order_id)
                
                # Create order line from available data
                line_data = self.create_default_order_line(row, upload_id, order_id, i)
                if line_data:
                    order_lines.append(line_data)
                    
            except Exception as e:
                logger.warning(f"Error processing custom row: {e}")
                continue
        
        # Save to database
        if orders_map:
            await storage.create_orders([self._filter_order_fields(o) for o in orders_map.values()])
        if order_lines:
            await storage.create_order_lines(order_lines)
    
    def sanitize_order(self, order_dict: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
        """Apply default values for NOT NULL database constraints to prevent insertion failures"""
        order_id = order_dict.get("order_id", "unknown")
        
        # Apply defaults for known NOT NULL fields using ORDER_NOT_NULL_DEFAULTS
        for field, default_func in ORDER_NOT_NULL_DEFAULTS.items():
            if order_dict.get(field) is None:
                order_dict[field] = default_func(row, order_id)
        
        return order_dict

    def sanitize_line(self, line_dict: Dict[str, Any], row: Dict[str, str]) -> Dict[str, Any]:
        """Apply default values for NOT NULL database constraints on order lines"""
        def empty(v): 
            return v is None or (isinstance(v, str) and v.strip() == "")
        
        out = dict(line_dict)
        
        # Apply defaults for NOT NULL fields
        for field, default_func in LINE_NOT_NULL_DEFAULTS.items():
            if field not in out or empty(out[field]):
                out[field] = default_func(row)
        
        # Ensure numeric fields are proper types and non-null
        for k in ("price", "cost", "weight_kg", "unit_price", "line_total", "line_discount"):
            if k in out and out[k] is None:
                out[k] = Decimal("0")
        
        for k in ("quantity", "hist_views", "hist_adds"):
            if k in out and out[k] is None:
                out[k] = 0
                
        return out

    def _filter_order_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only columns that exist on the Order model to avoid invalid kwargs."""
        allowed = {
            "order_id","csv_upload_id","customer_id","customer_email","customer_country",
            "customer_currency","clv_band","created_at","channel","device","subtotal",
            "discount","discount_code","shipping","taxes","total","financial_status",
            "fulfillment_status","returned","basket_item_count","basket_line_count","basket_value"
        }
        return {k: v for k, v in d.items() if k in allowed}
    
    def extract_order_data(self, row: Dict[str, str], upload_id: str) -> Dict[str, Any]:
        """Extract order header from Shopify-like Orders CSV (line-item grain)."""
        dec = lambda x: Decimal(str(x or '0'))
        # Derive order_id from multiple possible field names used by Shopify exports
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
        
        # Apply sanitization for NOT NULL constraints
        return self._filter_order_fields(self.sanitize_order(order_dict, row))
    
    def extract_order_line_data(self, row: Dict[str, str], upload_id: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Extract order line from Shopify-like Orders CSV."""
        # Must have at least one variant identifier - handle different field name variations
        variant_id = (row.get('variantId') or row.get('variant_id') or '').strip()
        sku = (row.get('sku') or '').strip()
        if not (variant_id or sku):
            return None
        
        # Safe numeric parsing
        def as_int(v: str, default=0) -> int:
            try:
                return int(float(v))
            except Exception:
                return default
        dec = lambda x: Decimal(str(x or '0'))
        
        # Compute line_total with fallback if discountedTotal is missing/blank
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
            # Map vendor/productType into legacy fields used in order_lines
            "brand": row.get('vendor'),
            "category": row.get('productType'),
            "quantity": line_qty,
            "unit_price": dec(row.get('originalUnitPrice')),
            "line_total": line_total,
            "line_discount": dec(row.get('lineItemTotalDiscount')),
        }
    
    def _filter_line_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Filter OrderLine fields to only include valid model fields"""
        allowed = {
            "id","csv_upload_id","order_id","sku","name","brand","category",
            "subcategory","color","material","price","cost","weight_kg","tags",
            "quantity","unit_price","line_total","hist_views","hist_adds",
            "variant_id","line_item_id","line_discount"
        }
        return {k:v for k,v in d.items() if k in allowed}
    
    def extract_product_data(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Extract product data from CSV row"""
        return {
            "sku": row.get('sku', '').strip(),
            "name": row.get('name', 'Unknown Product'),
            "brand": row.get('brand', 'Unknown'),
            "category": row.get('category', 'General'),
            "subcategory": row.get('subcategory', 'General'),
            "color": row.get('color', 'Unknown'),
            "material": row.get('material', 'Unknown'),
            "price": Decimal(str(row.get('price', '0') or '0')),
            "cost": Decimal(str(row.get('cost', '0') or '0')),
            "weight_kg": Decimal(str(row.get('weight_kg', '0') or '0')),
            "tags": row.get('tags', '')
        }
    
    def create_default_order(self, row: Dict[str, str], upload_id: str, order_id: str) -> Dict[str, Any]:
        """Create default order for custom format"""
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
            "subtotal": Decimal(str(row.get('subtotal', '100') or '100')),
            "discount": Decimal('0'),
            "discount_code": None,
            "shipping": Decimal('0'),
            "taxes": Decimal('0'),
            "total": Decimal(str(row.get('total', '100') or '100')),
            "financial_status": row.get('financial_status'),
            "fulfillment_status": row.get('fulfillment_status'),
            "returned": False,
            "basket_item_count": 1,
            "basket_line_count": 1,
            "basket_value": Decimal(str(row.get('total', '100') or '100'))
        }
        
        # Apply sanitization for NOT NULL constraints
        return self._filter_order_fields(self.sanitize_order(order_dict, row))
    
    def create_default_order_line(self, row: Dict[str, str], upload_id: str, order_id: str, index: int) -> Dict[str, Any]:
        """Create default order line for custom format"""
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
            "cost": price * Decimal('0.7'),  # Assume 70% cost ratio
            "weight_kg": Decimal('1.0'),
            "tags": row.get('tags', ''),
            "quantity": quantity,
            "unit_price": price,
            "line_total": price * quantity,
            "hist_views": 0,
            "hist_adds": 0
        }
    
    def create_default_product(self, row: Dict[str, str], sku: str) -> Dict[str, Any]:
        """Create default product for custom format"""
        price = Decimal(str(row.get('price', '100') or '100'))
        
        return {
            "sku": sku,
            "name": row.get('product_name', row.get('name', f'Product {sku}')),
            "brand": row.get('brand', 'Unknown'),
            "category": row.get('category', 'General'),
            "subcategory": row.get('subcategory', 'General'),
            "color": row.get('color', 'Unknown'),
            "material": row.get('material', 'Unknown'),
            "price": price,
            "cost": price * Decimal('0.7'),
            "weight_kg": Decimal('1.0'),
            "tags": row.get('tags', '')
        }
    
    def parse_datetime(self, date_str: Optional[str]) -> datetime:
        """Enhanced datetime parsing with ISO-8601 and timezone support"""
        if not date_str or not date_str.strip():
            return datetime.now()
        
        date_str = date_str.strip()
        
        # Return default for common invalid values
        common_invalid = ['0000-00-00', '1900-01-01', 'N/A', 'NULL', 'null', '', 'nan']
        if date_str.lower() in [x.lower() for x in common_invalid]:
            return datetime.now()
        
        # Enhanced formats including ISO-8601 with timezone support
        formats = [
            # ISO 8601 formats (most common)
            '%Y-%m-%dT%H:%M:%S.%fZ',     # 2023-03-15T14:30:45.123Z
            '%Y-%m-%dT%H:%M:%SZ',        # 2023-03-15T14:30:45Z
            '%Y-%m-%dT%H:%M:%S.%f%z',    # 2023-03-15T14:30:45.123+00:00
            '%Y-%m-%dT%H:%M:%S%z',       # 2023-03-15T14:30:45+00:00
            '%Y-%m-%dT%H:%M:%S.%f',      # 2023-03-15T14:30:45.123
            '%Y-%m-%dT%H:%M:%S',         # 2023-03-15T14:30:45
            '%Y-%m-%dT%H:%M',            # 2023-03-15T14:30
            # Standard datetime formats
            '%Y-%m-%d %H:%M:%S.%f',      # 2023-03-15 14:30:45.123
            '%Y-%m-%d %H:%M:%S',         # 2023-03-15 14:30:45
            '%Y-%m-%d %H:%M',            # 2023-03-15 14:30
            '%Y-%m-%d',                  # 2023-03-15
            # US format
            '%m/%d/%Y %H:%M:%S',         # 03/15/2023 14:30:45
            '%m/%d/%Y',                  # 03/15/2023
            # European format
            '%d/%m/%Y %H:%M:%S',         # 15/03/2023 14:30:45
            '%d/%m/%Y',                  # 15/03/2023
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Convert timezone-aware datetimes to UTC naive for database storage
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue
        
        # Special handling for timezone-aware strings ending with timezone codes
        if '+' in date_str or date_str.endswith('Z'):
            try:
                # Try to parse and ignore timezone for now
                clean_date = re.sub(r'[+-]\d{2}:?\d{2}$|Z$', '', date_str)
                return self.parse_datetime(clean_date)
            except Exception:
                pass
        
        # If all formats fail, return current datetime with warning
        logger.warning(f"Could not parse datetime: {date_str}")
        return datetime.now()
    
    async def process_products_variants_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Process products_variants.csv (product + variant metadata)"""
        products = []
        variants = []
        
        for row in rows:
            try:
                # Handle special cases
                product_status = row.get('product_status', '').upper()
                if product_status in ['ARCHIVED', 'DRAFT']:
                    continue  # Exclude from bundle generation
                
                price = Decimal(str(row.get('price', '0') or '0'))
                if price == 0:
                    # Mark for exclusion/lead magnet but still process
                    logger.info(f"Zero price product detected: {row.get('product_id')}")
                
                # Extract product data
                product_data = {
                    "product_id": row.get('product_id', '').strip(),
                    "csv_upload_id": upload_id,
                    "product_title": row.get('product_title', 'Unknown Product'),
                    "product_type": row.get('product_type', ''),
                    "tags": row.get('tags', ''),
                    "product_status": product_status,
                    "vendor": row.get('vendor', ''),
                    "product_created_at": self.parse_datetime(row.get('product_created_at')),
                    "product_published_at": self.parse_datetime(row.get('product_published_at')),
                    # Legacy compatibility fields
                    "sku": row.get('sku', ''),
                    "name": row.get('product_title', 'Unknown Product'),
                    "brand": row.get('vendor', ''),
                    "category": row.get('product_type', 'General'),
                    "price": price
                }
                
                products.append(product_data)
                
                # Extract variant data
                variant_data = {
                    "variant_id": row.get('variant_id', '').strip(),
                    "csv_upload_id": upload_id,
                    "product_id": row.get('product_id', '').strip(),
                    "sku": row.get('sku', ''),
                    "variant_title": row.get('variant_title', 'Default Title'),
                    "price": price,
                    "compare_at_price": Decimal(str(row.get('compare_at_price', '0') or '0')),
                    "inventory_item_id": row.get('inventory_item_id', '').strip(),
                    "inventory_item_created_at": self.parse_datetime(row.get('inventory_item_created_at'))
                }
                
                variants.append(variant_data)
                
            except Exception as e:
                logger.warning(f"Error processing products_variants row: {e}")
                continue
        
        # Save to database
        if products:
            await storage.create_products(products)
            logger.info(f"Created {len(products)} products")
        
        if variants:
            await storage.create_variants(variants)
            logger.info(f"Created {len(variants)} variants")
    
    async def process_inventory_levels_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Process inventory_levels.csv with raw inventory per location"""
        inventory_levels = []
        
        for row in rows:
            try:
                # Robust parsing of available field to handle floats and empty values
                raw_available = (row.get('available', '') or '').strip()
                try:
                    available = int(float(raw_available)) if raw_available != '' else 0
                except (ValueError, TypeError):
                    available = 0
                
                # Handle special case: available = -1 means inventory not tracked
                if available == -1:
                    logger.info(f"Inventory not tracked for item: {row.get('inventory_item_id')}")
                
                # Validate location_id (DB is NOT NULL) - skip invalid rows
                location_id = (row.get('location_id') or '').strip()
                if not location_id:
                    logger.warning(f"Skipping inventory_levels row with missing location_id: {row.get('inventory_item_id', 'unknown')}")
                    continue
                
                inventory_data = {
                    "inventory_item_id": row.get('inventory_item_id', '').strip(),
                    "csv_upload_id": upload_id,
                    "location_id": location_id,
                    "available": available,
                    "updated_at": self.parse_datetime(row.get('updated_at'))
                }
                
                inventory_levels.append(inventory_data)
                
            except Exception as e:
                logger.warning(f"Error processing inventory_levels row: {e}")
                continue
        
        # Save to database
        if inventory_levels:
            await storage.create_inventory_levels(inventory_levels)
            logger.info(f"Created {len(inventory_levels)} inventory level records")
    
    async def process_catalog_joined_csv(self, rows: List[Dict[str, str]], upload_id: str) -> None:
        """Process catalog_joined.csv (flattened product+variant+inventory snapshot)"""
        catalog_snapshots = []
        
        for row in rows:
            try:
                # Handle special cases
                product_status = row.get('product_status', '').upper()
                if product_status in ['ARCHIVED', 'DRAFT']:
                    continue  # Exclude from bundle generation
                
                price = Decimal(str(row.get('price', '0') or '0'))
                if price == 0:
                    logger.info(f"Zero price product in catalog: {row.get('product_id')}")
                
                available_total = row.get('available_total', '')
                if available_total == '' or available_total is None:
                    available_total = None
                else:
                    try:
                        # Handle decimal values by converting to float first
                        available_total = int(float(available_total))
                        if available_total == -1:
                            available_total = None  # Inventory not tracked
                    except (ValueError, TypeError):
                        available_total = None
                
                # Compute objective flags for later use
                is_slow_mover = False  # Will be computed after order processing
                is_new_launch = False
                is_seasonal = False
                is_high_margin = False
                is_gift_card = False  # Default value as requested
                
                # Check if product is new launch (created within last 30 days)
                created_at = self.parse_datetime(row.get('product_created_at'))
                if created_at:
                    days_since_creation = (datetime.now() - created_at).days
                    is_new_launch = days_since_creation <= 30
                
                # Check if seasonal based on tags
                tags = row.get('tags', '').lower()
                seasonal_keywords = ['holiday', 'festive', 'gift', 'christmas', 'halloween', 'valentine']
                is_seasonal = any(keyword in tags for keyword in seasonal_keywords)
                
                # Enhanced gift card detection
                product_type = row.get('product_type', '').lower()
                is_gift_card = (
                    'gift card' in product_type or 
                    'giftcard' in product_type or
                    'gift-card' in product_type or
                    ('gift' in tags and ('card' in tags or 'certificate' in tags))
                )
                
                # Apply guardrails: flag gift cards and zero-price variants for conditional inclusion
                # Bundle generator will decide inclusion based on objective type
                if is_gift_card:
                    logger.info(f"Gift card flagged in catalog: {row.get('product_title')}")
                elif price == 0:
                    logger.info(f"Zero-price variant flagged in catalog: {row.get('product_title')}")
                
                # Check high margin (simplified - compare_at_price vs price)
                compare_at_price = Decimal(str(row.get('compare_at_price', '0') or '0'))
                if compare_at_price > 0 and price > 0:
                    margin = (compare_at_price - price) / compare_at_price
                    is_high_margin = margin > 0.3  # High margin threshold (30%)
                else:
                    is_high_margin = False
                
                # Calculate avg_discount_pct on the fly as requested
                avg_discount_pct = 0.0
                if compare_at_price > 0 and price > 0:
                    discount = (compare_at_price - price) / compare_at_price * 100
                    avg_discount_pct = round(float(discount), 2)
                
                catalog_data = {
                    "product_id": row.get('product_id', '').strip(),
                    "csv_upload_id": upload_id,
                    "product_title": row.get('product_title', 'Unknown Product'),
                    "product_type": row.get('product_type', ''),
                    "tags": row.get('tags', ''),
                    "product_status": product_status,
                    "vendor": row.get('vendor', ''),
                    "product_created_at": created_at,
                    "product_published_at": self.parse_datetime(row.get('product_published_at')),
                    "variant_id": row.get('variant_id', '').strip(),
                    "sku": row.get('sku', ''),
                    "variant_title": row.get('variant_title', 'Default Title'),
                    "price": price,
                    "compare_at_price": compare_at_price,
                    "inventory_item_id": row.get('inventory_item_id', '').strip(),
                    "inventory_item_created_at": self.parse_datetime(row.get('inventory_item_created_at')),
                    "available_total": available_total,
                    "last_inventory_update": self.parse_datetime(row.get('last_inventory_update')),
                    # Computed objective flags (only fields that exist in database)
                    "is_slow_mover": is_slow_mover,  # Will be computed later from orders
                    "is_new_launch": is_new_launch,
                    "is_seasonal": is_seasonal,
                    "is_high_margin": is_high_margin,
                    # Note: is_gift_card, avg_discount_pct, objective_flags not in database
                    # Keeping calculations for internal logic but not storing in DB
                }
                
                catalog_snapshots.append(catalog_data)
                
            except Exception as e:
                logger.warning(f"Error processing catalog_joined row: {e}")
                continue
        
        # Save to database
        if catalog_snapshots:
            await storage.create_catalog_snapshots(catalog_snapshots)
            logger.info(f"Created {len(catalog_snapshots)} catalog snapshot records")
            
            # Recompute objective flags based on order data (90-day lookback)
            try:
                await storage.recompute_catalog_objectives(upload_id)
                logger.info(f"Recomputed objective flags for upload {upload_id}")
            except Exception as e:
                logger.error(f"Failed to recompute objective flags for upload {upload_id}: {e}")
                # Don't fail the entire upload for objective computation errors
                pass