"""
Data Mapping Service
Handles comprehensive linking of OrderLine.sku → Variant → Product → Inventory
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
from decimal import Decimal
from datetime import datetime, timedelta

from services.storage import storage

logger = logging.getLogger(__name__)

class DataMapper:
    """Enhanced data mapping service for OrderLine → Variant → Product → Inventory links"""
    
    def __init__(self):
        self.resolved_variant_cache: Dict[str, str] = {}
        self.unresolved_sku_cache: set = set()
    
    async def resolve_variant_from_sku(self, sku: str, csv_upload_id: str) -> Optional[str]:
        """Resolve variant_id from SKU with caching"""
        if sku in self.resolved_variant_cache:
            return self.resolved_variant_cache[sku]
        
        if sku in self.unresolved_sku_cache:
            return None
        
        # Try to find variant by SKU
        variant = await storage.get_variant_by_sku(sku, csv_upload_id)
        if variant:
            variant_id = variant.variant_id
            self.resolved_variant_cache[sku] = variant_id
            return variant_id
        
        # Cache unresolved SKUs to avoid repeated lookups
        self.unresolved_sku_cache.add(sku)
        logger.warning(f"Could not resolve variant_id for SKU: {sku}")
        return None
    
    async def enrich_order_lines_with_variants(self, csv_upload_id: str) -> Dict[str, Any]:
        """Enrich order lines with variant mappings and compute metrics"""
        metrics = {
            "total_order_lines": 0,
            "resolved_variants": 0,
            "unresolved_skus": 0,
            "missing_inventory": 0,
            "enriched_lines": 0
        }
        
        try:
            # Resolve run_id to correlate across files
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            # Get all order lines for this upload (or entire run if available)
            if run_id:
                order_lines = await storage.get_order_lines_by_run(run_id)
            else:
                order_lines = await storage.get_order_lines_by_upload(csv_upload_id)
            metrics["total_order_lines"] = len(order_lines)
            
            enriched_lines = []
            
            for line in order_lines:
                sku = getattr(line, "sku", None)
                line_vid = getattr(line, "variant_id", None)

                resolved_variant_id: Optional[str] = None
                resolved_variant_obj = None

                if run_id:
                    if sku:
                        resolved_variant_obj = await storage.get_variant_by_sku_run(sku, run_id)
                    if not resolved_variant_obj and line_vid:
                        resolved_variant_obj = await storage.get_variant_by_id_run(line_vid, run_id)
                else:
                    if sku:
                        vid = await self.resolve_variant_from_sku(sku, csv_upload_id)
                        if vid:
                            resolved_variant_id = vid
                    if not resolved_variant_id and line_vid:
                        v = await storage.get_variant_by_id(line_vid, csv_upload_id)
                        if v:
                            resolved_variant_obj = v

                if resolved_variant_obj and not resolved_variant_id:
                    resolved_variant_id = resolved_variant_obj.variant_id

                if not resolved_variant_id:
                    metrics["unresolved_skus"] += 1
                    continue

                metrics["resolved_variants"] += 1

                # Product + inventory enrichment
                product_data = None
                inventory_data = {"available_total": -1, "location_count": 0}

                if run_id:
                    use_sku = sku or (resolved_variant_obj.sku if resolved_variant_obj else None)
                    try:
                        if use_sku:
                            catalog_map = await storage.get_catalog_snapshots_map_by_run(run_id)
                            if catalog_map and use_sku in catalog_map:
                                pd = catalog_map[use_sku]
                                product_data = {
                                    "product_id": pd.product_id,
                                    "product_title": pd.product_title,
                                    "product_type": pd.product_type,
                                    "vendor": pd.vendor,
                                    "tags": pd.tags,
                                    "product_status": pd.product_status,
                                    "created_at": pd.product_created_at,
                                    "published_at": pd.product_published_at,
                                    "variant_title": pd.variant_title,
                                    "price": pd.price,
                                    "compare_at_price": pd.compare_at_price,
                                }
                    except Exception:
                        pass

                    if resolved_variant_obj and resolved_variant_obj.inventory_item_id:
                        inv_levels = await storage.get_inventory_levels_by_item_id_run(
                            resolved_variant_obj.inventory_item_id, run_id
                        )
                        if inv_levels:
                            total_available = sum(l.available for l in inv_levels)
                            inventory_data = {"available_total": total_available, "location_count": len(inv_levels)}
                else:
                    product_data = await self.get_product_data_for_variant(resolved_variant_id, csv_upload_id)
                    inv_tmp = await self.get_inventory_data_for_variant(resolved_variant_id, csv_upload_id)
                    if inv_tmp:
                        inventory_data = inv_tmp

                # Carry inventory_item_id forward if available for persistence
                inv_item_id = getattr(resolved_variant_obj, 'inventory_item_id', None) if resolved_variant_obj else None

                enriched_line = {
                    "order_line_id": line.id,
                    "sku": sku,
                    "variant_id": resolved_variant_id,
                    "inventory_item_id": inv_item_id,
                    "product_data": product_data,
                    "inventory_data": inventory_data,
                    "line_data": {
                        "quantity": line.quantity,
                        "unit_price": line.unit_price,
                        "line_total": line.line_total,
                        "name": line.name,
                        "category": line.category,
                        "brand": line.brand
                    }
                }

                if inventory_data:
                    enriched_line["stock_available"] = inventory_data.get("available_total", 0)
                else:
                    metrics["missing_inventory"] += 1
                    enriched_line["stock_available"] = -1

                enriched_lines.append(enriched_line)
                metrics["enriched_lines"] += 1
            
            # Store enriched data for later use
            await self.persist_enriched_mappings(csv_upload_id, enriched_lines)
            
            logger.info(f"Data mapping completed: {metrics}")
            return {
                "metrics": metrics,
                "enriched_lines": enriched_lines
            }
            
        except Exception as e:
            logger.error(f"Error in data mapping: {e}")
            raise
    
    async def get_product_data_for_variant(self, variant_id: str, csv_upload_id: str) -> Optional[Dict[str, Any]]:
        """Get product data for a variant"""
        try:
            variant = await storage.get_variant_by_id(variant_id, csv_upload_id)
            if not variant:
                return None
            
            # Try variant_id first (architect's recommendation), skip synthetic SKUs
            if variant.sku and variant.sku.startswith("no-sku-"):
                logger.warning(f"Skipping synthetic SKU: {variant.sku}")
                return None
                
            # Get product data from catalog_snapshot using variant.sku as fallback
            product_data = None
            if variant.sku:
                # ARCHITECT FIX: Use preloaded catalog_map instead of unsafe per-SKU queries
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
                product_data = catalog_map.get(variant.sku) if catalog_map else None
            if not product_data:
                return None
            
            return {
                "product_id": product_data.product_id,
                "product_title": product_data.product_title,
                "product_type": product_data.product_type,
                "vendor": product_data.vendor,
                "tags": product_data.tags,
                "product_status": product_data.product_status,
                "created_at": product_data.product_created_at,
                "published_at": product_data.product_published_at,
                "variant_title": product_data.variant_title,
                "price": product_data.price,
                "compare_at_price": product_data.compare_at_price
            }
            
        except Exception as e:
            logger.warning(f"Error getting product data for variant {variant_id}: {e}")
            return None
    
    async def get_inventory_data_for_variant(self, variant_id: str, csv_upload_id: str) -> Optional[Dict[str, Any]]:
        """Get inventory data for a variant"""
        try:
            variant = await storage.get_variant_by_id(variant_id, csv_upload_id)
            if not variant or not variant.inventory_item_id:
                return None
            
            inventory_levels = await storage.get_inventory_levels_by_item_id(
                variant.inventory_item_id, csv_upload_id
            )
            
            if not inventory_levels:
                return {"available_total": -1, "location_count": 0}  # Not tracked
            
            total_available = sum(level.available for level in inventory_levels)
            
            return {
                "available_total": total_available,
                "location_count": len(inventory_levels),
                "locations": [
                    {
                        "location_id": level.location_id,
                        "available": level.available
                    } for level in inventory_levels
                ]
            }
            
        except Exception as e:
            logger.warning(f"Error getting inventory data for variant {variant_id}: {e}")
            return None
    
    async def persist_enriched_mappings(self, csv_upload_id: str, enriched_lines: List[Dict[str, Any]]) -> None:
        """Persist enriched mappings for later use in bundle generation"""
        try:
            # Store in catalog_snapshot for fast access during bundle generation
            catalog_entries = []
            
            for line in enriched_lines:
                if line.get("variant_id") and line.get("product_data"):
                    product_data = line["product_data"]
                    inventory_data = line.get("inventory_data", {})
                    
                    # Compute velocity and other signals
                    velocity = await self.compute_velocity_for_variant(
                        line["variant_id"], csv_upload_id
                    )
                    
                    catalog_entry = {
                        "variant_id": line["variant_id"],
                        "csv_upload_id": csv_upload_id,
                        "sku": line["sku"],
                        "product_id": product_data["product_id"],
                        "product_title": product_data["product_title"],
                        "product_type": product_data["product_type"],
                        "vendor": product_data["vendor"],
                        "tags": product_data["tags"],
                        "product_status": product_data["product_status"],
                        "product_created_at": product_data.get("created_at"),
                        "product_published_at": product_data.get("published_at"),
                        "variant_title": product_data["variant_title"],
                        "price": product_data["price"],
                        "compare_at_price": product_data["compare_at_price"],
                        "inventory_item_id": line.get("inventory_item_id") or "",
                        "available_total": inventory_data.get("available_total", -1),
                        "last_inventory_update": datetime.now(),
                        # Note: objective flags and velocity are computed/updated elsewhere
                    }
                    
                    catalog_entries.append(catalog_entry)
            
            if catalog_entries:
                await storage.create_catalog_snapshots(catalog_entries)
                logger.info(f"Persisted {len(catalog_entries)} catalog entries")
            
        except Exception as e:
            logger.error(f"Error persisting enriched mappings: {e}")
            raise
    
    async def compute_velocity_for_variant(self, variant_id: str, csv_upload_id: str) -> Decimal:
        """Compute sales velocity for variant (units sold per day)"""
        try:
            # Get sales data for last 60 days
            sales_data = await storage.get_variant_sales_data(variant_id, csv_upload_id, days=60)
            if not sales_data:
                return Decimal('0')
            
            total_quantity = sum(sale.quantity for sale in sales_data if sale.quantity is not None)
            days_period = 60
            
            return Decimal(str(total_quantity)) / Decimal(str(days_period))
            
        except Exception as e:
            logger.warning(f"Error computing velocity for variant {variant_id}: {e}")
            return Decimal('0')
    
    def get_resolution_metrics(self) -> Dict[str, Any]:
        """Get data mapping resolution metrics"""
        return {
            "resolved_variants_cached": len(self.resolved_variant_cache),
            "unresolved_skus_cached": len(self.unresolved_sku_cache),
            "cache_hit_rate": len(self.resolved_variant_cache) / max(1, 
                len(self.resolved_variant_cache) + len(self.unresolved_sku_cache))
        }
