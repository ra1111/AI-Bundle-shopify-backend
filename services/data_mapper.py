"""
Data Mapping Service
Handles comprehensive linking of OrderLine.sku → Variant → Product → Inventory
"""
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timedelta
import time

from services.feature_flags import feature_flags

from services.storage import storage

logger = logging.getLogger(__name__)

class DataMapper:
    """Enhanced data mapping service for OrderLine → Variant → Product → Inventory links"""
    
    def __init__(self):
        self.resolved_variant_cache: Dict[str, str] = {}
        self.unresolved_sku_cache: Dict[str, datetime] = {}
        self._last_unresolved_log_at: Optional[datetime] = None
        self._variant_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._variant_id_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._inventory_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._catalog_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._product_meta_cache: Dict[str, Dict[str, Any]] = {}

    def _scope_key(self, csv_upload_id: str, run_id: Optional[str] = None) -> str:
        return f"run:{run_id}" if run_id else f"upload:{csv_upload_id}"

    async def resolve_variant_from_sku(self, sku: str, csv_upload_id: str) -> Optional[str]:
        """Resolve variant_id from SKU with caching"""
        if sku in self.resolved_variant_cache:
            return self.resolved_variant_cache[sku]

        scope_key = self._scope_key(csv_upload_id)
        scoped_variants = self._variant_map_by_scope.get(scope_key)
        if scoped_variants:
            cached_variant = scoped_variants.get(sku)
            if cached_variant and getattr(cached_variant, "variant_id", None):
                variant_id = cached_variant.variant_id
                self.resolved_variant_cache[sku] = variant_id
                return variant_id

        ttl_seconds = feature_flags.get_flag("data_mapping.unresolved_cache_ttl_seconds", 600) or 0
        cached_at = self.unresolved_sku_cache.get(sku)
        if cached_at:
            if ttl_seconds > 0:
                age = (datetime.utcnow() - cached_at).total_seconds()
                if age > ttl_seconds:
                    self.unresolved_sku_cache.pop(sku, None)
                else:
                    return None
            else:
                return None

        # Try to find variant by SKU
        variant = await storage.get_variant_by_sku(sku, csv_upload_id)
        if variant:
            variant_id = variant.variant_id
            self.resolved_variant_cache[sku] = variant_id
            return variant_id

        # Cache unresolved SKUs to avoid repeated lookups
        self.unresolved_sku_cache[sku] = datetime.utcnow()
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

        unresolved_samples: List[str] = []
        timing_enabled = feature_flags.get_flag("data_mapping.log_timings", True)
        overall_start = time.perf_counter() if timing_enabled else None
        fetch_start = time.perf_counter() if timing_enabled else None
        fetch_duration = 0.0
        prefetch_duration = 0.0
        mapping_duration = 0.0
        
        try:
            # Resolve run_id to correlate across files
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            # Get all order lines for this upload (or entire run if available)
            if run_id:
                order_lines = await storage.get_order_lines_by_run(run_id)
            else:
                order_lines = await storage.get_order_lines_by_upload(csv_upload_id)

            if timing_enabled and fetch_start is not None:
                fetch_duration = time.perf_counter() - fetch_start

            metrics["total_order_lines"] = len(order_lines)
            
            if not order_lines:
                logger.info(f"Data mapping completed: {metrics}")
                return {
                    "metrics": metrics,
                    "enriched_lines": []
                }

            prefetch_start = time.perf_counter() if timing_enabled else None
            prefetch_data = await self._prefetch_data(csv_upload_id, run_id, order_lines)
            if timing_enabled and prefetch_start is not None:
                prefetch_duration = time.perf_counter() - prefetch_start

            mapping_start = time.perf_counter() if timing_enabled else None

            concurrency_enabled = feature_flags.get_flag("data_mapping.concurrent_mapping", True)
            if concurrency_enabled and order_lines:
                concurrency_limit = feature_flags.get_flag("data_mapping.concurrent_map_limit", 25) or 25
                try:
                    concurrency_limit = int(concurrency_limit)
                except (TypeError, ValueError):
                    concurrency_limit = 25
                concurrency_limit = max(1, concurrency_limit)
                semaphore = asyncio.Semaphore(concurrency_limit)

                async def map_line(item: Tuple[int, Any]) -> Dict[str, Any]:
                    idx, order_line = item
                    async with semaphore:
                        return await self._map_order_line(idx, order_line, csv_upload_id, run_id, prefetch_data)

                results = await asyncio.gather(*[map_line(item) for item in enumerate(order_lines)])
            else:
                results = []
                for idx, order_line in enumerate(order_lines):
                    results.append(await self._map_order_line(idx, order_line, csv_upload_id, run_id, prefetch_data))

            if timing_enabled and mapping_start is not None:
                mapping_duration = time.perf_counter() - mapping_start

            results.sort(key=lambda r: r["index"])

            enriched_lines: List[Dict[str, Any]] = []

            for result in results:
                local_metrics = result.get("metrics", {})
                metrics["resolved_variants"] += local_metrics.get("resolved_variants", 0)
                metrics["unresolved_skus"] += local_metrics.get("unresolved_skus", 0)
                metrics["missing_inventory"] += local_metrics.get("missing_inventory", 0)
                metrics["enriched_lines"] += local_metrics.get("enriched_lines", 0)

                sample = result.get("unresolved_sample")
                if sample and len(unresolved_samples) < 10:
                    unresolved_samples.append(sample)

                enriched_line = result.get("enriched_line")
                if enriched_line:
                    enriched_lines.append(enriched_line)
            
            # Store enriched data for later use
            await self.persist_enriched_mappings(csv_upload_id, enriched_lines)
            
            if metrics["unresolved_skus"]:
                self._log_unresolved_summary(csv_upload_id, metrics["unresolved_skus"], unresolved_samples)

            if timing_enabled and overall_start is not None:
                total_duration = time.perf_counter() - overall_start
                logger.info(
                    "Data mapping timings | upload=%s total=%.2fs fetch=%.2fs prefetch=%.2fs map=%.2fs lines=%d",
                    csv_upload_id,
                    total_duration,
                    fetch_duration,
                    prefetch_duration,
                    mapping_duration,
                    len(order_lines),
                )

            logger.info(f"Data mapping completed: {metrics}")
            return {
                "metrics": metrics,
                "enriched_lines": enriched_lines
            }
            
        except Exception as e:
            logger.error(f"Error in data mapping: {e}")
            raise

    async def _prefetch_data(self, csv_upload_id: str, run_id: Optional[str], _order_lines: List[Any]) -> Dict[str, Any]:
        """Prefetch catalog, variant, and inventory data to minimize per-row I/O."""
        scope = self._scope_key(csv_upload_id, run_id)
        if not feature_flags.get_flag("data_mapping.prefetch_enabled", True):
            return {
                "scope": scope,
                "variants_by_sku": self._variant_map_by_scope.get(scope, {}),
                "variants_by_id": self._variant_id_map_by_scope.get(scope, {}),
                "inventory_map": self._inventory_map_by_scope.get(scope, {}),
                "catalog_map": self._catalog_map_by_scope.get(scope, {}),
            }

        tasks: Dict[str, asyncio.Task] = {}

        if scope not in self._variant_map_by_scope or scope not in self._variant_id_map_by_scope:
            tasks["variant_maps"] = asyncio.create_task(
                storage.get_variant_maps_by_run(run_id) if run_id else storage.get_variant_maps(csv_upload_id)
            )

        if scope not in self._inventory_map_by_scope:
            tasks["inventory_map"] = asyncio.create_task(
                storage.get_inventory_levels_map_by_run(run_id) if run_id else storage.get_inventory_levels_map(csv_upload_id)
            )

        if scope not in self._catalog_map_by_scope:
            tasks["catalog_map"] = asyncio.create_task(
                storage.get_catalog_snapshots_map_by_run(run_id) if run_id else storage.get_catalog_snapshots_map(csv_upload_id)
            )

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(
                        "DataMapper prefetch %s failed for upload=%s run=%s: %s",
                        key,
                        csv_upload_id,
                        run_id,
                        result,
                    )
                    continue
                if key == "variant_maps":
                    by_sku, by_id = result
                    self._variant_map_by_scope[scope] = by_sku
                    self._variant_id_map_by_scope[scope] = by_id
                elif key == "inventory_map":
                    self._inventory_map_by_scope[scope] = self._build_inventory_data_map(result)
                elif key == "catalog_map":
                    self._catalog_map_by_scope[scope] = result

        catalog_map = self._catalog_map_by_scope.get(scope, {})
        if catalog_map:
            for sku, snapshot in catalog_map.items():
                if not sku:
                    continue
                cache_key = self._product_cache_key(scope, sku)
                if cache_key not in self._product_meta_cache:
                    self._product_meta_cache[cache_key] = self._convert_snapshot_to_product_data(snapshot)

        return {
            "scope": scope,
            "variants_by_sku": self._variant_map_by_scope.get(scope, {}),
            "variants_by_id": self._variant_id_map_by_scope.get(scope, {}),
            "inventory_map": self._inventory_map_by_scope.get(scope, {}),
            "catalog_map": catalog_map,
        }

    async def _map_order_line(
        self,
        index: int,
        line: Any,
        csv_upload_id: str,
        run_id: Optional[str],
        prefetch_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single order line with optional prefetched data."""
        metrics = {
            "resolved_variants": 0,
            "unresolved_skus": 0,
            "missing_inventory": 0,
            "enriched_lines": 0,
        }
        scope = prefetch_data.get("scope", self._scope_key(csv_upload_id, run_id))
        base_variants_by_sku = prefetch_data.get("variants_by_sku", {})
        base_variants_by_id = prefetch_data.get("variants_by_id", {})
        base_inventory_map = prefetch_data.get("inventory_map", {})
        catalog_map = prefetch_data.get("catalog_map", {})

        scope_variant_map = self._variant_map_by_scope.setdefault(scope, base_variants_by_sku)
        scope_variant_id_map = self._variant_id_map_by_scope.setdefault(scope, base_variants_by_id)
        scope_inventory_map = self._inventory_map_by_scope.setdefault(scope, base_inventory_map)

        sku = getattr(line, "sku", None)
        line_vid = getattr(line, "variant_id", None)

        resolved_variant_obj = None
        resolved_variant_id: Optional[str] = None

        if run_id:
            if sku:
                resolved_variant_obj = scope_variant_map.get(sku)
            if not resolved_variant_obj and line_vid:
                resolved_variant_obj = scope_variant_id_map.get(line_vid)
            if feature_flags.get_flag("data_mapping.enable_run_scope_fallback", True) and not resolved_variant_obj:
                if sku:
                    resolved_variant_obj = await storage.get_variant_by_sku_run(sku, run_id)
                if not resolved_variant_obj and line_vid:
                    resolved_variant_obj = await storage.get_variant_by_id_run(line_vid, run_id)
                if not resolved_variant_obj and sku:
                    resolved_variant_obj = await storage.get_variant_by_sku(sku, csv_upload_id)
            if not resolved_variant_obj and line_vid:
                resolved_variant_obj = await storage.get_variant_by_id(line_vid, csv_upload_id)
        else:
            if sku:
                resolved_variant_obj = scope_variant_map.get(sku)
            if not resolved_variant_obj and line_vid:
                resolved_variant_obj = scope_variant_id_map.get(line_vid)
            if not resolved_variant_obj and sku:
                variant_id = await self.resolve_variant_from_sku(sku, csv_upload_id)
                if variant_id:
                    resolved_variant_id = variant_id
            if not resolved_variant_obj and resolved_variant_id:
                resolved_variant_obj = scope_variant_id_map.get(resolved_variant_id) or await storage.get_variant_by_id(
                    resolved_variant_id, csv_upload_id
                )
            if not resolved_variant_obj and line_vid:
                resolved_variant_obj = await storage.get_variant_by_id(line_vid, csv_upload_id)

        if resolved_variant_obj and not resolved_variant_id:
            resolved_variant_id = getattr(resolved_variant_obj, "variant_id", None)

        if not resolved_variant_id and line_vid:
            resolved_variant_id = line_vid

        if not resolved_variant_id:
            metrics["unresolved_skus"] += 1
            sample = sku or line_vid
            return {
                "index": index,
                "metrics": metrics,
                "unresolved_sample": sample,
                "enriched_line": None,
            }

        metrics["resolved_variants"] += 1

        if resolved_variant_obj:
            scope_variant_id_map.setdefault(resolved_variant_id, resolved_variant_obj)
            variant_sku = getattr(resolved_variant_obj, "sku", None)
            if variant_sku:
                scope_variant_map.setdefault(variant_sku, resolved_variant_obj)

        if sku:
            self.resolved_variant_cache[sku] = resolved_variant_id

        product_data = await self._get_product_data(
            sku or (getattr(resolved_variant_obj, "sku", None) if resolved_variant_obj else None),
            resolved_variant_id,
            csv_upload_id,
            scope,
            catalog_map,
        )

        inventory_item_id = getattr(resolved_variant_obj, "inventory_item_id", None) if resolved_variant_obj else None
        inventory_data = await self._get_inventory_data(
            inventory_item_id,
            resolved_variant_id,
            csv_upload_id,
            run_id,
            scope,
            scope_inventory_map,
        )

        if inventory_item_id and inventory_data:
            scope_inventory_map[inventory_item_id] = inventory_data

        if inventory_data is None:
            metrics["missing_inventory"] += 1
            inventory_data = {"available_total": -1, "location_count": 0}
            stock_available = -1
        else:
            stock_available = inventory_data.get("available_total", 0)

        enriched_line = {
            "order_line_id": line.id,
            "sku": sku,
            "variant_id": resolved_variant_id,
            "inventory_item_id": inventory_item_id,
            "product_data": product_data,
            "inventory_data": inventory_data,
            "line_data": {
                "quantity": getattr(line, "quantity", None),
                "unit_price": getattr(line, "unit_price", None),
                "line_total": getattr(line, "line_total", None),
                "name": getattr(line, "name", None),
                "category": getattr(line, "category", None),
                "brand": getattr(line, "brand", None),
            },
            "stock_available": stock_available,
        }

        metrics["enriched_lines"] += 1

        return {
            "index": index,
            "metrics": metrics,
            "unresolved_sample": None,
            "enriched_line": enriched_line,
        }

    def _product_cache_key(self, scope: str, sku: str) -> str:
        return f"{scope}::{sku}"

    def _convert_snapshot_to_product_data(self, snapshot: Any) -> Dict[str, Any]:
        if snapshot is None:
            return {}
        return {
            "product_id": getattr(snapshot, "product_id", None),
            "product_title": getattr(snapshot, "product_title", None),
            "product_type": getattr(snapshot, "product_type", None),
            "vendor": getattr(snapshot, "vendor", None),
            "tags": getattr(snapshot, "tags", None),
            "product_status": getattr(snapshot, "product_status", None),
            "created_at": getattr(snapshot, "product_created_at", None),
            "published_at": getattr(snapshot, "product_published_at", None),
            "variant_title": getattr(snapshot, "variant_title", None),
            "price": getattr(snapshot, "price", None),
            "compare_at_price": getattr(snapshot, "compare_at_price", None),
        }

    def _build_inventory_data_map(self, level_map: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        inventory_data: Dict[str, Dict[str, Any]] = {}
        for item_id, levels in level_map.items():
            inventory_data[item_id] = self._convert_inventory_levels(levels)
        return inventory_data

    def _convert_inventory_levels(self, levels: Any) -> Dict[str, Any]:
        if isinstance(levels, dict):
            return levels
        if not levels:
            return {"available_total": -1, "location_count": 0, "locations": []}
        total_available = 0
        locations = []
        for level in levels:
            available = getattr(level, "available", None)
            if available is None:
                available = 0
            total_available += available
            locations.append(
                {
                    "location_id": getattr(level, "location_id", None),
                    "available": available,
                }
            )
        return {
            "available_total": total_available,
            "location_count": len(levels),
            "locations": locations,
        }

    async def _get_product_data(
        self,
        sku: Optional[str],
        variant_id: Optional[str],
        csv_upload_id: str,
        scope: str,
        catalog_map: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        sku_key = sku or ""
        if sku_key:
            cache_key = self._product_cache_key(scope, sku_key)
            cached = self._product_meta_cache.get(cache_key)
            if cached:
                return cached
            snapshot = catalog_map.get(sku_key)
            if snapshot:
                product_data = self._convert_snapshot_to_product_data(snapshot)
                self._product_meta_cache[cache_key] = product_data
                return product_data

        if variant_id:
            snapshot = next(
                (snap for snap in catalog_map.values() if getattr(snap, "variant_id", None) == variant_id),
                None,
            )
            if snapshot:
                product_data = self._convert_snapshot_to_product_data(snapshot)
                sku_value = getattr(snapshot, "sku", None)
                if sku_value:
                    self._product_meta_cache[self._product_cache_key(scope, sku_value)] = product_data
                return product_data

        if variant_id:
            product_data = await self.get_product_data_for_variant(variant_id, csv_upload_id)
            if product_data and sku_key:
                self._product_meta_cache[self._product_cache_key(scope, sku_key)] = product_data
            return product_data

        return None

    async def _get_inventory_data(
        self,
        inventory_item_id: Optional[str],
        variant_id: Optional[str],
        csv_upload_id: str,
        run_id: Optional[str],
        scope: str,
        inventory_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if inventory_item_id:
            cached = inventory_map.get(inventory_item_id)
            if cached:
                return cached

        levels_data = None

        if inventory_item_id and run_id:
            levels = await storage.get_inventory_levels_by_item_id_run(inventory_item_id, run_id)
            if levels:
                levels_data = self._convert_inventory_levels(levels)
        elif inventory_item_id:
            levels = await storage.get_inventory_levels_by_item_id(inventory_item_id, csv_upload_id)
            if levels:
                levels_data = self._convert_inventory_levels(levels)

        if levels_data and inventory_item_id:
            inventory_map[inventory_item_id] = levels_data
            self._inventory_map_by_scope.setdefault(scope, inventory_map)[inventory_item_id] = levels_data
            return levels_data

        if variant_id:
            return await self.get_inventory_data_for_variant(variant_id, csv_upload_id)

        return None

    def reset_unresolved_cache(self) -> None:
        """Clear cached unresolved SKUs so new catalog data can be re-evaluated."""
        self.unresolved_sku_cache.clear()

    def reset_resolved_cache(self) -> None:
        """Clear resolved cache, typically when catalog variants are refreshed."""
        self.resolved_variant_cache.clear()

    def _log_unresolved_summary(self, upload_id: str, count: int, samples: List[str]) -> None:
        now = datetime.utcnow()
        if self._last_unresolved_log_at and (now - self._last_unresolved_log_at).total_seconds() < 30:
            return
        self._last_unresolved_log_at = now
        sample_preview = ", ".join(samples) if samples else "(none captured)"
        logger.warning(
            "DataMapper unresolved SKUs remain after enrichment: upload=%s count=%s sample=%s",
            upload_id,
            count,
            sample_preview
        )
    
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
