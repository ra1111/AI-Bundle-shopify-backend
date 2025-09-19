"""
Objective Scoring Service
Automated tagging for slow movers, new launches, seasonal items, high-margin products
"""
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal
from datetime import datetime, timedelta
import re

from services.storage import storage

logger = logging.getLogger(__name__)

class ObjectiveScorer:
    """Service for computing objective-relevant signals and flags"""
    
    def __init__(self):
        # Configurable thresholds for objective detection
        self.thresholds = {
            "slow_mover_velocity": Decimal('0.1'),  # < 0.1 units/day
            "slow_mover_stock": 10,  # Must have > 10 units in stock
            "new_launch_days": 30,  # Products created within 30 days
            "high_margin_discount_threshold": Decimal('0.05'),  # < 5% historical discount
            "seasonal_keywords": ["holiday", "christmas", "summer", "winter", "valentine", "halloween", "black friday"]
        }
    
    async def compute_objective_flags(self, csv_upload_id: str) -> Dict[str, Any]:
        """Compute objective flags for all catalog items"""
        metrics = {
            "total_items": 0,
            "slow_movers": 0,
            "new_launches": 0,
            "seasonal_items": 0,
            "high_margin_items": 0,
            "processing_errors": 0
        }
        
        try:
            # Get all catalog snapshots for this run (preferred), fallback to single upload
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                catalog_items = await storage.get_catalog_snapshots_by_run(run_id)
            else:
                catalog_items = await storage.get_catalog_snapshots_by_upload(csv_upload_id)
            metrics["total_items"] = len(catalog_items)
            
            updated_items = []
            
            for item in catalog_items:
                try:
                    flags = await self.compute_flags_for_item(item, csv_upload_id)
                    
                    # Update metrics
                    if flags.get("is_slow_mover"):
                        metrics["slow_movers"] += 1
                    if flags.get("is_new_launch"):
                        metrics["new_launches"] += 1
                    if flags.get("is_seasonal"):
                        metrics["seasonal_items"] += 1
                    if flags.get("is_high_margin"):
                        metrics["high_margin_items"] += 1
                    
                    # Prepare update data
                    updated_item = {
                        "variant_id": item.variant_id,
                        "csv_upload_id": csv_upload_id,
                        "objective_flags": flags,
                        "velocity": flags.get("velocity", Decimal('0')),
                        # avg_discount_pct removed from model
                    }
                    
                    updated_items.append(updated_item)
                    
                except Exception as e:
                    logger.warning(f"Error processing item {item.variant_id}: {e}")
                    metrics["processing_errors"] += 1
                    continue
            
            # Bulk update catalog snapshots with flags
            if updated_items:
                await storage.update_catalog_snapshots_with_flags(updated_items)
                logger.info(f"Updated {len(updated_items)} catalog items with objective flags")
            
            return {
                "metrics": metrics,
                "updated_items": len(updated_items)
            }
            
        except Exception as e:
            logger.error(f"Error computing objective flags: {e}")
            raise
    
    async def compute_flags_for_item(self, catalog_item, csv_upload_id: str) -> Dict[str, Any]:
        """Compute objective flags for a single catalog item"""
        flags = {
            "is_slow_mover": False,
            "is_new_launch": False,
            "is_seasonal": False,
            "is_high_margin": False,
            "velocity": 0.0,
            # avg_discount_pct removed from model
        }
        
        try:
            # 1. Compute velocity and slow mover flag
            velocity = await self.compute_velocity(catalog_item.variant_id, csv_upload_id)
            flags["velocity"] = float(velocity)
            
            if (velocity < self.thresholds["slow_mover_velocity"] and 
                catalog_item.available_total and catalog_item.available_total > self.thresholds["slow_mover_stock"]):
                flags["is_slow_mover"] = True
            
            # 2. Check new launch flag
            if catalog_item.product_created_at:
                days_since_creation = (datetime.now() - catalog_item.product_created_at).days
                if days_since_creation <= self.thresholds["new_launch_days"]:
                    flags["is_new_launch"] = True
            
            # 3. Check seasonal flag
            if self.is_seasonal_product(catalog_item):
                flags["is_seasonal"] = True
            
            # 4. Compute historical discount and high margin flag
            avg_discount = await self.compute_historical_discount(catalog_item.variant_id, csv_upload_id)
            # avg_discount_pct removed from model
            
            if avg_discount < self.thresholds["high_margin_discount_threshold"]:
                flags["is_high_margin"] = True
            
            return flags
            
        except Exception as e:
            logger.warning(f"Error computing flags for {catalog_item.variant_id}: {e}")
            return flags
    
    async def compute_velocity(self, variant_id: str, csv_upload_id: str) -> Decimal:
        """Compute sales velocity (units sold per day) for variant"""
        try:
            # Get order lines for this variant in last 60 days (run-scoped if available)
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                sales_data = await storage.get_variant_sales_data_run(variant_id, run_id, days=60)
            else:
                sales_data = await storage.get_variant_sales_data(variant_id, csv_upload_id, days=60)
            if not sales_data:
                return Decimal('0')
            
            total_quantity = sum(sale.quantity for sale in sales_data if sale.quantity is not None)
            return Decimal(str(total_quantity)) / Decimal('60')
            
        except Exception as e:
            logger.warning(f"Error computing velocity for {variant_id}: {e}")
            return Decimal('0')
    
    def is_seasonal_product(self, catalog_item) -> bool:
        """Check if product is seasonal based on tags and title"""
        text_to_check = f"{catalog_item.product_title} {catalog_item.tags}".lower()
        
        for keyword in self.thresholds["seasonal_keywords"]:
            if keyword in text_to_check:
                return True
        
        return False
    
    async def compute_historical_discount(self, variant_id: str, csv_upload_id: str) -> Decimal:
        """Compute average historical discount percentage for variant"""
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                sales_data = await storage.get_variant_sales_data_run(variant_id, run_id, days=180)
            else:
                sales_data = await storage.get_variant_sales_data(variant_id, csv_upload_id, days=180)
            if not sales_data:
                return Decimal('0')
            
            total_discount = Decimal('0')
            total_sales = Decimal('0')
            
            for sale in sales_data:
                if sale.unit_price and sale.unit_price > 0 and sale.quantity:
                    # Calculate discount percentage from line data
                    # Assuming original_price is available or using current catalog price
                    line_total = sale.unit_price * sale.quantity
                    total_sales += line_total
                    
                    # If discount information is available in order line
                    if hasattr(sale, 'discount_amount') and getattr(sale, 'discount_amount', None):
                        total_discount += getattr(sale, 'discount_amount', Decimal('0'))
            
            if total_sales > 0:
                return (total_discount / total_sales) * Decimal('100')
            
            return Decimal('0')
            
        except Exception as e:
            logger.warning(f"Error computing historical discount for {variant_id}: {e}")
            return Decimal('0')
    
    async def compute_objective_fit_score(self, bundle_products: List[str], objective: str, csv_upload_id: str) -> Decimal:
        """Compute objective fit score for a bundle"""
        try:
            if not bundle_products:
                return Decimal('0')
            
            # Get catalog data for all products in bundle
            catalog_items = await storage.get_catalog_snapshots_by_variants(bundle_products, csv_upload_id)
            if not catalog_items:
                return Decimal('0')
            
            total_score = Decimal('0')
            item_count = len(catalog_items)
            
            for item in catalog_items:
                item_score = Decimal('0')
                
                # Score based on objective
                if objective == "increase_aov":
                    # Higher scores for multi-item bundles with good price points
                    if item_count > 1:
                        item_score += Decimal('0.3')
                    if item.price and item.price > Decimal('20'):
                        item_score += Decimal('0.2')
                
                elif objective == "clear_slow_movers":
                    # Higher scores for slow-moving items with stock
                    if hasattr(item, 'objective_flags') and item.objective_flags:
                        flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                        if flags.get("is_slow_mover", False):
                            item_score += Decimal('0.8')
                        if item.available_total and item.available_total > 10:
                            item_score += Decimal('0.2')
                
                elif objective == "new_launch":
                    # Higher scores for new products
                    if hasattr(item, 'objective_flags') and item.objective_flags:
                        flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                        if flags.get("is_new_launch", False):
                            item_score += Decimal('0.8')
                        if item.available_total and item.available_total > 0:
                            item_score += Decimal('0.2')
                
                elif objective == "seasonal_promo":
                    # Higher scores for seasonal items
                    if hasattr(item, 'objective_flags') and item.objective_flags:
                        flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                        if flags.get("is_seasonal", False):
                            item_score += Decimal('0.7')
                        if item.available_total and item.available_total > 0:
                            item_score += Decimal('0.3')
                
                elif objective == "margin_guard":
                    # Higher scores for high-margin items, penalize over-discounting
                    if hasattr(item, 'objective_flags') and item.objective_flags:
                        flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                        if flags.get("is_high_margin", False):
                            item_score += Decimal('0.6')
                        discount_pct = Decimal('0')  # avg_discount_pct removed from model
                        if discount_pct < Decimal('10'):
                            item_score += Decimal('0.4')
                        elif discount_pct > Decimal('20'):
                            item_score -= Decimal('0.3')  # Penalty for high discount
                
                else:
                    # Default scoring for general objectives
                    if item.available_total and item.available_total > 0:
                        item_score += Decimal('0.5')
                    if item.price and item.price > Decimal('10'):
                        item_score += Decimal('0.3')
                
                total_score += item_score
            
            # Average score across all items in bundle
            average_score = total_score / Decimal(str(item_count))
            
            # Normalize to [0, 1] range
            return min(Decimal('1.0'), max(Decimal('0.0'), average_score))
            
        except Exception as e:
            logger.warning(f"Error computing objective fit score: {e}")
            return Decimal('0')
