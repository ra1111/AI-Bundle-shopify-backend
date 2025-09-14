"""
Bayesian Discount Shrinkage Service
Dynamic pricing based on historical data with category priors and objective-based caps
"""
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict

from services.storage import storage

logger = logging.getLogger(__name__)

class BayesianPricingEngine:
    """Service for computing Bayesian discount shrinkage with objective-based caps"""
    
    def __init__(self):
        # Objective-based discount caps
        self.objective_caps = {
            "margin_guard": {"min_discount": Decimal('0'), "max_discount": Decimal('10')},
            "clear_slow_movers": {"min_discount": Decimal('5'), "max_discount": Decimal('40')},
            "seasonal_promo": {"min_discount": Decimal('10'), "max_discount": Decimal('30')},
            "new_launch": {"min_discount": Decimal('0'), "max_discount": Decimal('15')},
            "increase_aov": {"min_discount": Decimal('5'), "max_discount": Decimal('20')},
            "category_bundle": {"min_discount": Decimal('10'), "max_discount": Decimal('25')},
            "gift_box": {"min_discount": Decimal('15'), "max_discount": Decimal('35')},
            "subscription_push": {"min_discount": Decimal('5'), "max_discount": Decimal('20')}
        }
        
        # Shrinkage parameters
        self.min_transactions_for_confidence = 20
        self.category_prior_weight = 0.3
        self.global_prior_weight = 0.1
    
    async def compute_bundle_pricing(self, bundle_products: List[str], objective: str, 
                                   csv_upload_id: str, bundle_type: str) -> Dict[str, Any]:
        """Compute bundle pricing with Bayesian discount shrinkage"""
        try:
            # ARCHITECT FIX: Preload catalog_map once for in-memory lookups
            catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            # Check if all SKUs exist in catalog - return success=False if missing
            missing_skus = [sku for sku in bundle_products if sku not in catalog_map]
            if missing_skus:
                logger.warning(f"Missing SKUs in catalog: {missing_skus}")
                return {
                    "success": False,
                    "error": f"Missing catalog data for SKUs: {missing_skus}",
                    "pricing": {"original_total": Decimal('0'), "bundle_price": Decimal('0'), "discount_amount": Decimal('0')}
                }
            
            # Get historical data for all products in bundle
            product_data = await self.get_product_pricing_data(bundle_products, csv_upload_id, catalog_map)
            
            # Compute shrunk discount baselines for each product
            shrunk_discounts = {}
            category_priors = await self.compute_category_priors(csv_upload_id)
            global_prior = await self.compute_global_prior(csv_upload_id)
            
            for product_sku in bundle_products:
                if product_sku in product_data:
                    shrunk_discount = await self.compute_shrunk_discount(
                        product_data[product_sku], category_priors, global_prior
                    )
                    shrunk_discounts[product_sku] = shrunk_discount
                else:
                    # Fallback for products without historical data
                    shrunk_discounts[product_sku] = self.get_default_discount(objective)
            
            # Apply objective-based caps
            capped_discounts = self.apply_objective_caps(shrunk_discounts, objective)
            
            # Compute bundle pricing using preloaded catalog_map
            bundle_pricing = await self.compute_bundle_totals(
                bundle_products, capped_discounts, csv_upload_id, bundle_type, catalog_map
            )
            
            return {
                "success": True,
                "pricing": bundle_pricing,
                "individual_discounts": capped_discounts,
                "shrunk_baselines": shrunk_discounts,
                "objective_caps": self.objective_caps.get(objective, {}),
                "methodology": "bayesian_shrinkage"
            }
            
        except Exception as e:
            logger.error(f"Error computing bundle pricing: {e}")
            return {
                "success": False,
                "error": str(e),
                "pricing": {"original_total": Decimal('0'), "bundle_price": Decimal('0'), "discount_amount": Decimal('0')}
            }
    
    async def get_product_pricing_data(self, product_skus: List[str], csv_upload_id: str, catalog_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get historical pricing data for products using preloaded catalog_map"""
        product_data = {}
        
        for sku in product_skus:
            try:
                # Get historical sales data for this product
                sales_data = await storage.get_variant_sales_data(sku, csv_upload_id, days=180)
                catalog_item = catalog_map.get(sku)  # Use in-memory lookup instead of DB query
                
                if sales_data and catalog_item:
                    # Compute historical discount statistics
                    total_revenue = Decimal('0')
                    total_discount = Decimal('0')
                    transaction_count = 0
                    
                    for sale in sales_data:
                        line_revenue = sale.unit_price * sale.quantity
                        total_revenue += line_revenue
                        
                        # Estimate discount from price difference (if compare_at_price available)
                        if catalog_item.compare_at_price and catalog_item.compare_at_price > sale.unit_price:
                            line_discount = (catalog_item.compare_at_price - sale.unit_price) * sale.quantity
                            total_discount += line_discount
                        
                        transaction_count += 1
                    
                    discount_pct = Decimal('0')
                    if total_revenue > 0:
                        discount_pct = (total_discount / total_revenue) * Decimal('100')
                    
                    product_data[sku] = {
                        "historical_discount_pct": discount_pct,
                        "transaction_count": transaction_count,
                        "category": catalog_item.product_type or "general",
                        "current_price": catalog_item.price,
                        "compare_at_price": catalog_item.compare_at_price,
                        "total_revenue": total_revenue
                    }
                
            except Exception as e:
                logger.warning(f"Error getting pricing data for {sku}: {e}")
                continue
        
        return product_data
    
    async def compute_category_priors(self, csv_upload_id: str) -> Dict[str, Decimal]:
        """Compute category-level discount priors"""
        try:
            # Get all catalog items with sales data
            catalog_items = await storage.get_catalog_snapshots_by_upload(csv_upload_id)
            category_stats = defaultdict(lambda: {"total_discount": Decimal('0'), "total_revenue": Decimal('0'), "count": 0})
            
            for item in catalog_items:
                try:
                    sales_data = await storage.get_variant_sales_data(item.sku, csv_upload_id, days=90)
                    if sales_data:
                        category = item.product_type or "general"
                        
                        for sale in sales_data:
                            line_revenue = sale.unit_price * sale.quantity
                            category_stats[category]["total_revenue"] += line_revenue
                            
                            # Estimate discount
                            if item.compare_at_price and item.compare_at_price > sale.unit_price:
                                line_discount = (item.compare_at_price - sale.unit_price) * sale.quantity
                                category_stats[category]["total_discount"] += line_discount
                            
                            category_stats[category]["count"] += 1
                
                except Exception as e:
                    logger.warning(f"Error processing category data for {item.sku}: {e}")
                    continue
            
            # Compute category priors
            category_priors = {}
            for category, stats in category_stats.items():
                if stats["total_revenue"] > 0 and stats["count"] >= 5:
                    discount_pct = (stats["total_discount"] / stats["total_revenue"]) * Decimal('100')
                    category_priors[category] = discount_pct
                else:
                    category_priors[category] = Decimal('10')  # Default 10% for sparse categories
            
            return category_priors
            
        except Exception as e:
            logger.warning(f"Error computing category priors: {e}")
            return {"general": Decimal('10')}
    
    async def compute_global_prior(self, csv_upload_id: str) -> Decimal:
        """Compute global discount prior across all products"""
        try:
            # Get overall discount statistics
            all_sales = await storage.get_all_sales_data(csv_upload_id, days=90)
            catalog_items_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            total_revenue = Decimal('0')
            total_discount = Decimal('0')
            
            for sale in all_sales:
                line_revenue = sale.unit_price * sale.quantity
                total_revenue += line_revenue
                
                # Get catalog data for discount calculation
                catalog_item = catalog_items_map.get(sale.sku)
                if catalog_item and catalog_item.compare_at_price and catalog_item.compare_at_price > sale.unit_price:
                    line_discount = (catalog_item.compare_at_price - sale.unit_price) * sale.quantity
                    total_discount += line_discount
            
            if total_revenue > 0:
                return (total_discount / total_revenue) * Decimal('100')
            else:
                return Decimal('12')  # Default global prior
                
        except Exception as e:
            logger.warning(f"Error computing global prior: {e}")
            return Decimal('12')
    
    async def compute_shrunk_discount(self, product_data: Dict[str, Any], 
                                    category_priors: Dict[str, Decimal], 
                                    global_prior: Decimal) -> Decimal:
        """Compute Bayesian shrunk discount for a product"""
        try:
            historical_discount = product_data["historical_discount_pct"]
            transaction_count = product_data["transaction_count"]
            category = product_data["category"]
            
            # Get category prior
            category_prior = category_priors.get(category, global_prior)
            
            # Compute shrinkage weight based on transaction count
            confidence_weight = min(1.0, float(transaction_count) / self.min_transactions_for_confidence)
            
            # Bayesian shrinkage formula
            # shrunk_estimate = w * sample_mean + (1-w) * prior
            if transaction_count >= self.min_transactions_for_confidence:
                # High confidence: mostly use historical data
                shrunk_discount = (
                    Decimal(str(confidence_weight)) * historical_discount +
                    Decimal(str(1 - confidence_weight)) * category_prior
                )
            elif transaction_count >= 5:
                # Medium confidence: blend historical, category, and global
                w_historical = Decimal(str(confidence_weight * 0.6))
                w_category = Decimal(str(self.category_prior_weight))
                w_global = Decimal('1') - w_historical - w_category
                
                shrunk_discount = (
                    w_historical * historical_discount +
                    w_category * category_prior +
                    w_global * global_prior
                )
            else:
                # Low confidence: mostly use priors
                w_category = Decimal(str(1 - self.global_prior_weight))
                w_global = Decimal(str(self.global_prior_weight))
                
                shrunk_discount = (
                    w_category * category_prior +
                    w_global * global_prior
                )
            
            return max(Decimal('0'), shrunk_discount)
            
        except Exception as e:
            logger.warning(f"Error computing shrunk discount: {e}")
            return Decimal('10')  # Safe fallback
    
    def apply_objective_caps(self, shrunk_discounts: Dict[str, Decimal], objective: str) -> Dict[str, Decimal]:
        """Apply objective-based caps to shrunk discounts"""
        caps = self.objective_caps.get(objective, {"min_discount": Decimal('0'), "max_discount": Decimal('30')})
        min_cap = caps["min_discount"]
        max_cap = caps["max_discount"]
        
        capped_discounts = {}
        for sku, discount in shrunk_discounts.items():
            capped_discount = max(min_cap, min(max_cap, discount))
            capped_discounts[sku] = capped_discount
        
        return capped_discounts
    
    def get_default_discount(self, objective: str) -> Decimal:
        """Get default discount for products without historical data"""
        caps = self.objective_caps.get(objective, {"min_discount": Decimal('5'), "max_discount": Decimal('20')})
        # Use middle of the range as default
        return (caps["min_discount"] + caps["max_discount"]) / Decimal('2')
    
    async def compute_bundle_totals(self, product_skus: List[str], capped_discounts: Dict[str, Decimal], 
                                  csv_upload_id: str, bundle_type: str, catalog_map: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final bundle pricing totals using preloaded catalog_map"""
        try:
            original_total = Decimal('0')
            discount_amount = Decimal('0')
            product_details = []
            
            for sku in product_skus:
                # Use preloaded catalog_map for in-memory lookup
                catalog_item = catalog_map.get(sku)
                if catalog_item:
                    product_price = catalog_item.price
                    discount_pct = capped_discounts.get(sku, Decimal('10'))
                    
                    product_discount = product_price * (discount_pct / Decimal('100'))
                    discounted_price = product_price - product_discount
                    
                    original_total += product_price
                    discount_amount += product_discount
                    
                    product_details.append({
                        "sku": sku,
                        "name": catalog_item.product_title,
                        "original_price": product_price,
                        "discount_pct": discount_pct,
                        "discount_amount": product_discount,
                        "discounted_price": discounted_price
                    })
                else:
                    # Architect's fix: Fallback to order_lines pricing when catalog missing
                    logger.info(f"Catalog missing for SKU: {sku}, trying order_lines fallback")
                    
                    # Get order lines data for pricing fallback
                    order_lines = await storage.get_order_lines_by_sku(sku, csv_upload_id)
                    if order_lines:
                        # Use average unit_price from order_lines as fallback
                        avg_price = sum(line.unit_price for line in order_lines if line.unit_price) / len(order_lines)
                        product_price = Decimal(str(avg_price))
                        discount_pct = capped_discounts.get(sku, Decimal('10'))
                        
                        product_discount = product_price * (discount_pct / Decimal('100'))
                        discounted_price = product_price - product_discount
                        
                        original_total += product_price
                        discount_amount += product_discount
                        
                        # Use first order line for name fallback
                        product_name = order_lines[0].name if order_lines[0].name else f"Product {sku}"
                        
                        product_details.append({
                            "sku": sku,
                            "name": product_name,
                            "original_price": product_price,
                            "discount_pct": discount_pct,
                            "discount_amount": product_discount,
                            "discounted_price": discounted_price,
                            "fallback_source": "order_lines"
                        })
                        logger.info(f"Used order_lines fallback pricing for SKU: {sku}, price: {product_price}")
                    else:
                        logger.warning(f"No pricing data available for SKU: {sku} in catalog or order_lines")
            
            bundle_price = original_total - discount_amount
            
            # Apply bundle-type specific adjustments
            if bundle_type == "VOLUME_DISCOUNT":
                # Additional volume discount
                additional_discount = original_total * Decimal('0.05')  # 5% extra
                discount_amount += additional_discount
                bundle_price -= additional_discount
            
            elif bundle_type == "BXGY":
                # Buy X Get Y pricing (simplest item free)
                if product_details:
                    cheapest_item = min(product_details, key=lambda x: x["discounted_price"])
                    bxgy_discount = cheapest_item["discounted_price"]
                    discount_amount += bxgy_discount
                    bundle_price -= bxgy_discount
            
            # Ensure reasonable pricing
            bundle_price = max(bundle_price, original_total * Decimal('0.3'))  # Min 30% of original
            discount_amount = original_total - bundle_price
            
            # Round to nice endings (.99, .95, .00)
            bundle_price = self.round_to_nice_ending(bundle_price)
            discount_amount = original_total - bundle_price
            
            total_discount_pct = Decimal('0')
            if original_total > 0:
                total_discount_pct = (discount_amount / original_total) * Decimal('100')
            
            return {
                "original_total": original_total,
                "bundle_price": bundle_price,
                "discount_amount": discount_amount,
                "total_discount_pct": total_discount_pct,
                "product_details": product_details,
                "bundle_type_adjustment": bundle_type
            }
            
        except Exception as e:
            logger.error(f"Error computing bundle totals: {e}")
            return {
                "original_total": Decimal('0'),
                "bundle_price": Decimal('0'),
                "discount_amount": Decimal('0'),
                "total_discount_pct": Decimal('0'),
                "product_details": [],
                "error": str(e)
            }
    
    def round_to_nice_ending(self, price: Decimal) -> Decimal:
        """Round price to nice psychological endings (.99, .95, .00)"""
        try:
            price_int = int(price)
            price_cents = int((price - price_int) * 100)
            
            # Choose nice ending based on price range
            if price < Decimal('10'):
                # Small prices: .99
                return Decimal(f"{price_int}.99")
            elif price < Decimal('50'):
                # Medium prices: .95 or .99
                return Decimal(f"{price_int}.95")
            else:
                # Larger prices: .00 or .99
                if price_cents > 50:
                    return Decimal(str(price_int + 1))  # Round up to .00
                else:
                    return Decimal(f"{price_int}.99")
                    
        except Exception as e:
            logger.warning(f"Error rounding price {price}: {e}")
            return price  # Return original if rounding fails