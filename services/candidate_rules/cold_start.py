"""
Cold-Start Bundle Rules Generator (PR-6)
Generates bundle candidates for products with no purchase history
"""
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass

from services.storage import storage
from services.embeddings.metadata_backfill import MetadataBackfillEngine

logger = logging.getLogger(__name__)

@dataclass
class ColdStartRule:
    """Cold-start bundle generation rule"""
    rule_type: str
    anchor_product: str
    target_products: List[str]
    reasoning: str
    confidence: float
    bundle_type: str
    objective_alignment: str

class ColdStartRulesGenerator:
    """Generates bundle rules for new products without purchase history"""
    
    def __init__(self):
        self.metadata_engine = MetadataBackfillEngine()
        
        # Rule generation strategies
        self.rule_strategies = {
            "category_clustering": {
                "enabled": True,
                "max_bundles": 5,
                "min_similarity": 0.6,
                "bundle_size": [2, 3, 4]
            },
            "price_band_matching": {
                "enabled": True,
                "max_bundles": 3,
                "price_tolerance": 0.3,
                "bundle_size": [2, 3]
            },
            "brand_affinity": {
                "enabled": True,
                "max_bundles": 4,
                "min_brand_similarity": 0.8,
                "bundle_size": [2, 3, 4]
            },
            "complementary_pairing": {
                "enabled": True,
                "max_bundles": 6,
                "cross_category_bonus": 0.2,
                "bundle_size": [2, 3]
            }
        }
        
        # Objective alignment rules
        self.objective_rules = {
            "new_launch": {
                "prefer_new_products": True,
                "max_established_products": 1,
                "bundle_types": ["FBT", "MIX_MATCH"]
            },
            "clear_slow_movers": {
                "include_slow_movers": True,
                "slow_mover_ratio": 0.5,
                "bundle_types": ["VOLUME_DISCOUNT", "MIX_MATCH"]
            },
            "increase_aov": {
                "prefer_higher_price": True,
                "min_bundle_value": 50.0,
                "bundle_types": ["FBT", "FIXED"]
            },
            "category_bundle": {
                "same_category_preference": True,
                "max_categories": 2,
                "bundle_types": ["MIX_MATCH", "FIXED"]
            }
        }
    
    async def generate_cold_start_rules(self, csv_upload_id: str, 
                                      new_products: List[str], 
                                      objective: str = "increase_aov") -> List[ColdStartRule]:
        """Generate bundle rules for new products"""
        try:
            logger.info(f"Generating cold-start rules for {len(new_products)} new products with objective: {objective}")
            
            # Get product similarities
            similarities = await self.metadata_engine.generate_cold_start_similarities(
                csv_upload_id, new_products
            )
            
            # Generate rules using different strategies
            all_rules = []
            
            for strategy_name, strategy_config in self.rule_strategies.items():
                if not strategy_config.get("enabled", True):
                    continue
                
                strategy_rules = await self._apply_strategy(
                    strategy_name, strategy_config, similarities, objective, csv_upload_id
                )
                all_rules.extend(strategy_rules)
            
            # Deduplicate and rank rules
            unique_rules = self._deduplicate_rules(all_rules)
            ranked_rules = self._rank_rules(unique_rules, objective)
            
            logger.info(f"Generated {len(ranked_rules)} cold-start rules")
            return ranked_rules
            
        except Exception as e:
            logger.error(f"Error generating cold-start rules: {e}")
            return []
    
    async def _apply_strategy(self, strategy_name: str, strategy_config: Dict[str, Any], 
                            similarities: Dict[str, List], objective: str, 
                            csv_upload_id: str) -> List[ColdStartRule]:
        """Apply a specific rule generation strategy"""
        try:
            strategy_method = getattr(self, f"_generate_{strategy_name}_rules", None)
            if not strategy_method:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return []
            
            return await strategy_method(strategy_config, similarities, objective, csv_upload_id)
            
        except Exception as e:
            logger.error(f"Error applying strategy {strategy_name}: {e}")
            return []
    
    async def _generate_category_clustering_rules(self, config: Dict[str, Any], 
                                                similarities: Dict[str, List], 
                                                objective: str, csv_upload_id: str) -> List[ColdStartRule]:
        """Generate rules based on category clustering"""
        rules = []
        
        for anchor_product, similar_products in similarities.items():
            # Filter by category similarity
            category_similar = [
                sim for sim in similar_products 
                if sim.similarity_factors.get("category", 0) >= config.get("min_similarity", 0.6)
            ]
            
            if len(category_similar) < 1:
                continue
            
            # Generate bundles of different sizes
            for bundle_size in config.get("bundle_size", [2, 3]):
                if len(category_similar) >= bundle_size - 1:
                    target_products = [sim.product_b for sim in category_similar[:bundle_size-1]]
                    
                    avg_similarity = sum(sim.similarity_score for sim in category_similar[:bundle_size-1]) / (bundle_size - 1)
                    
                    rules.append(ColdStartRule(
                        rule_type="category_clustering",
                        anchor_product=anchor_product,
                        target_products=target_products,
                        reasoning=f"Products clustered by category similarity (avg: {avg_similarity:.2f})",
                        confidence=avg_similarity,
                        bundle_type=self._select_bundle_type(objective, "category"),
                        objective_alignment=objective
                    ))
        
        return rules[:config.get("max_bundles", 5)]
    
    async def _generate_price_band_matching_rules(self, config: Dict[str, Any], 
                                                similarities: Dict[str, List], 
                                                objective: str, csv_upload_id: str) -> List[ColdStartRule]:
        """Generate rules based on price band matching"""
        rules = []
        
        for anchor_product, similar_products in similarities.items():
            # Filter by price similarity
            price_similar = [
                sim for sim in similar_products 
                if sim.similarity_factors.get("price_band", 0) >= 0.5  # Same or adjacent price band
            ]
            
            if len(price_similar) < 1:
                continue
            
            # Generate bundles focused on price alignment
            for bundle_size in config.get("bundle_size", [2, 3]):
                if len(price_similar) >= bundle_size - 1:
                    target_products = [sim.product_b for sim in price_similar[:bundle_size-1]]
                    
                    avg_price_similarity = sum(sim.similarity_factors.get("price_band", 0) for sim in price_similar[:bundle_size-1]) / (bundle_size - 1)
                    
                    rules.append(ColdStartRule(
                        rule_type="price_band_matching",
                        anchor_product=anchor_product,
                        target_products=target_products,
                        reasoning=f"Products matched by price band compatibility (avg: {avg_price_similarity:.2f})",
                        confidence=avg_price_similarity * 0.8,  # Lower confidence for price-only matching
                        bundle_type=self._select_bundle_type(objective, "price"),
                        objective_alignment=objective
                    ))
        
        return rules[:config.get("max_bundles", 3)]
    
    async def _generate_brand_affinity_rules(self, config: Dict[str, Any], 
                                           similarities: Dict[str, List], 
                                           objective: str, csv_upload_id: str) -> List[ColdStartRule]:
        """Generate rules based on brand affinity"""
        rules = []
        
        for anchor_product, similar_products in similarities.items():
            # Filter by brand similarity
            brand_similar = [
                sim for sim in similar_products 
                if sim.similarity_factors.get("brand", 0) >= config.get("min_brand_similarity", 0.8)
            ]
            
            if len(brand_similar) < 1:
                continue
            
            # Generate brand-focused bundles
            for bundle_size in config.get("bundle_size", [2, 3, 4]):
                if len(brand_similar) >= bundle_size - 1:
                    target_products = [sim.product_b for sim in brand_similar[:bundle_size-1]]
                    
                    avg_brand_similarity = sum(sim.similarity_factors.get("brand", 0) for sim in brand_similar[:bundle_size-1]) / (bundle_size - 1)
                    
                    rules.append(ColdStartRule(
                        rule_type="brand_affinity",
                        anchor_product=anchor_product,
                        target_products=target_products,
                        reasoning=f"Products from same/similar brands (avg: {avg_brand_similarity:.2f})",
                        confidence=avg_brand_similarity * 0.9,  # High confidence for brand matching
                        bundle_type=self._select_bundle_type(objective, "brand"),
                        objective_alignment=objective
                    ))
        
        return rules[:config.get("max_bundles", 4)]
    
    async def _generate_complementary_pairing_rules(self, config: Dict[str, Any], 
                                                  similarities: Dict[str, List], 
                                                  objective: str, csv_upload_id: str) -> List[ColdStartRule]:
        """Generate rules for complementary product pairing"""
        rules = []
        
        for anchor_product, similar_products in similarities.items():
            # Look for cross-category opportunities
            cross_category = [
                sim for sim in similar_products 
                if sim.similarity_factors.get("category", 1) < 0.5  # Different categories
                and sim.similarity_score > 0.3  # But overall similarity
            ]
            
            same_category = [
                sim for sim in similar_products 
                if sim.similarity_factors.get("category", 0) > 0.7  # Same categories
            ]
            
            # Mix cross-category and same-category for complementary bundles
            if cross_category and same_category:
                for bundle_size in config.get("bundle_size", [2, 3]):
                    if bundle_size == 2 and cross_category:
                        # Simple cross-category pair
                        target_products = [cross_category[0].product_b]
                        reasoning = "Cross-category complementary pairing"
                        confidence = cross_category[0].similarity_score + config.get("cross_category_bonus", 0.2)
                    elif bundle_size == 3 and cross_category and same_category:
                        # Mixed bundle
                        target_products = [cross_category[0].product_b, same_category[0].product_b]
                        reasoning = "Mixed complementary bundle (cross-category + same-category)"
                        confidence = (cross_category[0].similarity_score + same_category[0].similarity_score) / 2
                    else:
                        continue
                    
                    rules.append(ColdStartRule(
                        rule_type="complementary_pairing",
                        anchor_product=anchor_product,
                        target_products=target_products,
                        reasoning=reasoning,
                        confidence=min(1.0, confidence),
                        bundle_type=self._select_bundle_type(objective, "complementary"),
                        objective_alignment=objective
                    ))
        
        return rules[:config.get("max_bundles", 6)]
    
    def _select_bundle_type(self, objective: str, strategy_hint: str) -> str:
        """Select appropriate bundle type based on objective and strategy"""
        objective_rules = self.objective_rules.get(objective, {})
        preferred_types = objective_rules.get("bundle_types", ["FBT", "MIX_MATCH"])
        
        # Strategy-specific preferences
        if strategy_hint == "category":
            return "MIX_MATCH" if "MIX_MATCH" in preferred_types else preferred_types[0]
        elif strategy_hint == "price":
            return "VOLUME_DISCOUNT" if "VOLUME_DISCOUNT" in preferred_types else preferred_types[0]
        elif strategy_hint == "brand":
            return "FIXED" if "FIXED" in preferred_types else preferred_types[0]
        elif strategy_hint == "complementary":
            return "FBT" if "FBT" in preferred_types else preferred_types[0]
        else:
            return preferred_types[0]
    
    def _deduplicate_rules(self, rules: List[ColdStartRule]) -> List[ColdStartRule]:
        """Remove duplicate rules"""
        seen_combinations = set()
        unique_rules = []
        
        for rule in rules:
            # Create a unique key for the product combination
            products_key = tuple(sorted([rule.anchor_product] + rule.target_products))
            combination_key = (products_key, rule.bundle_type, rule.objective_alignment)
            
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                unique_rules.append(rule)
        
        return unique_rules
    
    def _rank_rules(self, rules: List[ColdStartRule], objective: str) -> List[ColdStartRule]:
        """Rank rules by confidence and objective alignment"""
        # Sort by confidence descending
        rules.sort(key=lambda r: r.confidence, reverse=True)
        
        # Apply objective-specific boosts
        for rule in rules:
            if rule.objective_alignment == objective:
                rule.confidence *= 1.1  # 10% boost for objective alignment
        
        # Re-sort after boosts
        rules.sort(key=lambda r: r.confidence, reverse=True)
        
        return rules