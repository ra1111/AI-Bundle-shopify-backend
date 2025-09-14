"""
Weighted Linear Ranker Service
Multi-factor scoring beyond confidence × lift with transparency and tunability
"""
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal
import math

from services.storage import storage

logger = logging.getLogger(__name__)

class WeightedLinearRanker:
    """Advanced multi-factor ranking system for bundle recommendations"""
    
    def __init__(self):
        # Default weight configuration (can be tuned per objective)
        self.default_weights = {
            "confidence": Decimal('0.35'),
            "lift": Decimal('0.25'),
            "objective_fit": Decimal('0.20'),
            "inventory_term": Decimal('0.10'),
            "price_sanity": Decimal('0.10'),
            "novelty_penalty": Decimal('0.10')  # Subtracted from total
        }
        
        # Objective-specific weight overrides
        self.objective_weights = {
            "clear_slow_movers": {
                "confidence": Decimal('0.25'),
                "lift": Decimal('0.20'),
                "objective_fit": Decimal('0.35'),  # Higher weight for objective fit
                "inventory_term": Decimal('0.15'),
                "price_sanity": Decimal('0.05'),
                "novelty_penalty": Decimal('0.05')
            },
            "margin_guard": {
                "confidence": Decimal('0.30'),
                "lift": Decimal('0.25'),
                "objective_fit": Decimal('0.15'),
                "inventory_term": Decimal('0.05'),
                "price_sanity": Decimal('0.25'),  # Higher weight for price sanity
                "novelty_penalty": Decimal('0.10')
            },
            "increase_aov": {
                "confidence": Decimal('0.40'),  # Focus on statistical confidence
                "lift": Decimal('0.30'),
                "objective_fit": Decimal('0.15'),
                "inventory_term": Decimal('0.10'),
                "price_sanity": Decimal('0.05'),
                "novelty_penalty": Decimal('0.05')
            }
        }
        
        # Anchor exposure tracking for novelty penalty
        self.anchor_usage_count = {}
    
    async def rank_bundle_recommendations(self, candidates: List[Dict[str, Any]], 
                                        objective: str, csv_upload_id: str) -> List[Dict[str, Any]]:
        """Rank bundle candidates using weighted linear scoring"""
        try:
            if not candidates:
                return []
            
            # Get objective-specific weights
            weights = self.get_weights_for_objective(objective)
            
            # Compute all ranking features
            enriched_candidates = []
            for i, candidate in enumerate(candidates):
                enriched = await self.compute_ranking_features(candidate, objective, csv_upload_id)
                enriched["candidate_index"] = i
                enriched_candidates.append(enriched)
            
            # Normalize features to [0, 1] range
            normalized_candidates = self.normalize_features(enriched_candidates)
            
            # Compute weighted scores
            scored_candidates = []
            for candidate in normalized_candidates:
                score = self.compute_weighted_score(candidate, weights)
                candidate["ranking_score"] = float(score)
                candidate["ranking_components"] = self.get_score_components(candidate, weights)
                scored_candidates.append(candidate)
            
            # Sort by ranking score (descending)
            scored_candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
            
            # Assign rank positions and apply novelty penalty
            final_candidates = self.apply_novelty_penalty_and_rank(scored_candidates)
            
            logger.info(f"Ranked {len(final_candidates)} candidates for objective: {objective}")
            return final_candidates
            
        except Exception as e:
            logger.error(f"Error ranking bundle recommendations: {e}")
            return candidates  # Return original candidates if ranking fails
    
    async def compute_ranking_features(self, candidate: Dict[str, Any], 
                                     objective: str, csv_upload_id: str, catalog_map: Dict = None) -> Dict[str, Any]:
        """Compute all ranking features for a candidate"""
        try:
            # ARCHITECT FIX: Preload catalog_map once per batch instead of per-candidate
            if catalog_map is None:
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            # Start with existing candidate data
            enriched = candidate.copy()
            
            # 1. Statistical features (confidence, lift, support)
            enriched["confidence_raw"] = Decimal(str(candidate.get("confidence", 0)))
            enriched["lift_raw"] = Decimal(str(candidate.get("lift", 1)))
            enriched["support_raw"] = Decimal(str(candidate.get("support", 0)))
            
            # 2. Objective fit score
            enriched["objective_fit_raw"] = await self.compute_objective_fit(
                candidate.get("products", []), objective, csv_upload_id, catalog_map
            )
            
            # 3. Inventory term
            enriched["inventory_term_raw"] = await self.compute_inventory_term(
                candidate.get("products", []), csv_upload_id, catalog_map
            )
            
            # 4. Price sanity check
            enriched["price_sanity_raw"] = await self.compute_price_sanity(
                candidate.get("products", []), candidate.get("pricing", {}), csv_upload_id, catalog_map
            )
            
            # 5. Novelty/diversity features
            enriched["novelty_penalty_raw"] = self.compute_novelty_penalty(candidate)
            
            # 6. Additional context features
            enriched["bundle_size"] = len(candidate.get("products", []))
            enriched["generation_sources"] = candidate.get("generation_sources", [])
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Error computing ranking features: {e}")
            return candidate
    
    async def compute_objective_fit(self, product_skus: List[str], objective: str, csv_upload_id: str, catalog_map: Dict = None) -> Decimal:
        """Compute objective fit score for bundle products"""
        try:
            if not product_skus:
                return Decimal('0')
            
            # ARCHITECT FIX: Use preloaded catalog_map instead of unsafe per-pair queries
            if catalog_map is None:
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            catalog_items = [catalog_map.get(sku) for sku in product_skus if sku in catalog_map]
            if not catalog_items:
                return Decimal('0')
            
            total_score = Decimal('0')
            for item in catalog_items:
                item_score = Decimal('0')
                
                # Get objective flags if available
                flags = {}
                if hasattr(item, 'objective_flags') and item.objective_flags:
                    flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                
                # Score based on objective
                if objective == "clear_slow_movers":
                    if flags.get("is_slow_mover", False):
                        item_score += Decimal('0.8')
                    if item.available_total > 10:
                        item_score += Decimal('0.2')
                
                elif objective == "new_launch":
                    if flags.get("is_new_launch", False):
                        item_score += Decimal('0.8')
                    if item.available_total > 0:
                        item_score += Decimal('0.2')
                
                elif objective == "margin_guard":
                    if flags.get("is_high_margin", False):
                        item_score += Decimal('0.7')
                    avg_discount = Decimal('0')  # avg_discount_pct removed from model
                    if avg_discount < Decimal('10'):
                        item_score += Decimal('0.3')
                    elif avg_discount > Decimal('25'):
                        item_score -= Decimal('0.2')  # Penalty for high historical discounts
                
                elif objective == "increase_aov":
                    # Favor higher-priced items and multi-item bundles
                    if item.price > Decimal('25'):
                        item_score += Decimal('0.4')
                    if len(product_skus) > 2:
                        item_score += Decimal('0.3')
                    if item.available_total > 0:
                        item_score += Decimal('0.3')
                
                elif objective == "seasonal_promo":
                    if flags.get("is_seasonal", False):
                        item_score += Decimal('0.7')
                    if item.available_total > 0:
                        item_score += Decimal('0.3')
                
                else:
                    # Default scoring
                    if item.available_total > 0:
                        item_score += Decimal('0.5')
                    if item.price > Decimal('10'):
                        item_score += Decimal('0.3')
                    if not flags.get("is_gift_card", False):
                        item_score += Decimal('0.2')
                
                total_score += item_score
            
            # Average score across items, normalized to [0, 1]
            avg_score = total_score / Decimal(str(len(catalog_items)))
            return min(Decimal('1'), max(Decimal('0'), avg_score))
            
        except Exception as e:
            logger.warning(f"Error computing objective fit: {e}")
            return Decimal('0')
    
    async def compute_inventory_term(self, product_skus: List[str], csv_upload_id: str, catalog_map: Dict = None) -> Decimal:
        """Compute inventory availability score"""
        try:
            if not product_skus:
                return Decimal('0')
            
            # ARCHITECT FIX: Use preloaded catalog_map instead of unsafe per-pair queries
            if catalog_map is None:
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            catalog_items = [catalog_map.get(sku) for sku in product_skus if sku in catalog_map]
            if not catalog_items:
                return Decimal('0')
            
            total_score = Decimal('0')
            for item in catalog_items:
                if item.available_total > 10:
                    total_score += Decimal('1.0')  # High stock
                elif item.available_total > 0:
                    total_score += Decimal('0.7')  # Some stock
                elif item.available_total == -1:
                    total_score += Decimal('0.8')  # Not tracked (assume available)
                else:
                    total_score += Decimal('0.0')  # Out of stock
            
            # Average across items
            return total_score / Decimal(str(len(catalog_items)))
            
        except Exception as e:
            logger.warning(f"Error computing inventory term: {e}")
            return Decimal('0.5')  # Default middle score
    
    async def compute_price_sanity(self, product_skus: List[str], pricing: Dict[str, Any], csv_upload_id: str, catalog_map: Dict = None) -> Decimal:
        """Compute price sanity score (penalize over-discounting)"""
        try:
            if not product_skus or not pricing:
                return Decimal('0.5')  # Neutral score if no pricing data
            
            # ARCHITECT FIX: Use preloaded catalog_map instead of unsafe per-pair queries
            if catalog_map is None:
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            # Get historical discount data for products
            historical_discounts = {}
            for sku in product_skus:
                # ARCHITECT FIX: Use preloaded catalog_map instead of unsafe per-SKU queries
                try:
                    if catalog_map and sku in catalog_map:
                        catalog_item = catalog_map[sku]
                        if catalog_item and hasattr(catalog_item, 'objective_flags') and catalog_item.objective_flags:
                            flags = catalog_item.objective_flags if isinstance(catalog_item.objective_flags, dict) else {}
                            historical_discounts[sku] = Decimal('10')  # Default since avg_discount_pct removed
                        else:
                            historical_discounts[sku] = Decimal('10')  # Default
                    else:
                        historical_discounts[sku] = Decimal('10')  # Default
                except Exception:
                    historical_discounts[sku] = Decimal('10')
            
            # Check bundle discount against historical baselines
            bundle_discount_pct = Decimal('0')
            if pricing.get("total_discount_pct"):
                bundle_discount_pct = Decimal(str(pricing["total_discount_pct"]))
            
            # Calculate average historical discount
            avg_historical = sum(historical_discounts.values()) / Decimal(str(len(historical_discounts)))
            
            # Score based on discount reasonableness
            discount_ratio = bundle_discount_pct / max(avg_historical, Decimal('1'))
            
            if discount_ratio <= Decimal('1.5'):
                return Decimal('1.0')  # Reasonable discount
            elif discount_ratio <= Decimal('2.0'):
                return Decimal('0.7')  # Moderate over-discount
            elif discount_ratio <= Decimal('3.0'):
                return Decimal('0.4')  # High over-discount
            else:
                return Decimal('0.0')  # Excessive over-discount
                
        except Exception as e:
            logger.warning(f"Error computing price sanity: {e}")
            return Decimal('0.5')
    
    def compute_novelty_penalty(self, candidate: Dict[str, Any]) -> Decimal:
        """Compute novelty penalty to avoid anchor over-exposure"""
        try:
            products = candidate.get("products", [])
            if not products:
                return Decimal('0')
            
            # Use first product as anchor for simplicity
            anchor_product = products[0]
            
            # Track usage count
            if anchor_product not in self.anchor_usage_count:
                self.anchor_usage_count[anchor_product] = 0
            
            usage_count = self.anchor_usage_count[anchor_product]
            
            # Penalty increases with usage count
            if usage_count == 0:
                penalty = Decimal('0')
            elif usage_count <= 2:
                penalty = Decimal('0.1')
            elif usage_count <= 4:
                penalty = Decimal('0.3')
            else:
                penalty = Decimal('0.5')
            
            # Update usage count
            self.anchor_usage_count[anchor_product] += 1
            
            return penalty
            
        except Exception as e:
            logger.warning(f"Error computing novelty penalty: {e}")
            return Decimal('0')
    
    def normalize_features(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize all features to [0, 1] range"""
        try:
            if not candidates:
                return candidates
            
            # Features to normalize
            features_to_normalize = [
                "confidence_raw", "lift_raw", "support_raw", 
                "objective_fit_raw", "inventory_term_raw", "price_sanity_raw"
            ]
            
            # Find min/max for each feature
            feature_ranges = {}
            for feature in features_to_normalize:
                values = [c.get(feature, Decimal('0')) for c in candidates]
                if values:
                    feature_ranges[feature] = {
                        "min": min(values),
                        "max": max(values)
                    }
                else:
                    feature_ranges[feature] = {"min": Decimal('0'), "max": Decimal('1')}
            
            # Normalize each candidate
            normalized = []
            for candidate in candidates:
                normalized_candidate = candidate.copy()
                
                for feature in features_to_normalize:
                    raw_value = candidate.get(feature, Decimal('0'))
                    min_val = feature_ranges[feature]["min"]
                    max_val = feature_ranges[feature]["max"]
                    
                    # Min-max normalization
                    if max_val > min_val:
                        normalized_value = (raw_value - min_val) / (max_val - min_val)
                    else:
                        normalized_value = Decimal('0.5')  # Default if no variance
                    
                    # Store normalized value
                    norm_feature_name = feature.replace("_raw", "_norm")
                    normalized_candidate[norm_feature_name] = normalized_value
                
                normalized.append(normalized_candidate)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Error normalizing features: {e}")
            return candidates
    
    def compute_weighted_score(self, candidate: Dict[str, Any], weights: Dict[str, Decimal]) -> Decimal:
        """Compute final weighted score"""
        try:
            score = Decimal('0')
            
            # Add weighted components
            score += weights["confidence"] * candidate.get("confidence_norm", Decimal('0'))
            score += weights["lift"] * candidate.get("lift_norm", Decimal('0'))
            score += weights["objective_fit"] * candidate.get("objective_fit_norm", Decimal('0'))
            score += weights["inventory_term"] * candidate.get("inventory_term_norm", Decimal('0'))
            score += weights["price_sanity"] * candidate.get("price_sanity_norm", Decimal('0'))
            
            # Subtract novelty penalty
            score -= weights["novelty_penalty"] * candidate.get("novelty_penalty_raw", Decimal('0'))
            
            # Ensure score is in [0, 1] range
            return max(Decimal('0'), min(Decimal('1'), score))
            
        except Exception as e:
            logger.warning(f"Error computing weighted score: {e}")
            return Decimal('0.5')
    
    def get_score_components(self, candidate: Dict[str, Any], weights: Dict[str, Decimal]) -> Dict[str, float]:
        """Get detailed score component breakdown for explainability"""
        try:
            components = {}
            
            # Individual weighted contributions
            components["confidence_contribution"] = float(
                weights["confidence"] * candidate.get("confidence_norm", Decimal('0'))
            )
            components["lift_contribution"] = float(
                weights["lift"] * candidate.get("lift_norm", Decimal('0'))
            )
            components["objective_fit_contribution"] = float(
                weights["objective_fit"] * candidate.get("objective_fit_norm", Decimal('0'))
            )
            components["inventory_contribution"] = float(
                weights["inventory_term"] * candidate.get("inventory_term_norm", Decimal('0'))
            )
            components["price_sanity_contribution"] = float(
                weights["price_sanity"] * candidate.get("price_sanity_norm", Decimal('0'))
            )
            components["novelty_penalty"] = float(
                weights["novelty_penalty"] * candidate.get("novelty_penalty_raw", Decimal('0'))
            )
            
            # Raw values for reference
            components["raw_values"] = {
                "confidence": float(candidate.get("confidence_raw", Decimal('0'))),
                "lift": float(candidate.get("lift_raw", Decimal('0'))),
                "objective_fit": float(candidate.get("objective_fit_raw", Decimal('0'))),
                "inventory_term": float(candidate.get("inventory_term_raw", Decimal('0'))),
                "price_sanity": float(candidate.get("price_sanity_raw", Decimal('0')))
            }
            
            return components
            
        except Exception as e:
            logger.warning(f"Error getting score components: {e}")
            return {}
    
    def apply_novelty_penalty_and_rank(self, scored_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply final novelty adjustments and assign rank positions"""
        try:
            # Final sort by adjusted ranking score
            scored_candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
            
            # Assign rank positions
            for i, candidate in enumerate(scored_candidates):
                candidate["rank_position"] = i + 1
            
            return scored_candidates
            
        except Exception as e:
            logger.warning(f"Error applying novelty penalty and ranking: {e}")
            return scored_candidates
    
    def get_weights_for_objective(self, objective: str) -> Dict[str, Decimal]:
        """Get weights for specific objective or default weights"""
        if objective in self.objective_weights:
            return self.objective_weights[objective].copy()
        else:
            return self.default_weights.copy()
    
    def reset_anchor_usage(self):
        """Reset anchor usage tracking (call between different bundle generation runs)"""
        self.anchor_usage_count.clear()
        
    def get_ranking_explanation(self, candidate: Dict[str, Any]) -> str:
        """Generate human-readable ranking explanation"""
        try:
            components = candidate.get("ranking_components", {})
            score = candidate.get("ranking_score", 0)
            
            explanation_parts = []
            explanation_parts.append(f"Overall score: {score:.3f}")
            
            # Identify top contributing factors
            contributions = {
                "Statistical confidence": components.get("confidence_contribution", 0),
                "Lift factor": components.get("lift_contribution", 0),
                "Objective alignment": components.get("objective_fit_contribution", 0),
                "Inventory availability": components.get("inventory_contribution", 0),
                "Price reasonableness": components.get("price_sanity_contribution", 0)
            }
            
            # Sort by contribution
            sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            
            # Add top 2 factors
            for factor, contrib in sorted_contributions[:2]:
                if contrib > 0.01:  # Only mention significant contributions
                    explanation_parts.append(f"{factor}: {contrib:.3f}")
            
            # Mention penalties if significant
            penalty = components.get("novelty_penalty", 0)
            if penalty > 0.05:
                explanation_parts.append(f"Novelty penalty: -{penalty:.3f}")
            
            return " | ".join(explanation_parts)
            
        except Exception as e:
            logger.warning(f"Error generating ranking explanation: {e}")
            return f"Score: {candidate.get('ranking_score', 0):.3f}"