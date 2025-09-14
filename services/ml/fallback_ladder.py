"""
Fallback Ladder for Small Shop Recommendations

Implements a 7-tier cascade system to ensure every shop gets useful bundle recommendations,
even when traditional association rules fail due to low transaction volumes.

Tiers:
1. Association rules (strict) - traditional min_support/confidence
2. Adaptive relax - scale thresholds to shop size
3. Smoothed co-occurrence - Laplace smoothing P(B|A) = (count(A∩B)+α)/(count(A)+2α)
4. Item-item similarity - cosine/Jaccard over basket vectors
5. Heuristics - category complements, price-band add-ons
6. Popularity/trending - top sellers, curated bundles
7. Cold-start content - attribute-based recommendations
"""

import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TransactionStats:
    """Precomputed statistics for efficient fallback generation"""
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    item_counts: Dict[str, int] = field(default_factory=dict)  # c(A)
    pair_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)  # c(A∩B)
    total_orders: int = 0
    recent_orders: List[Dict[str, Any]] = field(default_factory=list)  # last N days
    products: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    async def build(cls, storage, csv_upload_id: Optional[str] = None, recent_days: int = 30):
        """Build transaction statistics from storage"""
        stats = cls()
        
        # Get orders and order lines
        orders = await storage.get_orders(csv_upload_id=csv_upload_id)
        order_lines = await storage.get_order_lines(csv_upload_id=csv_upload_id)
        products = await storage.get_products()  # Storage service doesn't support csv_upload_id filtering for products
        
        # Convert SQLAlchemy objects to dictionaries for easier access
        orders_dict = []
        for order in orders:
            orders_dict.append({
                'order_id': order.order_id,
                'created_at': order.created_at,
                'total': order.total,
                'customer_id': order.customer_id
            })
        
        order_lines_dict = []
        for line in order_lines:
            order_lines_dict.append({
                'order_id': line.order_id,
                'sku': line.sku,
                'quantity': line.quantity,
                'price': line.price
            })
        
        products_dict = []
        for product in products:
            products_dict.append({
                'sku': product.sku,
                'name': product.name,
                'brand': product.brand,
                'category': product.category,
                'subcategory': product.subcategory,
                'color': product.color,
                'material': product.material,
                'price': product.price,
                'cost': product.cost,
                'weight_kg': product.weight_kg,
                'tags': product.tags
            })
        
        stats.products = products_dict
        stats.total_orders = len(orders_dict)
        
        # Group order lines by order
        orders_by_id = {order['order_id']: order for order in orders_dict}
        transactions = defaultdict(list)
        
        for line in order_lines_dict:
            order_id = line['order_id']
            if order_id in orders_by_id:
                transactions[order_id].append(line['sku'])
        
        # Build transaction list
        recent_cutoff = datetime.now() - timedelta(days=recent_days)
        
        for order_id, skus in transactions.items():
            order = orders_by_id[order_id]
            transaction = {
                'order_id': order_id,
                'skus': list(set(skus)),  # Remove duplicates
                'created_at': order.get('created_at'),
                'total': order.get('total', 0)
            }
            stats.transactions.append(transaction)
            
            # Check if recent
            if order.get('created_at') and order['created_at'] >= recent_cutoff:
                stats.recent_orders.append(transaction)
        
        # Compute item counts
        for transaction in stats.transactions:
            for sku in transaction['skus']:
                stats.item_counts[sku] = stats.item_counts.get(sku, 0) + 1
        
        # Compute pair counts (limit to items with minimum frequency to bound complexity)
        min_item_count = max(2, math.ceil(0.005 * stats.total_orders))  # 0.5% threshold
        frequent_items = [sku for sku, count in stats.item_counts.items() if count >= min_item_count]
        
        for transaction in stats.transactions:
            transaction_skus = [sku for sku in transaction['skus'] if sku in frequent_items]
            # Generate pairs
            for i, sku_a in enumerate(transaction_skus):
                for sku_b in transaction_skus[i+1:]:
                    if sku_a and sku_b and sku_a != sku_b:  # Ensure valid, different SKUs
                        pair = (min(sku_a, sku_b), max(sku_a, sku_b))  # Create ordered pair
                        stats.pair_counts[pair] = stats.pair_counts.get(pair, 0) + 1
        
        logger.info(f"Built TransactionStats: {stats.total_orders} orders, {len(stats.item_counts)} items, {len(stats.pair_counts)} pairs")
        return stats


@dataclass
class FallbackCandidate:
    """A bundle candidate with source tier and features"""
    bundle_type: str
    products: List[str]
    pricing: Dict[str, Any]
    features: Dict[str, float] = field(default_factory=dict)
    source_tier: str = ""
    explanation: str = ""


class FallbackLadder:
    """7-tier fallback system for bundle recommendations"""
    
    def __init__(self, storage):
        self.storage = storage
        self.stats: Optional[TransactionStats] = None
    
    async def generate_candidates(
        self,
        csv_upload_id: Optional[str] = None,
        objective: str = "revenue",
        bundle_type: str = "FBT",
        target_n: int = 10
    ) -> List[FallbackCandidate]:
        """Generate candidates using fallback ladder until target_n reached"""
        
        # Build statistics once
        self.stats = await TransactionStats.build(self.storage, csv_upload_id)
        
        candidates = []
        tier_metrics = {}
        
        # Tier 1: Association rules (strict)
        tier_candidates = await self._tier1_association_rules(csv_upload_id, bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier1_strict_rules'] = len(tier_candidates)
        logger.info(f"Tier 1 (strict rules): {len(tier_candidates)} candidates")
        
        if len(candidates) >= target_n:
            return candidates[:target_n]
        
        # Tier 2: Adaptive relax
        tier_candidates = await self._tier2_adaptive_relax(csv_upload_id, bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier2_adaptive'] = len(tier_candidates)
        logger.info(f"Tier 2 (adaptive): {len(tier_candidates)} candidates")
        
        if len(candidates) >= target_n:
            return candidates[:target_n]
        
        # Tier 3: Smoothed co-occurrence
        tier_candidates = self._tier3_smoothed_cooccurrence(bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier3_smoothed'] = len(tier_candidates)
        logger.info(f"Tier 3 (smoothed): {len(tier_candidates)} candidates")
        
        if len(candidates) >= target_n:
            return candidates[:target_n]
        
        # Tier 4: Item-item similarity
        tier_candidates = self._tier4_item_similarity(bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier4_similarity'] = len(tier_candidates)
        logger.info(f"Tier 4 (similarity): {len(tier_candidates)} candidates")
        
        if len(candidates) >= target_n:
            return candidates[:target_n]
        
        # Tier 5: Heuristics
        tier_candidates = self._tier5_heuristics(bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier5_heuristics'] = len(tier_candidates)
        logger.info(f"Tier 5 (heuristics): {len(tier_candidates)} candidates")
        
        if len(candidates) >= target_n:
            return candidates[:target_n]
        
        # Tier 6: Popularity/trending
        tier_candidates = self._tier6_popularity(bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier6_popularity'] = len(tier_candidates)
        logger.info(f"Tier 6 (popularity): {len(tier_candidates)} candidates")
        
        if len(candidates) >= target_n:
            return candidates[:target_n]
        
        # Tier 7: Cold-start content
        tier_candidates = self._tier7_cold_start(bundle_type)
        candidates.extend(tier_candidates)
        tier_metrics['tier7_cold_start'] = len(tier_candidates)
        logger.info(f"Tier 7 (cold-start): {len(tier_candidates)} candidates")
        
        logger.info(f"FallbackLadder generated {len(candidates)} total candidates: {tier_metrics}")
        return candidates
    
    async def _tier1_association_rules(self, csv_upload_id: Optional[str], bundle_type: str) -> List[FallbackCandidate]:
        """Tier 1: Use existing association rules with strict thresholds"""
        candidates = []
        
        try:
            # Get existing association rules
            rules = await self.storage.get_association_rules(csv_upload_id=csv_upload_id, limit=20)
            
            for rule in rules:
                antecedent = rule.get('antecedent', [])
                consequent = rule.get('consequent', [])
                
                if len(antecedent) > 0 and len(consequent) > 0:
                    candidate = FallbackCandidate(
                        bundle_type=bundle_type,
                        products=antecedent + consequent,
                        pricing={"discount_percent": 10},  # Default discount
                        features={
                            "support": rule.get('support', 0),
                            "confidence": rule.get('confidence', 0),
                            "lift": rule.get('lift', 0),
                            "tier_weight": 1.0  # Highest priority
                        },
                        source_tier="strict_rules",
                        explanation=f"Strong association: {antecedent} → {consequent}"
                    )
                    candidates.append(candidate)
        
        except Exception as e:
            logger.warning(f"Tier 1 association rules failed: {e}")
        
        return candidates
    
    async def _tier2_adaptive_relax(self, csv_upload_id: Optional[str], bundle_type: str) -> List[FallbackCandidate]:
        """Tier 2: Adaptive threshold relaxation based on shop size"""
        candidates = []
        
        if not self.stats:
            return candidates
        
        # Calculate adaptive thresholds
        min_support_orders = max(3, math.ceil(0.015 * self.stats.total_orders))  # 1.5% of orders
        min_confidence = 0.2  # Reduced from typical 0.3
        
        logger.info(f"Adaptive thresholds: min_support={min_support_orders}, min_confidence={min_confidence}")
        
        # Generate relaxed rules
        for item_a, count_a in self.stats.item_counts.items():
            if count_a < min_support_orders:
                continue
                
            for item_b, count_b in self.stats.item_counts.items():
                if item_a >= item_b:  # Avoid duplicates
                    continue
                
                pair = (min(item_a, item_b), max(item_a, item_b))  # Create ordered pair
                pair_count = self.stats.pair_counts.get(pair, 0)
                
                if pair_count >= min_support_orders:
                    confidence_a_to_b = pair_count / count_a
                    confidence_b_to_a = pair_count / count_b
                    
                    if max(confidence_a_to_b, confidence_b_to_a) >= min_confidence:
                        support = pair_count / self.stats.total_orders
                        lift = pair_count * self.stats.total_orders / (count_a * count_b)
                        
                        candidate = FallbackCandidate(
                            bundle_type=bundle_type,
                            products=[item_a, item_b],
                            pricing={"discount_percent": 8},
                            features={
                                "support": support,
                                "confidence": max(confidence_a_to_b, confidence_b_to_a),
                                "lift": lift,
                                "tier_weight": 0.9
                            },
                            source_tier="adaptive_relax",
                            explanation=f"Relaxed association: {item_a} ↔ {item_b}"
                        )
                        candidates.append(candidate)
        
        return candidates[:10]  # Limit to top 10
    
    def _tier3_smoothed_cooccurrence(self, bundle_type: str) -> List[FallbackCandidate]:
        """Tier 3: Smoothed co-occurrence with Laplace smoothing"""
        candidates = []
        
        if not self.stats:
            return candidates
        
        alpha = 1.0  # Laplace smoothing parameter
        min_smoothed_prob = 0.1
        
        for item_a, count_a in self.stats.item_counts.items():
            if count_a < 2:  # Skip very rare items
                continue
            
            recommendations = []
            for item_b, count_b in self.stats.item_counts.items():
                if item_a == item_b:
                    continue
                
                pair = (min(item_a, item_b), max(item_a, item_b))  # Create ordered pair
                pair_count = self.stats.pair_counts.get(pair, 0)
                
                # Smoothed probability P(B|A) = (count(A∩B) + α) / (count(A) + 2α)
                smoothed_prob = (pair_count + alpha) / (count_a + 2 * alpha)
                
                if smoothed_prob >= min_smoothed_prob:
                    recommendations.append((item_b, smoothed_prob))
            
            # Sort by smoothed probability and take top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for item_b, prob in recommendations[:3]:  # Top 3 per item
                candidate = FallbackCandidate(
                    bundle_type=bundle_type,
                    products=[item_a, item_b],
                    pricing={"discount_percent": 5},
                    features={
                        "smoothed_p": prob,
                        "tier_weight": 0.8
                    },
                    source_tier="smoothed_cooccurrence",
                    explanation=f"Smoothed co-occurrence: {item_a} → {item_b}"
                )
                candidates.append(candidate)
        
        return candidates[:10]
    
    def _tier4_item_similarity(self, bundle_type: str) -> List[FallbackCandidate]:
        """Tier 4: Item-item similarity using cosine/Jaccard"""
        candidates = []
        
        if not self.stats:
            return candidates
        
        for item_a, count_a in self.stats.item_counts.items():
            if count_a < 2:
                continue
            
            similarities = []
            for item_b, count_b in self.stats.item_counts.items():
                if item_a >= item_b:  # Avoid duplicates
                    continue
                
                pair = (min(item_a, item_b), max(item_a, item_b))  # Create ordered pair
                pair_count = self.stats.pair_counts.get(pair, 0)
                
                if pair_count > 0:
                    # Cosine similarity
                    cosine_sim = pair_count / math.sqrt(count_a * count_b)
                    
                    # Jaccard similarity
                    jaccard_sim = pair_count / (count_a + count_b - pair_count)
                    
                    # Combined similarity score
                    combined_sim = (cosine_sim + jaccard_sim) / 2
                    
                    if combined_sim > 0.1:  # Threshold for similarity
                        similarities.append((item_b, cosine_sim, jaccard_sim, combined_sim))
            
            # Sort by combined similarity
            similarities.sort(key=lambda x: x[3], reverse=True)
            
            for item_b, cosine_sim, jaccard_sim, combined_sim in similarities[:2]:  # Top 2 per item
                candidate = FallbackCandidate(
                    bundle_type=bundle_type,
                    products=[item_a, item_b],
                    pricing={"discount_percent": 5},
                    features={
                        "sim_cosine": cosine_sim,
                        "sim_jaccard": jaccard_sim,
                        "sim_combined": combined_sim,
                        "tier_weight": 0.7
                    },
                    source_tier="item_similarity",
                    explanation=f"Similar purchase patterns: {item_a} ↔ {item_b}"
                )
                candidates.append(candidate)
        
        return candidates[:10]
    
    def _tier5_heuristics(self, bundle_type: str) -> List[FallbackCandidate]:
        """Tier 5: Heuristic category complements and price-band recommendations"""
        candidates = []
        
        if not self.stats or not self.stats.products:
            return candidates
        
        # Performance guard: Limit products to prevent infinite loops
        max_products = min(50, len(self.stats.products))
        products_subset = self.stats.products[:max_products]
        
        # Pre-index products by category to avoid O(N^2) scans
        products_by_category = defaultdict(list)
        for product in products_subset:
            category = (product.get('category') or '').lower()
            title = (product.get('title') or '').lower()
            product['_search_text'] = category + ' ' + title
            products_by_category['all'].append(product)
        
        # Category complement rules (simple heuristics)
        complement_rules = {
            'dress': ['belt', 'shoes', 'bag'],
            'shirt': ['pants', 'tie', 'jacket'],
            'board': ['bindings', 'boots'],
            'camera': ['lens', 'memory', 'battery'],
            'phone': ['case', 'charger', 'screen']
        }
        
        logger.info(f"Tier 5: Processing {len(products_subset)} products")
        
        for i, product in enumerate(products_subset):
            # Performance guard: Early exit if we have enough candidates
            if len(candidates) >= 8:
                break
                
            # Performance guard: Limit iterations per product
            if i > 20:  # Process max 20 products 
                break
                
            sku = product.get('sku')
            search_text = product.get('_search_text', '')
            price = product.get('price', 0)
            
            # Find complements by category
            for base_cat, complement_cats in complement_rules.items():
                if base_cat in search_text:
                    for comp_cat in complement_cats:
                        # Find products in complement category (LIMITED SEARCH)
                        matches = 0
                        for comp_product in products_by_category['all'][:10]:  # Limit to first 10 products
                            if matches >= 2:  # Max 2 matches per complement category
                                break
                                
                            comp_sku = comp_product.get('sku')
                            comp_search_text = comp_product.get('_search_text', '')
                            comp_price = comp_product.get('price', 0)
                            
                            if comp_sku and sku and comp_sku != sku and comp_cat in comp_search_text:
                                # Price compatibility (within 3x range)
                                if price > 0 and comp_price > 0:
                                    price_ratio = max(price, comp_price) / min(price, comp_price)
                                    if price_ratio <= 3.0:
                                        candidate = FallbackCandidate(
                                            bundle_type="MIX_MATCH",
                                            products=[sku, comp_sku],
                                            pricing={"discount_percent": 10},
                                            features={
                                                "heuristic_match": 1.0,
                                                "price_ratio": price_ratio,
                                                "tier_weight": 0.6
                                            },
                                            source_tier="heuristics",
                                            explanation=f"Category complement: {base_cat} + {comp_cat}"
                                        )
                                        candidates.append(candidate)
                                        matches += 1
        
        logger.info(f"Tier 5: Generated {len(candidates)} heuristic candidates")
        return candidates[:8]
    
    def _tier6_popularity(self, bundle_type: str) -> List[FallbackCandidate]:
        """Tier 6: Popularity and trending recommendations"""
        candidates = []
        
        if not self.stats:
            return candidates
        
        # Sort items by frequency (popularity)
        popular_items = sorted(self.stats.item_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Recent popularity (last 30 days)
        recent_counts = Counter()
        for transaction in self.stats.recent_orders:
            for sku in transaction['skus']:
                recent_counts[sku] += 1
        
        recent_items = sorted(recent_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create bundles with top sellers
        if len(popular_items) >= 2:
            # Top 2 overall
            top_items = [item[0] for item in popular_items[:3]]
            for i in range(len(top_items)):
                for j in range(i + 1, len(top_items)):
                    candidate = FallbackCandidate(
                        bundle_type=bundle_type,
                        products=[top_items[i], top_items[j]],
                        pricing={"discount_percent": 15},  # Higher discount for popular items
                        features={
                            "popularity_score": (popular_items[i][1] + popular_items[j][1]) / 2,
                            "tier_weight": 0.5
                        },
                        source_tier="popularity",
                        explanation=f"Best sellers bundle"
                    )
                    candidates.append(candidate)
        
        # Recent trending pairs
        if len(recent_items) >= 2:
            for i in range(min(2, len(recent_items))):
                for j in range(i + 1, min(3, len(recent_items))):
                    candidate = FallbackCandidate(
                        bundle_type=bundle_type,
                        products=[recent_items[i][0], recent_items[j][0]],
                        pricing={"discount_percent": 12},
                        features={
                            "trending_score": (recent_items[i][1] + recent_items[j][1]) / 2,
                            "tier_weight": 0.5
                        },
                        source_tier="trending",
                        explanation=f"Trending items bundle"
                    )
                    candidates.append(candidate)
        
        return candidates[:6]
    
    def _tier7_cold_start(self, bundle_type: str) -> List[FallbackCandidate]:
        """Tier 7: Cold-start attribute-based recommendations"""
        candidates = []
        
        if not self.stats or not self.stats.products:
            return candidates
        
        for product in self.stats.products:
            sku = product.get('sku')
            category = product.get('category', '')
            brand = product.get('brand', '')
            price = product.get('price', 0)
            
            # Find similar products by attributes
            for other_product in self.stats.products:
                other_sku = other_product.get('sku')
                if other_sku == sku:
                    continue
                
                other_category = other_product.get('category', '')
                other_brand = other_product.get('brand', '')
                other_price = other_product.get('price', 0)
                
                attr_sim = 0.0
                
                # Same category bonus
                if category and category == other_category:
                    attr_sim += 0.5
                
                # Same brand bonus
                if brand and brand == other_brand:
                    attr_sim += 0.3
                
                # Price similarity (normalized difference)
                if price > 0 and other_price > 0:
                    price_diff = abs(price - other_price) / max(price, other_price)
                    price_sim = max(0, 1 - price_diff)  # 1 = same price, 0 = very different
                    attr_sim += 0.2 * price_sim
                
                if attr_sim > 0.4 and sku and other_sku:  # Minimum similarity threshold and valid SKUs
                    candidate = FallbackCandidate(
                        bundle_type="FIXED",
                        products=[sku, other_sku],
                        pricing={"discount_percent": 8},
                        features={
                            "attr_similarity": attr_sim,
                            "tier_weight": 0.4
                        },
                        source_tier="cold_start",
                        explanation=f"Attribute similarity: {category}/{brand}"
                    )
                    candidates.append(candidate)
        
        return candidates[:5]