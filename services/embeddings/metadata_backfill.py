"""
Metadata-based Product Similarity for Cold-Start Coverage (PR-6)
Handles new products with no purchase history using product attributes
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import re

from services.storage import storage

logger = logging.getLogger(__name__)

@dataclass
class ProductSimilarity:
    """Product similarity result"""
    product_a: str
    product_b: str
    similarity_score: float
    similarity_factors: Dict[str, float]
    confidence: float

@dataclass
class ColdStartRecommendation:
    """Cold-start bundle recommendation"""
    anchor_product: str
    recommended_products: List[str]
    similarity_scores: List[float]
    reasoning: str
    confidence: float
    fallback_method: str

class MetadataBackfillEngine:
    """Generates product similarities based on metadata when purchase data is unavailable"""
    
    def __init__(self):
        # Similarity weights for different metadata factors
        self.similarity_weights = {
            "category": 0.35,      # Same category = strong similarity
            "brand": 0.25,         # Same brand = moderate similarity  
            "price_band": 0.20,    # Similar price = moderate similarity
            "title_similarity": 0.15,  # Title overlap = weak similarity
            "vendor": 0.05         # Same vendor = weak similarity
        }
        
        # Price band ranges (relative to dataset)
        self.price_bands = [
            (0, 25, "budget"),
            (25, 75, "mid_range"), 
            (75, 150, "premium"),
            (150, float('inf'), "luxury")
        ]
        
        # Category hierarchy weights
        self.category_weights = {
            "exact_match": 1.0,
            "parent_match": 0.7,
            "sibling_match": 0.4,
            "cousin_match": 0.2
        }
        
    async def generate_cold_start_similarities(self, csv_upload_id: str, 
                                             new_products: List[str]) -> Dict[str, List[ProductSimilarity]]:
        """Generate similarity recommendations for new products with no purchase history"""
        try:
            logger.info(f"Generating cold-start similarities for {len(new_products)} new products")
            
            # Get all product metadata from the dataset
            all_products = await storage.get_products_by_upload(csv_upload_id)
            variants = await storage.get_variants_by_upload(csv_upload_id)
            
            # Create product metadata lookup
            product_metadata = await self._build_product_metadata(all_products, variants, csv_upload_id)
            
            # Calculate price bands for this dataset
            price_bands = self._calculate_dataset_price_bands(product_metadata)
            
            similarities = {}
            
            for new_product in new_products:
                if new_product not in product_metadata:
                    logger.warning(f"No metadata found for new product: {new_product}")
                    continue
                
                # Find similar products using metadata
                similar_products = await self._find_similar_products(
                    new_product, product_metadata, price_bands, csv_upload_id
                )
                
                similarities[new_product] = similar_products
            
            logger.info(f"Generated similarities for {len(similarities)} new products")
            return similarities
            
        except Exception as e:
            logger.error(f"Error generating cold-start similarities: {e}")
            return {}
    
    async def _build_product_metadata(self, products: List, variants: List, csv_upload_id: str) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive product metadata lookup"""
        try:
            metadata = {}
            
            # Build from products
            for product in products:
                if not product.product_id:
                    continue
                    
                metadata[product.product_id] = {
                    "title": product.title or "",
                    "category": product.category or "uncategorized",
                    "tags": product.tags or [],
                    "vendor": product.vendor or "unknown",
                    "product_type": product.product_type or "general",
                    "variants": []
                }
            
            # Add variant information
            for variant in variants:
                if not variant.product_id or variant.product_id not in metadata:
                    continue
                
                variant_info = {
                    "sku": variant.variant_sku,
                    "title": variant.variant_title or "",
                    "price": float(variant.price) if variant.price else 0.0,
                    "compare_at_price": float(variant.compare_at_price) if variant.compare_at_price else 0.0,
                    "weight": float(variant.weight) if variant.weight else 0.0,
                    "option1": variant.option1_value or "",
                    "option2": variant.option2_value or "",
                    "option3": variant.option3_value or ""
                }
                
                metadata[variant.product_id]["variants"].append(variant_info)
            
            # Calculate aggregate metrics per product
            for product_id, product_data in metadata.items():
                if product_data["variants"]:
                    prices = [v["price"] for v in product_data["variants"] if v["price"] > 0]
                    if prices:
                        product_data["avg_price"] = np.mean(prices)
                        product_data["min_price"] = min(prices)
                        product_data["max_price"] = max(prices)
                        product_data["price_range"] = max(prices) - min(prices)
                    else:
                        product_data["avg_price"] = 0.0
                        product_data["min_price"] = 0.0
                        product_data["max_price"] = 0.0
                        product_data["price_range"] = 0.0
                else:
                    product_data["avg_price"] = 0.0
                    product_data["min_price"] = 0.0
                    product_data["max_price"] = 0.0
                    product_data["price_range"] = 0.0
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error building product metadata: {e}")
            return {}
    
    def _calculate_dataset_price_bands(self, product_metadata: Dict[str, Dict[str, Any]]) -> List[Tuple[float, float, str]]:
        """Calculate price bands based on actual dataset price distribution"""
        try:
            prices = []
            for product_data in product_metadata.values():
                if product_data.get("avg_price", 0) > 0:
                    prices.append(product_data["avg_price"])
            
            if not prices:
                return self.price_bands
            
            # Calculate percentile-based price bands
            prices.sort()
            p25 = np.percentile(prices, 25)
            p50 = np.percentile(prices, 50)
            p75 = np.percentile(prices, 75)
            p90 = np.percentile(prices, 90)
            
            return [
                (0, p25, "budget"),
                (p25, p50, "economy"),
                (p50, p75, "mid_range"),
                (p75, p90, "premium"),
                (p90, float('inf'), "luxury")
            ]
            
        except Exception as e:
            logger.warning(f"Error calculating price bands: {e}")
            return self.price_bands
    
    async def _find_similar_products(self, target_product: str, 
                                   product_metadata: Dict[str, Dict[str, Any]], 
                                   price_bands: List[Tuple[float, float, str]], 
                                   csv_upload_id: str) -> List[ProductSimilarity]:
        """Find products similar to target product using metadata"""
        try:
            if target_product not in product_metadata:
                return []
            
            target_data = product_metadata[target_product]
            similarities = []
            
            for candidate_product, candidate_data in product_metadata.items():
                if candidate_product == target_product:
                    continue
                
                # Calculate similarity score
                similarity_factors = {}
                
                # Category similarity
                similarity_factors["category"] = self._calculate_category_similarity(
                    target_data.get("category", ""), candidate_data.get("category", "")
                )
                
                # Brand similarity
                similarity_factors["brand"] = self._calculate_brand_similarity(
                    target_data.get("vendor", ""), candidate_data.get("vendor", "")
                )
                
                # Price band similarity
                similarity_factors["price_band"] = self._calculate_price_similarity(
                    target_data.get("avg_price", 0), candidate_data.get("avg_price", 0), price_bands
                )
                
                # Title similarity
                similarity_factors["title_similarity"] = self._calculate_title_similarity(
                    target_data.get("title", ""), candidate_data.get("title", "")
                )
                
                # Vendor similarity (additional to brand)
                similarity_factors["vendor"] = 1.0 if target_data.get("vendor") == candidate_data.get("vendor") else 0.0
                
                # Calculate weighted similarity score
                total_score = 0.0
                for factor, score in similarity_factors.items():
                    weight = self.similarity_weights.get(factor, 0.0)
                    total_score += weight * score
                
                # Calculate confidence based on available metadata quality
                confidence = self._calculate_confidence(target_data, candidate_data, similarity_factors)
                
                if total_score > 0.2:  # Minimum similarity threshold
                    similarities.append(ProductSimilarity(
                        product_a=target_product,
                        product_b=candidate_product,
                        similarity_score=total_score,
                        similarity_factors=similarity_factors,
                        confidence=confidence
                    ))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top 10 most similar products
            return similarities[:10]
            
        except Exception as e:
            logger.error(f"Error finding similar products for {target_product}: {e}")
            return []
    
    def _calculate_category_similarity(self, cat1: str, cat2: str) -> float:
        """Calculate category similarity with hierarchy awareness"""
        if not cat1 or not cat2:
            return 0.0
        
        cat1_parts = cat1.lower().split(' > ')
        cat2_parts = cat2.lower().split(' > ')
        
        # Exact match
        if cat1.lower() == cat2.lower():
            return 1.0
        
        # Check hierarchy levels
        common_levels = 0
        for i in range(min(len(cat1_parts), len(cat2_parts))):
            if cat1_parts[i] == cat2_parts[i]:
                common_levels += 1
            else:
                break
        
        if common_levels == 0:
            return 0.0
        elif common_levels == len(cat1_parts) - 1 or common_levels == len(cat2_parts) - 1:
            return 0.7  # Parent-child relationship
        elif common_levels >= 1:
            return 0.4  # Sibling categories
        else:
            return 0.2  # Distant cousins
    
    def _calculate_brand_similarity(self, brand1: str, brand2: str) -> float:
        """Calculate brand similarity"""
        if not brand1 or not brand2:
            return 0.0
        
        if brand1.lower() == brand2.lower():
            return 1.0
        
        # Check for partial brand matches (e.g., "Nike" vs "Nike Sports")
        brand1_clean = re.sub(r'[^a-z0-9]', '', brand1.lower())
        brand2_clean = re.sub(r'[^a-z0-9]', '', brand2.lower())
        
        if brand1_clean in brand2_clean or brand2_clean in brand1_clean:
            return 0.7
        
        return 0.0
    
    def _calculate_price_similarity(self, price1: float, price2: float, 
                                  price_bands: List[Tuple[float, float, str]]) -> float:
        """Calculate price band similarity"""
        if price1 <= 0 or price2 <= 0:
            return 0.0
        
        # Find price bands for both products
        band1 = self._get_price_band(price1, price_bands)
        band2 = self._get_price_band(price2, price_bands)
        
        if band1 == band2:
            return 1.0
        
        # Adjacent price bands get partial similarity
        band_names = [band[2] for band in price_bands]
        try:
            idx1 = band_names.index(band1)
            idx2 = band_names.index(band2)
            
            if abs(idx1 - idx2) == 1:
                return 0.5  # Adjacent bands
            elif abs(idx1 - idx2) == 2:
                return 0.2  # One band apart
            else:
                return 0.0  # Too far apart
        except (ValueError, IndexError):
            return 0.0
    
    def _get_price_band(self, price: float, price_bands: List[Tuple[float, float, str]]) -> str:
        """Get price band for a given price"""
        for min_price, max_price, band_name in price_bands:
            if min_price <= price < max_price:
                return band_name
        return "unknown"
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using word overlap"""
        if not title1 or not title2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(re.findall(r'\w+', title1.lower()))
        words2 = set(re.findall(r'\w+', title2.lower()))
        
        # Remove common stop words
        stop_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence(self, target_data: Dict[str, Any], 
                            candidate_data: Dict[str, Any], 
                            similarity_factors: Dict[str, float]) -> float:
        """Calculate confidence in similarity recommendation"""
        confidence_factors = []
        
        # Data completeness
        target_completeness = self._calculate_data_completeness(target_data)
        candidate_completeness = self._calculate_data_completeness(candidate_data)
        data_quality = (target_completeness + candidate_completeness) / 2
        confidence_factors.append(data_quality)
        
        # Similarity strength
        max_similarity = max(similarity_factors.values()) if similarity_factors else 0
        confidence_factors.append(max_similarity)
        
        # Multiple similarity factors
        strong_factors = sum(1 for score in similarity_factors.values() if score > 0.5)
        factor_diversity = min(1.0, strong_factors / 3)  # Normalize to 3 factors
        confidence_factors.append(factor_diversity)
        
        return np.mean(confidence_factors)
    
    def _calculate_data_completeness(self, product_data: Dict[str, Any]) -> float:
        """Calculate how complete the product metadata is"""
        required_fields = ["title", "category", "vendor", "avg_price"]
        present_fields = sum(1 for field in required_fields if product_data.get(field))
        return present_fields / len(required_fields)