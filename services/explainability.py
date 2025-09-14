"""
Explainability Service
Merchant-facing explanation strings for bundle recommendations
"""
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """Service for generating human-readable explanations of bundle recommendations"""
    
    def __init__(self):
        # Templates for different explanation types
        self.templates = {
            "statistical": {
                "high_confidence": "Products frequently bought together (confidence: {confidence:.0%})",
                "high_lift": "Customers {lift:.1f}x more likely to buy together",
                "high_support": "Appears in {support:.1%} of multi-item orders"
            },
            "objective": {
                "clear_slow_movers": "Helps move slow-selling inventory",
                "increase_aov": "Increases average order value",
                "new_launch": "Promotes newly launched products",
                "seasonal_promo": "Perfect for seasonal promotions",
                "margin_guard": "Maintains healthy profit margins",
                "category_bundle": "Cross-category bundle opportunity",
                "gift_box": "Great gift bundle combination",
                "subscription_push": "Encourages subscription signup"
            },
            "inventory": {
                "in_stock": "All items in stock",
                "limited_stock": "Limited stock available",
                "not_tracked": "Inventory not tracked",
                "mixed_stock": "Mixed inventory levels"
            },
            "pricing": {
                "good_discount": "Reasonable discount based on history",
                "aggressive_discount": "Higher discount to drive sales",
                "conservative_discount": "Conservative discount to protect margins"
            }
        }
    
    def generate_explanation(self, recommendation: Dict[str, Any], 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive explanation for a bundle recommendation"""
        try:
            explanation_parts = []
            
            # 1. Statistical explanation
            stat_explanation = self.generate_statistical_explanation(recommendation)
            if stat_explanation:
                explanation_parts.append(stat_explanation)
            
            # 2. Objective explanation
            obj_explanation = self.generate_objective_explanation(recommendation)
            if obj_explanation:
                explanation_parts.append(obj_explanation)
            
            # 3. Inventory status
            inventory_explanation = self.generate_inventory_explanation(recommendation)
            if inventory_explanation:
                explanation_parts.append(inventory_explanation)
            
            # 4. Pricing explanation
            pricing_explanation = self.generate_pricing_explanation(recommendation)
            if pricing_explanation:
                explanation_parts.append(pricing_explanation)
            
            # 5. Generation method context
            method_explanation = self.generate_method_explanation(recommendation)
            if method_explanation:
                explanation_parts.append(method_explanation)
            
            # Combine all parts
            if explanation_parts:
                full_explanation = ". ".join(explanation_parts) + "."
                return self.polish_explanation(full_explanation)
            else:
                return self.generate_fallback_explanation(recommendation)
                
        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            return self.generate_fallback_explanation(recommendation)
    
    def generate_statistical_explanation(self, recommendation: Dict[str, Any]) -> Optional[str]:
        """Generate explanation based on statistical measures"""
        try:
            confidence = recommendation.get("confidence", 0)
            lift = recommendation.get("lift", 1)
            support = recommendation.get("support", 0)
            
            explanations = []
            
            # High confidence
            if confidence >= 0.4:
                explanations.append(
                    self.templates["statistical"]["high_confidence"].format(confidence=confidence)
                )
            
            # High lift
            if lift >= 2.0:
                explanations.append(
                    self.templates["statistical"]["high_lift"].format(lift=lift)
                )
            
            # Decent support
            if support >= 0.05:
                explanations.append(
                    self.templates["statistical"]["high_support"].format(support=support)
                )
            
            # Choose the most impressive statistic
            if explanations:
                if confidence >= 0.6:
                    return explanations[0]  # Lead with confidence if very high
                elif lift >= 3.0:
                    return explanations[1] if len(explanations) > 1 else explanations[0]
                else:
                    return explanations[0]
            
            # Fallback for lower statistics
            if confidence > 0.2:
                return f"Co-purchased with {confidence:.0%} confidence"
            
            return None
            
        except Exception as e:
            logger.warning(f"Error generating statistical explanation: {e}")
            return None
    
    def generate_objective_explanation(self, recommendation: Dict[str, Any]) -> Optional[str]:
        """Generate explanation based on objective alignment"""
        try:
            objective = recommendation.get("objective", "")
            objective_fit = recommendation.get("objective_fit_raw", 0)
            
            if not objective or objective_fit < 0.3:
                return None
            
            base_explanation = self.templates["objective"].get(objective, "")
            
            # Enhance based on objective fit score
            if objective_fit >= 0.8:
                enhancement = "perfectly"
            elif objective_fit >= 0.6:
                enhancement = "well"
            else:
                enhancement = ""
            
            if base_explanation and enhancement:
                return f"{enhancement} {base_explanation.lower()}"
            elif base_explanation:
                return base_explanation
            else:
                return f"Supports '{objective}' objective"
                
        except Exception as e:
            logger.warning(f"Error generating objective explanation: {e}")
            return None
    
    def generate_inventory_explanation(self, recommendation: Dict[str, Any]) -> Optional[str]:
        """Generate explanation based on inventory status"""
        try:
            # Get inventory data from ranking components or product details
            inventory_term = recommendation.get("inventory_term_raw", 0)
            pricing = recommendation.get("pricing", {})
            product_details = pricing.get("product_details", [])
            
            if not product_details:
                return None
            
            # Analyze inventory levels
            in_stock = 0
            out_of_stock = 0
            not_tracked = 0
            limited_stock = 0
            
            for product in product_details:
                stock = getattr(product, 'stock_available', None)
                if isinstance(stock, dict):
                    stock = stock.get('available_total', -1)
                elif stock is None:
                    stock = -1
                
                if stock > 10:
                    in_stock += 1
                elif stock > 0:
                    limited_stock += 1
                elif stock == -1:
                    not_tracked += 1
                else:
                    out_of_stock += 1
            
            total_products = len(product_details)
            
            # Generate appropriate message
            if out_of_stock > 0:
                return None  # Don't mention inventory if any items are out of stock
            elif in_stock == total_products:
                return self.templates["inventory"]["in_stock"]
            elif limited_stock > 0:
                return self.templates["inventory"]["limited_stock"]
            elif not_tracked == total_products:
                return None  # Don't mention if inventory not tracked
            else:
                return self.templates["inventory"]["mixed_stock"]
                
        except Exception as e:
            logger.warning(f"Error generating inventory explanation: {e}")
            return None
    
    def generate_pricing_explanation(self, recommendation: Dict[str, Any]) -> Optional[str]:
        """Generate explanation based on pricing strategy"""
        try:
            pricing = recommendation.get("pricing", {})
            total_discount_pct = pricing.get("total_discount_pct", 0)
            price_sanity = recommendation.get("price_sanity_raw", 0.5)
            
            if total_discount_pct <= 0:
                return None
            
            # Determine pricing explanation based on discount level and sanity
            if total_discount_pct >= 25:
                if price_sanity >= 0.7:
                    return self.templates["pricing"]["aggressive_discount"]
                else:
                    return None  # Don't explain if price sanity is poor
            elif total_discount_pct >= 10:
                return self.templates["pricing"]["good_discount"]
            else:
                return self.templates["pricing"]["conservative_discount"]
                
        except Exception as e:
            logger.warning(f"Error generating pricing explanation: {e}")
            return None
    
    def generate_method_explanation(self, recommendation: Dict[str, Any]) -> Optional[str]:
        """Generate explanation based on generation method"""
        try:
            generation_sources = recommendation.get("generation_sources", [])
            generation_method = recommendation.get("generation_method", "")
            
            if not generation_sources and not generation_method:
                return None
            
            # Explain generation approach
            if "item2vec" in generation_sources:
                return "Based on product similarity analysis"
            elif "fpgrowth" in generation_sources or generation_method == "fpgrowth":
                return "Identified through advanced pattern mining"
            elif "apriori" in generation_sources or generation_method == "apriori":
                return "Found through market basket analysis"
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error generating method explanation: {e}")
            return None
    
    def polish_explanation(self, explanation: str) -> str:
        """Polish the explanation text for better readability"""
        try:
            # Remove duplicate periods
            explanation = explanation.replace("..", ".")
            
            # Ensure proper capitalization
            if explanation and explanation[0].islower():
                explanation = explanation[0].upper() + explanation[1:]
            
            # Add final period if missing
            if explanation and not explanation.endswith('.'):
                explanation += "."
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Error polishing explanation: {e}")
            return explanation
    
    def generate_fallback_explanation(self, recommendation: Dict[str, Any]) -> str:
        """Generate basic fallback explanation"""
        try:
            products = recommendation.get("products", [])
            bundle_type = recommendation.get("bundle_type", "bundle")
            
            product_count = len(products)
            
            if product_count == 2:
                return f"Recommended {bundle_type.lower().replace('_', ' ')} pairing."
            elif product_count > 2:
                return f"Recommended {product_count}-item {bundle_type.lower().replace('_', ' ')}."
            else:
                return "Recommended bundle based on customer behavior."
                
        except Exception as e:
            logger.warning(f"Error generating fallback explanation: {e}")
            return "Recommended bundle."
    
    def generate_detailed_explanation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanation with component breakdown"""
        try:
            explanation_data = {
                "summary": self.generate_explanation(recommendation),
                "components": {},
                "statistics": {},
                "reasoning": {}
            }
            
            # Statistical components
            confidence = recommendation.get("confidence", 0)
            lift = recommendation.get("lift", 1)
            support = recommendation.get("support", 0)
            
            explanation_data["statistics"] = {
                "confidence": {
                    "value": float(confidence),
                    "interpretation": self.interpret_confidence(confidence)
                },
                "lift": {
                    "value": float(lift),
                    "interpretation": self.interpret_lift(lift)
                },
                "support": {
                    "value": float(support),
                    "interpretation": self.interpret_support(support)
                }
            }
            
            # Ranking components
            ranking_components = recommendation.get("ranking_components", {})
            if ranking_components:
                explanation_data["components"]["ranking_breakdown"] = {
                    "total_score": recommendation.get("ranking_score", 0),
                    "confidence_contribution": ranking_components.get("confidence_contribution", 0),
                    "lift_contribution": ranking_components.get("lift_contribution", 0),
                    "objective_contribution": ranking_components.get("objective_fit_contribution", 0),
                    "inventory_contribution": ranking_components.get("inventory_contribution", 0),
                    "price_contribution": ranking_components.get("price_sanity_contribution", 0)
                }
            
            # Objective reasoning
            objective = recommendation.get("objective", "")
            objective_fit = recommendation.get("objective_fit_raw", 0)
            if objective:
                explanation_data["reasoning"]["objective"] = {
                    "objective_name": objective,
                    "fit_score": float(objective_fit),
                    "explanation": self.templates["objective"].get(objective, "")
                }
            
            # Generation reasoning
            generation_sources = recommendation.get("generation_sources", [])
            if generation_sources:
                explanation_data["reasoning"]["generation"] = {
                    "methods": generation_sources,
                    "primary_method": generation_sources[0] if generation_sources else "",
                    "explanation": self.explain_generation_methods(generation_sources)
                }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating detailed explanation: {e}")
            return {
                "summary": self.generate_fallback_explanation(recommendation),
                "error": str(e)
            }
    
    def interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence value for merchants"""
        if confidence >= 0.7:
            return "Very strong co-purchase pattern"
        elif confidence >= 0.5:
            return "Strong co-purchase pattern"
        elif confidence >= 0.3:
            return "Moderate co-purchase pattern"
        else:
            return "Weak co-purchase pattern"
    
    def interpret_lift(self, lift: float) -> str:
        """Interpret lift value for merchants"""
        if lift >= 3.0:
            return "Customers much more likely to buy together"
        elif lift >= 2.0:
            return "Customers significantly more likely to buy together"
        elif lift >= 1.5:
            return "Customers moderately more likely to buy together"
        else:
            return "Slight increase in purchase likelihood"
    
    def interpret_support(self, support: float) -> str:
        """Interpret support value for merchants"""
        if support >= 0.1:
            return "Common in customer orders"
        elif support >= 0.05:
            return "Appears regularly in orders"
        elif support >= 0.02:
            return "Occasionally seen in orders"
        else:
            return "Rare but meaningful pattern"
    
    def explain_generation_methods(self, methods: List[str]) -> str:
        """Explain what generation methods mean"""
        explanations = []
        
        if "apriori" in methods:
            explanations.append("traditional market basket analysis")
        if "fpgrowth" in methods:
            explanations.append("advanced pattern mining")
        if "item2vec" in methods:
            explanations.append("product similarity analysis")
        
        if len(explanations) > 1:
            return f"Found through {', '.join(explanations[:-1])} and {explanations[-1]}"
        elif explanations:
            return f"Found through {explanations[0]}"
        else:
            return "Identified through data analysis"
    
    def generate_merchant_summary(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level summary for merchants"""
        try:
            if not recommendations:
                return {"message": "No bundle recommendations available"}
            
            # Aggregate statistics
            total_bundles = len(recommendations)
            avg_confidence = sum(r.get("confidence", 0) for r in recommendations) / total_bundles
            avg_lift = sum(r.get("lift", 1) for r in recommendations) / total_bundles
            
            # Count by bundle type
            type_counts = {}
            for rec in recommendations:
                bundle_type = rec.get("bundle_type", "unknown")
                type_counts[bundle_type] = type_counts.get(bundle_type, 0) + 1
            
            # Count by objective
            objective_counts = {}
            for rec in recommendations:
                objective = rec.get("objective", "unknown")
                objective_counts[objective] = objective_counts.get(objective, 0) + 1
            
            summary = {
                "total_recommendations": total_bundles,
                "average_confidence": round(avg_confidence, 3),
                "average_lift": round(avg_lift, 2),
                "bundle_types": type_counts,
                "objectives": objective_counts,
                "top_recommendation": recommendations[0] if recommendations else None
            }
            
            # Generate narrative summary
            if total_bundles > 0:
                top_type = max(type_counts.items(), key=lambda x: x[1])[0]
                top_objective = max(objective_counts.items(), key=lambda x: x[1])[0]
                
                summary["narrative"] = (
                    f"Generated {total_bundles} bundle recommendations with "
                    f"{avg_confidence:.0%} average confidence. "
                    f"Most common type: {top_type.replace('_', ' ').title()}. "
                    f"Primary objective: {top_objective.replace('_', ' ').title()}."
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating merchant summary: {e}")
            return {"error": str(e)}