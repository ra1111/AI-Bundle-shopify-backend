"""
Enterprise Constraint Management System
Handles business rules, inventory limits, and optimization constraints
"""
from typing import List, Dict, Any, Optional
import logging
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from services.storage import storage
from services.ml.optimization_engine import OptimizationConstraint

logger = logging.getLogger(__name__)

class ConstraintCategory(Enum):
    """Categories of business constraints"""
    INVENTORY = "inventory"
    FINANCIAL = "financial"
    BUSINESS_RULES = "business_rules"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"

@dataclass
class ConstraintTemplate:
    """Template for creating constraints"""
    name: str
    category: ConstraintCategory
    description: str
    constraint_type: str
    parameter: str
    operator: str
    default_value: Any
    is_hard: bool = True
    weight: float = 1.0

class EnterpriseConstraintManager:
    """Manages enterprise-level constraints for bundle optimization"""
    
    def __init__(self):
        # Predefined constraint templates
        self.constraint_templates = {
            # Inventory constraints
            "min_inventory_level": ConstraintTemplate(
                name="Minimum Inventory Level",
                category=ConstraintCategory.INVENTORY,
                description="Ensure minimum stock levels for bundle products",
                constraint_type="inventory",
                parameter="available_total",
                operator=">=",
                default_value=1,
                is_hard=True
            ),
            "inventory_turnover": ConstraintTemplate(
                name="Inventory Turnover Rate",
                category=ConstraintCategory.INVENTORY,
                description="Prefer products with higher turnover rates",
                constraint_type="inventory",
                parameter="velocity",
                operator=">=",
                default_value=0.1,
                is_hard=False,
                weight=0.3
            ),
            
            # Financial constraints
            "max_bundle_value": ConstraintTemplate(
                name="Maximum Bundle Value",
                category=ConstraintCategory.FINANCIAL,
                description="Limit maximum bundle price to maintain accessibility",
                constraint_type="budget",
                parameter="total_price",
                operator="<=",
                default_value=200,
                is_hard=True
            ),
            "min_margin_requirement": ConstraintTemplate(
                name="Minimum Margin Requirement",
                category=ConstraintCategory.FINANCIAL,
                description="Ensure minimum profit margin on bundles",
                constraint_type="margin",
                parameter="margin_score",
                operator=">=",
                default_value=0.10,
                is_hard=True
            ),
            "discount_limit": ConstraintTemplate(
                name="Maximum Discount Limit",
                category=ConstraintCategory.FINANCIAL,
                description="Limit maximum discount to protect margins",
                constraint_type="budget",
                parameter="discount_percentage",
                operator="<=",
                default_value=30,
                is_hard=True
            ),
            
            # Business rules
            "max_bundle_size": ConstraintTemplate(
                name="Maximum Bundle Size",
                category=ConstraintCategory.BUSINESS_RULES,
                description="Limit number of products in bundle for simplicity",
                constraint_type="product_limit",
                parameter="product_count",
                operator="<=",
                default_value=4,
                is_hard=True
            ),
            "category_diversity": ConstraintTemplate(
                name="Category Diversity",
                category=ConstraintCategory.BUSINESS_RULES,
                description="Require products from different categories",
                constraint_type="category_mix",
                parameter="unique_categories",
                operator=">=",
                default_value=2,
                is_hard=False,
                weight=0.4
            ),
            "seasonal_alignment": ConstraintTemplate(
                name="Seasonal Alignment",
                category=ConstraintCategory.BUSINESS_RULES,
                description="Prefer seasonal products during relevant periods",
                constraint_type="seasonal",
                parameter="is_seasonal",
                operator="==",
                default_value=True,
                is_hard=False,
                weight=0.2
            ),
            
            # Performance constraints
            "processing_time_limit": ConstraintTemplate(
                name="Processing Time Limit",
                category=ConstraintCategory.PERFORMANCE,
                description="Limit optimization processing time",
                constraint_type="performance",
                parameter="processing_time_ms",
                operator="<=",
                default_value=30000,  # 30 seconds
                is_hard=True
            ),
            "recommendation_count": ConstraintTemplate(
                name="Recommendation Count Limit",
                category=ConstraintCategory.PERFORMANCE,
                description="Limit number of recommendations generated",
                constraint_type="performance",
                parameter="recommendation_count",
                operator="<=",
                default_value=50,
                is_hard=True
            )
        }
        
        # Objective-specific constraint sets
        self.objective_constraint_sets = {
            "clear_slow_movers": [
                "min_inventory_level",
                "max_bundle_value", 
                "min_margin_requirement",
                "max_bundle_size",
                "processing_time_limit"
            ],
            "increase_aov": [
                "min_inventory_level",
                "max_bundle_value",
                "min_margin_requirement", 
                "category_diversity",
                "max_bundle_size",
                "processing_time_limit"
            ],
            "new_launch": [
                "min_inventory_level",
                "max_bundle_value",
                "category_diversity",
                "max_bundle_size",
                "processing_time_limit"
            ],
            "seasonal_promo": [
                "min_inventory_level",
                "max_bundle_value",
                "seasonal_alignment",
                "max_bundle_size",
                "processing_time_limit"
            ],
            "margin_guard": [
                "min_inventory_level",
                "min_margin_requirement",
                "discount_limit",
                "max_bundle_size",
                "processing_time_limit"
            ]
        }
    
    async def get_constraints_for_objective(self, objective: str, csv_upload_id: str, 
                                          custom_constraints: Optional[Dict[str, Any]] = None) -> List[OptimizationConstraint]:
        """Get constraint set for specific objective with dynamic values"""
        try:
            constraint_names = self.objective_constraint_sets.get(objective, [
                "min_inventory_level", "max_bundle_value", "max_bundle_size", "processing_time_limit"
            ])
            
            constraints = []
            
            # Get dynamic constraint values based on catalog data
            catalog_stats = await self._get_catalog_statistics(csv_upload_id)
            
            for constraint_name in constraint_names:
                template = self.constraint_templates.get(constraint_name)
                if not template:
                    continue
                
                # Apply dynamic values or custom overrides
                constraint_value = await self._get_dynamic_constraint_value(
                    template, catalog_stats, custom_constraints
                )
                
                constraint = OptimizationConstraint(
                    constraint_type=template.constraint_type,
                    parameter=template.parameter,
                    operator=template.operator,
                    value=constraint_value,
                    weight=template.weight,
                    is_hard=template.is_hard
                )
                
                constraints.append(constraint)
                logger.debug(f"Added constraint: {template.name} = {constraint_value}")
            
            logger.info(f"Generated {len(constraints)} constraints for objective: {objective}")
            return constraints
            
        except Exception as e:
            logger.error(f"Error generating constraints for {objective}: {e}")
            return []
    
    async def _get_catalog_statistics(self, csv_upload_id: str) -> Dict[str, Any]:
        """Get catalog statistics for dynamic constraint tuning"""
        try:
            catalog_items = await storage.get_catalog_snapshots_by_upload(csv_upload_id)
            
            if not catalog_items:
                return {}
            
            # Calculate statistics
            prices = [float(item.price) for item in catalog_items if item.price]
            inventories = [item.available_total for item in catalog_items if item.available_total > 0]
            
            stats = {
                "total_products": len(catalog_items),
                "avg_price": sum(prices) / len(prices) if prices else 0,
                "max_price": max(prices) if prices else 0,
                "min_price": min(prices) if prices else 0,
                "avg_inventory": sum(inventories) / len(inventories) if inventories else 0,
                "max_inventory": max(inventories) if inventories else 0,
                "min_inventory": min(inventories) if inventories else 0,
                "out_of_stock_count": sum(1 for item in catalog_items if item.available_total <= 0)
            }
            
            # Calculate margin statistics from objective flags
            high_margin_count = 0
            total_with_flags = 0
            
            for item in catalog_items:
                if hasattr(item, 'objective_flags') and item.objective_flags:
                    flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                    total_with_flags += 1
                    if flags.get("is_high_margin", False):
                        high_margin_count += 1
            
            stats["high_margin_ratio"] = high_margin_count / max(total_with_flags, 1)
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error getting catalog statistics: {e}")
            return {}
    
    async def _get_dynamic_constraint_value(self, template: ConstraintTemplate, 
                                          catalog_stats: Dict[str, Any],
                                          custom_constraints: Optional[Dict[str, Any]]) -> Any:
        """Calculate dynamic constraint value based on catalog data"""
        try:
            # Check for custom override first
            if custom_constraints and template.name in custom_constraints:
                return custom_constraints[template.name]
            
            # Apply dynamic logic based on constraint type
            if template.name == "min_inventory_level":
                # Set minimum inventory based on average inventory
                avg_inventory = catalog_stats.get("avg_inventory", 10)
                return max(1, int(avg_inventory * 0.1))  # 10% of average
            
            elif template.name == "max_bundle_value":
                # Set max bundle value based on price distribution
                avg_price = catalog_stats.get("avg_price", 50)
                return min(500, avg_price * 4)  # 4x average price
            
            elif template.name == "min_margin_requirement":
                # Adjust margin requirement based on catalog margin profile
                high_margin_ratio = catalog_stats.get("high_margin_ratio", 0.3)
                if high_margin_ratio > 0.5:
                    return 0.3  # Higher requirement for high-margin catalogs
                else:
                    return 0.2  # Lower requirement for mixed catalogs
            
            elif template.name == "discount_limit":
                # Adjust discount limit based on margin profile
                high_margin_ratio = catalog_stats.get("high_margin_ratio", 0.3)
                if high_margin_ratio > 0.5:
                    return 35  # Can afford higher discounts
                else:
                    return 25  # Conservative discounts
            
            elif template.name == "category_diversity":
                # Require diversity based on catalog size
                total_products = catalog_stats.get("total_products", 10)
                if total_products > 100:
                    return 3  # Require 3 categories for large catalogs
                elif total_products > 50:
                    return 2  # Require 2 categories for medium catalogs
                else:
                    return 1  # Flexible for small catalogs
            
            else:
                # Use template default
                return template.default_value
                
        except Exception as e:
            logger.warning(f"Error calculating dynamic constraint value for {template.name}: {e}")
            return template.default_value
    
    def validate_constraint_set(self, constraints: List[OptimizationConstraint]) -> Dict[str, Any]:
        """Validate constraint set for conflicts and feasibility"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Check for conflicting constraints
            inventory_constraints = [c for c in constraints if c.constraint_type == "inventory"]
            budget_constraints = [c for c in constraints if c.constraint_type == "budget"]
            
            # Check inventory constraint conflicts
            for i, constraint_a in enumerate(inventory_constraints):
                for constraint_b in inventory_constraints[i+1:]:
                    if (constraint_a.parameter == constraint_b.parameter and 
                        constraint_a.operator != constraint_b.operator):
                        validation_result["warnings"].append(
                            f"Conflicting inventory constraints on {constraint_a.parameter}"
                        )
            
            # Check for overly restrictive constraints
            for constraint in constraints:
                if constraint.constraint_type == "inventory" and constraint.value > 100:
                    validation_result["warnings"].append(
                        "High inventory requirement may severely limit candidate pool"
                    )
                elif constraint.constraint_type == "budget" and constraint.value < 10:
                    validation_result["warnings"].append(
                        "Very low budget constraint may eliminate all candidates"
                    )
            
            # Performance recommendations
            hard_constraint_count = sum(1 for c in constraints if c.is_hard)
            if hard_constraint_count > 10:
                validation_result["recommendations"].append(
                    "Consider making some constraints soft to improve optimization performance"
                )
            
            if len(constraints) > 15:
                validation_result["recommendations"].append(
                    "Large constraint sets may impact optimization speed"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating constraint set: {e}")
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result
    
    def get_constraint_explanation(self, constraint: OptimizationConstraint) -> str:
        """Generate human-readable explanation for constraint"""
        try:
            # Find matching template
            template = None
            for tmpl in self.constraint_templates.values():
                if (tmpl.constraint_type == constraint.constraint_type and 
                    tmpl.parameter == constraint.parameter):
                    template = tmpl
                    break
            
            if template:
                base_explanation = template.description
                constraint_type = "Hard" if constraint.is_hard else "Soft"
                
                return f"{constraint_type} constraint: {base_explanation} ({constraint.parameter} {constraint.operator} {constraint.value})"
            else:
                return f"Custom constraint: {constraint.parameter} {constraint.operator} {constraint.value}"
                
        except Exception as e:
            logger.warning(f"Error generating constraint explanation: {e}")
            return f"Constraint: {constraint.parameter} {constraint.operator} {constraint.value}"
    
    def get_constraint_categories_summary(self, constraints: List[OptimizationConstraint]) -> Dict[str, int]:
        """Get summary of constraints by category"""
        try:
            category_counts = {}
            
            for constraint in constraints:
                # Map constraint types to categories
                if constraint.constraint_type in ["inventory"]:
                    category = "Inventory"
                elif constraint.constraint_type in ["budget", "margin"]:
                    category = "Financial"
                elif constraint.constraint_type in ["product_limit", "category_mix", "seasonal"]:
                    category = "Business Rules"
                elif constraint.constraint_type in ["performance"]:
                    category = "Performance"
                else:
                    category = "Other"
                
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return category_counts
            
        except Exception as e:
            logger.warning(f"Error getting constraint summary: {e}")
            return {}
    
    async def suggest_constraint_adjustments(self, objective: str, csv_upload_id: str, 
                                           optimization_results: Dict[str, Any]) -> List[str]:
        """Suggest constraint adjustments based on optimization results"""
        suggestions = []
        
        try:
            metrics = optimization_results.get("metrics", {})
            feasible_solutions = metrics.get("feasible_solutions", 0)
            pareto_solutions = metrics.get("pareto_solutions", 0)
            
            # Low feasibility suggestions
            if feasible_solutions < 5:
                suggestions.append(
                    "Very few feasible solutions found. Consider relaxing inventory or budget constraints."
                )
            
            # No Pareto solutions
            if pareto_solutions == 0:
                suggestions.append(
                    "No Pareto-optimal solutions found. Consider adjusting objective weights or constraint values."
                )
            
            # Performance suggestions
            processing_time = metrics.get("processing_time_ms", 0)
            if processing_time > 20000:  # 20 seconds
                suggestions.append(
                    "Optimization took longer than expected. Consider reducing constraint complexity or population size."
                )
            
            # Objective-specific suggestions
            if objective == "clear_slow_movers" and pareto_solutions < 10:
                suggestions.append(
                    "Limited slow-mover bundle options. Consider relaxing margin requirements or increasing inventory tolerance."
                )
            
            elif objective == "increase_aov" and pareto_solutions < 10:
                suggestions.append(
                    "Limited high-value bundle options. Consider increasing maximum bundle value constraint."
                )
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Error generating constraint suggestions: {e}")
            return ["Unable to generate constraint suggestions due to analysis error."]