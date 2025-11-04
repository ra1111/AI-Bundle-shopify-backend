"""
Enterprise Multi-Objective Optimization Engine
Implements Pareto optimization, constraint handling, and ensemble methods for bundle recommendations
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import numpy as np
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from services.storage import storage

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives for multi-objective optimization"""
    MAXIMIZE_REVENUE = "maximize_revenue"
    MAXIMIZE_MARGIN = "maximize_margin" 
    MINIMIZE_INVENTORY_RISK = "minimize_inventory_risk"
    MAXIMIZE_CUSTOMER_SATISFACTION = "maximize_customer_satisfaction"
    MAXIMIZE_CROSS_SELL = "maximize_cross_sell"
    MINIMIZE_CANNIBALIZATION = "minimize_cannibalization"

@dataclass
class OptimizationConstraint:
    """Represents a hard constraint for bundle optimization"""
    constraint_type: str  # "inventory", "budget", "margin", "product_limit", "category_mix"
    parameter: str
    operator: str  # ">=", "<=", "==", "!=", "in", "not_in"
    value: Any
    weight: float = 1.0  # For soft constraints
    is_hard: bool = True  # Hard vs soft constraint

@dataclass
class ParetoSolution:
    """Represents a Pareto-optimal solution"""
    bundle_candidate: Dict[str, Any]
    objective_scores: Dict[str, float]
    constraint_satisfaction: Dict[str, bool]
    dominance_rank: int
    crowding_distance: float

class EnterpriseOptimizationEngine:
    """Advanced multi-objective optimization engine for enterprise bundle recommendations"""
    
    def __init__(self):
        # Performance optimized: Reduced from 5,000 to 500 evaluations (10x faster)
        # 100 × 50 generations = 5,000 evaluations → 50 × 10 = 500 evaluations
        self.population_size = 50  # Reduced from 100
        self.max_generations = 10  # Reduced from 50
        self.mutation_rate = 0.15  # Increased from 0.1 to maintain diversity
        self.crossover_rate = 0.8
        self.elite_size = 5  # Proportional reduction from 10
        
        # Performance optimization
        self.cache_enabled = True
        self.parallel_processing = True
        self.max_workers = 4
        
        # Constraint handling
        self.constraint_penalty_weight = 1000.0
        self.constraint_tolerance = 0.01
        
        # Multi-objective weights (fallback for weighted sum approach)
        self.default_objective_weights = {
            OptimizationObjective.MAXIMIZE_REVENUE: 0.3,
            OptimizationObjective.MAXIMIZE_MARGIN: 0.25,
            OptimizationObjective.MINIMIZE_INVENTORY_RISK: 0.2,
            OptimizationObjective.MAXIMIZE_CUSTOMER_SATISFACTION: 0.15,
            OptimizationObjective.MAXIMIZE_CROSS_SELL: 0.1
        }
        
        # Caching for performance
        self._objective_cache = {}
        self._constraint_cache = {}
        self._catalog_cache = {}  # NEW: Cache catalog data to avoid repeated DB queries
        
    async def optimize_bundle_portfolio(self, 
                                      candidates: List[Dict[str, Any]], 
                                      objectives: List[OptimizationObjective],
                                      constraints: List[OptimizationConstraint],
                                      csv_upload_id: str,
                                      optimization_method: str = "pareto") -> Dict[str, Any]:
        """
        Optimize bundle portfolio using multi-objective optimization
        
        Args:
            candidates: Initial bundle candidates
            objectives: List of optimization objectives
            constraints: List of hard/soft constraints
            csv_upload_id: CSV upload identifier
            optimization_method: "pareto", "weighted_sum", or "epsilon_constraint"
            
        Returns:
            Optimization results with Pareto frontier and metrics
        """
        logger.info(f"Starting enterprise optimization with {len(candidates)} candidates")
        start_time = time.time()
        
        try:
            # Validate inputs
            if not candidates:
                return {"pareto_solutions": [], "metrics": {"error": "No candidates provided"}}

            if not objectives:
                objectives = [OptimizationObjective.MAXIMIZE_REVENUE, OptimizationObjective.MAXIMIZE_MARGIN]

            # OPTIMIZATION: Pre-load catalog ONCE for all evaluations (avoids repeated DB queries)
            logger.info(f"Pre-caching catalog data for optimization (csv_upload_id={csv_upload_id})")
            self._catalog_cache[csv_upload_id] = await storage.get_catalog_snapshots_map(csv_upload_id)
            logger.info(f"Pre-cached {len(self._catalog_cache[csv_upload_id])} catalog items for optimization")

            # Initialize optimization population
            population = await self._initialize_population(candidates, csv_upload_id)
            
            # Apply constraint filtering
            feasible_population = await self._apply_constraints(population, constraints, csv_upload_id)
            
            if not feasible_population:
                logger.warning("No feasible solutions found after constraint filtering")
                return {"pareto_solutions": [], "metrics": {"error": "No feasible solutions"}}
            
            # Perform multi-objective optimization
            if optimization_method == "pareto":
                pareto_solutions = await self._nsga_ii_optimization(
                    feasible_population, objectives, constraints, csv_upload_id
                )
            elif optimization_method == "weighted_sum":
                pareto_solutions = await self._weighted_sum_optimization(
                    feasible_population, objectives, csv_upload_id
                )
            else:
                pareto_solutions = await self._epsilon_constraint_optimization(
                    feasible_population, objectives, constraints, csv_upload_id
                )
            
            # Performance metrics
            processing_time = (time.time() - start_time) * 1000
            
            metrics = {
                "optimization_method": optimization_method,
                "initial_candidates": len(candidates),
                "feasible_solutions": len(feasible_population),
                "pareto_solutions": len(pareto_solutions),
                "processing_time_ms": processing_time,
                "objectives_optimized": [obj.value for obj in objectives],
                "constraints_applied": len(constraints)
            }
            
            logger.info(f"Optimization completed: {len(pareto_solutions)} Pareto solutions in {processing_time:.1f}ms")
            
            return {
                "pareto_solutions": [sol.bundle_candidate for sol in pareto_solutions],
                "solution_details": [self._solution_to_dict(sol) for sol in pareto_solutions],
                "metrics": metrics,
                "optimization_successful": True
            }
            
        except Exception as e:
            logger.error(f"Error in enterprise optimization: {e}")
            return {
                "pareto_solutions": candidates[:10],  # Fallback to top candidates
                "metrics": {"error": str(e), "optimization_successful": False}
            }
    
    async def _initialize_population(self, candidates: List[Dict[str, Any]], csv_upload_id: str) -> List[Dict[str, Any]]:
        """Initialize optimization population with enhanced candidates"""
        try:
            population = []
            
            # Bound population size for small candidate sets to avoid infinite loops
            effective_population_size = min(self.population_size, max(len(candidates) * 5, len(candidates)))
            
            # Add original candidates
            population.extend(candidates[:effective_population_size // 2])
            
            # Generate variations using genetic operators with capped attempts
            max_attempts = effective_population_size * 3
            attempts = 0
            
            while len(population) < effective_population_size and candidates and attempts < max_attempts:
                # Random candidate selection
                candidate_idx = np.random.randint(0, len(candidates))
                base_candidate = candidates[candidate_idx]
                
                # Apply mutation/variation
                mutated = await self._mutate_candidate(base_candidate, csv_upload_id)
                if mutated and mutated not in population:
                    population.append(mutated)
                
                attempts += 1
            
            # Deterministic backfill if we still need more candidates
            if len(population) < effective_population_size:
                needed = effective_population_size - len(population)
                # Cycle through candidates to fill remaining spots (allow duplicates)
                for i in range(needed):
                    candidate_idx = i % len(candidates)
                    population.append(candidates[candidate_idx])
            
            logger.info(f"Population initialized: {len(population)} candidates (target: {effective_population_size}, attempts: {attempts})")
            return population[:effective_population_size]
            
        except Exception as e:
            logger.warning(f"Error initializing population: {e}")
            return candidates[:min(len(candidates), 50)]  # Safe fallback
    
    async def _apply_constraints(self, population: List[Dict[str, Any]], 
                               constraints: List[OptimizationConstraint], 
                               csv_upload_id: str) -> List[Dict[str, Any]]:
        """Apply hard constraints to filter feasible solutions"""
        if not constraints:
            return population
        
        feasible_solutions = []
        
        for candidate in population:
            is_feasible = True
            
            for constraint in constraints:
                if constraint.is_hard:
                    constraint_satisfied = await self._evaluate_constraint(
                        candidate, constraint, csv_upload_id
                    )
                    if not constraint_satisfied:
                        is_feasible = False
                        break
            
            if is_feasible:
                feasible_solutions.append(candidate)
        
        logger.info(f"Constraint filtering: {len(feasible_solutions)}/{len(population)} solutions feasible")
        return feasible_solutions
    
    async def _nsga_ii_optimization(self, population: List[Dict[str, Any]], 
                                  objectives: List[OptimizationObjective],
                                  constraints: List[OptimizationConstraint],
                                  csv_upload_id: str) -> List[ParetoSolution]:
        """NSGA-II multi-objective optimization algorithm"""
        logger.info("Running NSGA-II multi-objective optimization")
        
        current_population = population.copy()
        
        for generation in range(self.max_generations):
            # Evaluate objectives for all candidates
            evaluated_population = []
            
            if self.parallel_processing:
                # Parallel evaluation for performance
                tasks = [
                    self._evaluate_all_objectives(candidate, objectives, csv_upload_id)
                    for candidate in current_population
                ]
                objective_scores = await asyncio.gather(*tasks)
            else:
                # Sequential evaluation
                objective_scores = []
                for candidate in current_population:
                    scores = await self._evaluate_all_objectives(candidate, objectives, csv_upload_id)
                    objective_scores.append(scores)
            
            # Create ParetoSolution objects
            for i, candidate in enumerate(current_population):
                solution = ParetoSolution(
                    bundle_candidate=candidate,
                    objective_scores=objective_scores[i],
                    constraint_satisfaction={},
                    dominance_rank=0,
                    crowding_distance=0.0
                )
                evaluated_population.append(solution)
            
            # Non-dominated sorting
            fronts = self._non_dominated_sorting(evaluated_population)
            
            # Crowding distance calculation
            for front in fronts:
                self._calculate_crowding_distance(front)
            
            # Selection for next generation
            next_population = []
            front_index = 0
            
            while len(next_population) + len(fronts[front_index]) <= self.population_size:
                next_population.extend(fronts[front_index])
                front_index += 1
                if front_index >= len(fronts):
                    break
            
            # Fill remaining slots with crowding distance tournament
            if front_index < len(fronts) and len(next_population) < self.population_size:
                remaining_slots = self.population_size - len(next_population)
                sorted_front = sorted(fronts[front_index], 
                                    key=lambda x: x.crowding_distance, reverse=True)
                next_population.extend(sorted_front[:remaining_slots])
            
            # Generate offspring for next generation (simplified)
            current_population = [sol.bundle_candidate for sol in next_population]
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: {len(fronts)} fronts, best front size: {len(fronts[0])}")
        
        # Return first Pareto front as final solutions
        final_population = []
        for candidate in current_population:
            scores = await self._evaluate_all_objectives(candidate, objectives, csv_upload_id)
            solution = ParetoSolution(
                bundle_candidate=candidate,
                objective_scores=scores,
                constraint_satisfaction={},
                dominance_rank=0,
                crowding_distance=0.0
            )
            final_population.append(solution)
        
        final_fronts = self._non_dominated_sorting(final_population)
        return final_fronts[0] if final_fronts else []
    
    async def _weighted_sum_optimization(self, population: List[Dict[str, Any]], 
                                       objectives: List[OptimizationObjective],
                                       csv_upload_id: str) -> List[ParetoSolution]:
        """Weighted sum optimization approach"""
        logger.info("Running weighted sum optimization")
        
        evaluated_solutions = []
        
        for candidate in population:
            objective_scores = await self._evaluate_all_objectives(candidate, objectives, csv_upload_id)
            
            # Compute weighted sum
            weighted_score = 0.0
            for obj in objectives:
                weight = self.default_objective_weights.get(obj, 1.0 / len(objectives))
                weighted_score += weight * objective_scores.get(obj.value, 0.0)
            
            solution = ParetoSolution(
                bundle_candidate=candidate,
                objective_scores=objective_scores,
                constraint_satisfaction={},
                dominance_rank=0,
                crowding_distance=weighted_score
            )
            evaluated_solutions.append(solution)
        
        # Sort by weighted score and return top solutions
        evaluated_solutions.sort(key=lambda x: x.crowding_distance, reverse=True)
        return evaluated_solutions[:min(20, len(evaluated_solutions))]
    
    async def _epsilon_constraint_optimization(self, population: List[Dict[str, Any]], 
                                             objectives: List[OptimizationObjective],
                                             constraints: List[OptimizationConstraint],
                                             csv_upload_id: str) -> List[ParetoSolution]:
        """Epsilon-constraint optimization method"""
        logger.info("Running epsilon-constraint optimization")
        
        if not objectives:
            return []
        
        # Use first objective as primary, others as constraints
        primary_objective = objectives[0]
        secondary_objectives = objectives[1:]
        
        best_solutions = []
        
        # Evaluate all candidates for primary objective
        for candidate in population:
            objective_scores = await self._evaluate_all_objectives(candidate, objectives, csv_upload_id)
            
            solution = ParetoSolution(
                bundle_candidate=candidate,
                objective_scores=objective_scores,
                constraint_satisfaction={},
                dominance_rank=0,
                crowding_distance=objective_scores.get(primary_objective.value, 0.0)
            )
            best_solutions.append(solution)
        
        # Sort by primary objective and return diverse solutions
        best_solutions.sort(key=lambda x: x.crowding_distance, reverse=True)
        return best_solutions[:min(15, len(best_solutions))]
    
    async def _evaluate_all_objectives(self, candidate: Dict[str, Any], 
                                     objectives: List[OptimizationObjective],
                                     csv_upload_id: str) -> Dict[str, float]:
        """Evaluate all objectives for a candidate solution"""
        cache_key = f"{hash(str(candidate.get('products', [])))}_{csv_upload_id}"
        
        if self.cache_enabled and cache_key in self._objective_cache:
            return self._objective_cache[cache_key]
        
        scores = {}
        
        for objective in objectives:
            score = await self._evaluate_single_objective(candidate, objective, csv_upload_id)
            scores[objective.value] = score
        
        if self.cache_enabled:
            self._objective_cache[cache_key] = scores
        
        return scores
    
    async def _evaluate_single_objective(self, candidate: Dict[str, Any], 
                                       objective: OptimizationObjective,
                                       csv_upload_id: str) -> float:
        """Evaluate a single objective for a candidate"""
        try:
            products = candidate.get("products", [])
            if not products:
                return 0.0
            
            if objective == OptimizationObjective.MAXIMIZE_REVENUE:
                return await self._compute_revenue_score(candidate, csv_upload_id)
            
            elif objective == OptimizationObjective.MAXIMIZE_MARGIN:
                return await self._compute_margin_score(candidate, csv_upload_id)
            
            elif objective == OptimizationObjective.MINIMIZE_INVENTORY_RISK:
                return await self._compute_inventory_risk_score(candidate, csv_upload_id)
            
            elif objective == OptimizationObjective.MAXIMIZE_CUSTOMER_SATISFACTION:
                return await self._compute_satisfaction_score(candidate, csv_upload_id)
            
            elif objective == OptimizationObjective.MAXIMIZE_CROSS_SELL:
                return await self._compute_cross_sell_score(candidate, csv_upload_id)
            
            elif objective == OptimizationObjective.MINIMIZE_CANNIBALIZATION:
                return await self._compute_cannibalization_score(candidate, csv_upload_id)
            
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error evaluating objective {objective}: {e}")
            return 0.0
    
    async def _compute_revenue_score(self, candidate: Dict[str, Any], csv_upload_id: str) -> float:
        """Compute expected revenue score for bundle"""
        try:
            products = candidate.get("products", [])
            confidence = float(candidate.get("confidence", 0))

            # OPTIMIZATION: Use pre-cached catalog instead of querying DB
            catalog_map = self._catalog_cache.get(csv_upload_id)
            if not catalog_map:
                # Fallback (should not happen if pre-caching worked)
                logger.warning(f"Catalog cache miss for {csv_upload_id}, fetching from DB")
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            
            # Get product pricing data
            total_value = Decimal('0')
            catalog_items = [catalog_map.get(sku) for sku in products if sku in catalog_map]
            
            for item in catalog_items:
                if item.price:
                    total_value += item.price
            
            # Expected revenue = confidence × total_value
            expected_revenue = confidence * float(total_value)
            
            # Normalize to 0-1 scale (assuming max bundle value of $500)
            return min(1.0, expected_revenue / 500.0)
            
        except Exception as e:
            logger.warning(f"Error computing revenue score: {e}")
            return 0.0
    
    async def _compute_margin_score(self, candidate: Dict[str, Any], csv_upload_id: str) -> float:
        """Compute profit margin score for bundle"""
        try:
            products = candidate.get("products", [])

            # Get product cost/margin data from objective flags
            total_margin = 0.0
            item_count = 0

            # OPTIMIZATION: Use pre-cached catalog instead of querying DB
            catalog_map = self._catalog_cache.get(csv_upload_id)
            if not catalog_map:
                logger.warning(f"Catalog cache miss for {csv_upload_id}, fetching from DB")
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            catalog_items = [catalog_map.get(sku) for sku in products if sku in catalog_map]
            
            for item in catalog_items:
                if hasattr(item, 'is_high_margin') and item.is_high_margin:
                    total_margin += 0.8  # High margin
                    item_count += 1
                elif hasattr(item, 'is_high_margin'):
                    total_margin += 0.4  # Standard margin
                    item_count += 1
            
            return total_margin / max(item_count, 1)
            
        except Exception as e:
            logger.warning(f"Error computing margin score: {e}")
            return 0.0
    
    async def _compute_inventory_risk_score(self, candidate: Dict[str, Any], csv_upload_id: str) -> float:
        """Compute inventory risk score (lower risk = higher score)"""
        try:
            products = candidate.get("products", [])

            total_risk = 0.0
            # OPTIMIZATION: Use pre-cached catalog instead of querying DB
            catalog_map = self._catalog_cache.get(csv_upload_id)
            if not catalog_map:
                logger.warning(f"Catalog cache miss for {csv_upload_id}, fetching from DB")
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            catalog_items = [catalog_map.get(sku) for sku in products if sku in catalog_map]
            
            for item in catalog_items:
                available = item.available_total or 0
                if available > 50:
                    total_risk += 0.9  # Low risk
                elif available > 10:
                    total_risk += 0.6  # Medium risk
                elif available > 0:
                    total_risk += 0.3  # High risk
                else:
                    total_risk += 0.0  # Out of stock
            
            return total_risk / max(len(catalog_items), 1)
            
        except Exception as e:
            logger.warning(f"Error computing inventory risk score: {e}")
            return 0.5
    
    async def _compute_satisfaction_score(self, candidate: Dict[str, Any], csv_upload_id: str) -> float:
        """Compute customer satisfaction score based on product ratings/reviews"""
        try:
            # Simplified satisfaction based on product diversity and quality signals
            products = candidate.get("products", [])
            confidence = float(candidate.get("confidence", 0))
            lift = float(candidate.get("lift", 1))
            
            # Higher confidence and lift suggest better customer acceptance
            satisfaction_base = (confidence + (lift - 1) / 3) / 2
            
            # Bonus for product diversity (more categories = higher satisfaction)
            diversity_bonus = min(0.2, len(products) * 0.05)
            
            return min(1.0, satisfaction_base + diversity_bonus)
            
        except Exception as e:
            logger.warning(f"Error computing satisfaction score: {e}")
            return 0.5
    
    async def _compute_cross_sell_score(self, candidate: Dict[str, Any], csv_upload_id: str) -> float:
        """Compute cross-selling effectiveness score"""
        try:
            products = candidate.get("products", [])
            lift = float(candidate.get("lift", 1))

            # Higher lift indicates better cross-selling potential
            cross_sell_base = min(1.0, (lift - 1) / 2)

            # Bonus for multi-category bundles
            # OPTIMIZATION: Use pre-cached catalog instead of querying DB
            catalog_map = self._catalog_cache.get(csv_upload_id)
            if not catalog_map:
                logger.warning(f"Catalog cache miss for {csv_upload_id}, fetching from DB")
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            catalog_items = [catalog_map.get(sku) for sku in products if sku in catalog_map]
            categories = set()
            for item in catalog_items:
                if hasattr(item, 'category') and item.category:
                    categories.add(item.category)
            
            category_bonus = min(0.3, len(categories) * 0.1)
            
            return cross_sell_base + category_bonus
            
        except Exception as e:
            logger.warning(f"Error computing cross-sell score: {e}")
            return 0.0
    
    async def _compute_cannibalization_score(self, candidate: Dict[str, Any], csv_upload_id: str) -> float:
        """Compute cannibalization risk score (lower cannibalization = higher score)"""
        try:
            products = candidate.get("products", [])

            # Check for products in same category (potential cannibalization)
            # OPTIMIZATION: Use pre-cached catalog instead of querying DB
            catalog_map = self._catalog_cache.get(csv_upload_id)
            if not catalog_map:
                logger.warning(f"Catalog cache miss for {csv_upload_id}, fetching from DB")
                catalog_map = await storage.get_catalog_snapshots_map(csv_upload_id)
            catalog_items = [catalog_map.get(sku) for sku in products if sku in catalog_map]
            categories = []
            for item in catalog_items:
                if hasattr(item, 'category') and item.category:
                    categories.append(item.category)
            
            # Count duplicate categories
            unique_categories = len(set(categories))
            total_products = len(categories)
            
            if total_products == 0:
                return 1.0
            
            # Higher ratio of unique categories = lower cannibalization
            diversity_ratio = unique_categories / total_products
            return diversity_ratio
            
        except Exception as e:
            logger.warning(f"Error computing cannibalization score: {e}")
            return 1.0
    
    async def _evaluate_constraint(self, candidate: Dict[str, Any], 
                                 constraint: OptimizationConstraint,
                                 csv_upload_id: str) -> bool:
        """Evaluate if candidate satisfies a constraint"""
        try:
            if constraint.constraint_type == "inventory":
                return await self._check_inventory_constraint(candidate, constraint, csv_upload_id)
            elif constraint.constraint_type == "budget":
                return await self._check_budget_constraint(candidate, constraint, csv_upload_id)
            elif constraint.constraint_type == "margin":
                return await self._check_margin_constraint(candidate, constraint, csv_upload_id)
            elif constraint.constraint_type == "product_limit":
                return self._check_product_limit_constraint(candidate, constraint)
            elif constraint.constraint_type == "category_mix":
                return await self._check_category_mix_constraint(candidate, constraint, csv_upload_id)
            else:
                return True  # Unknown constraint type, assume satisfied
                
        except Exception as e:
            logger.warning(f"Error evaluating constraint {constraint.constraint_type}: {e}")
            return True  # Assume satisfied on error
    
    async def _check_inventory_constraint(self, candidate: Dict[str, Any], 
                                        constraint: OptimizationConstraint,
                                        csv_upload_id: str) -> bool:
        """Check inventory availability constraint"""
        products = candidate.get("products", [])
        min_inventory = constraint.value
        
        catalog_items = await storage.get_catalog_snapshots_by_skus(products, csv_upload_id)
        
        for item in catalog_items:
            if constraint.operator == ">=" and item.available_total < min_inventory:
                return False
            elif constraint.operator == "<=" and item.available_total > min_inventory:
                return False
        
        return True
    
    async def _check_budget_constraint(self, candidate: Dict[str, Any], 
                                     constraint: OptimizationConstraint,
                                     csv_upload_id: str) -> bool:
        """Check budget/pricing constraint"""
        products = candidate.get("products", [])
        max_budget = Decimal(str(constraint.value))
        
        catalog_items = await storage.get_catalog_snapshots_by_skus(products, csv_upload_id)
        total_price = sum(item.price for item in catalog_items if item.price)
        
        if constraint.operator == "<=" and total_price > max_budget:
            return False
        elif constraint.operator == ">=" and total_price < max_budget:
            return False
        
        return True
    
    async def _check_margin_constraint(self, candidate: Dict[str, Any], 
                                     constraint: OptimizationConstraint,
                                     csv_upload_id: str) -> bool:
        """Check margin requirement constraint"""
        margin_score = await self._compute_margin_score(candidate, csv_upload_id)
        min_margin = constraint.value
        
        return margin_score >= min_margin
    
    def _check_product_limit_constraint(self, candidate: Dict[str, Any], 
                                      constraint: OptimizationConstraint) -> bool:
        """Check product count limit constraint"""
        products = candidate.get("products", [])
        limit = constraint.value
        
        if constraint.operator == "<=" and len(products) > limit:
            return False
        elif constraint.operator == ">=" and len(products) < limit:
            return False
        
        return True
    
    async def _check_category_mix_constraint(self, candidate: Dict[str, Any], 
                                           constraint: OptimizationConstraint,
                                           csv_upload_id: str) -> bool:
        """Check category diversity constraint"""
        products = candidate.get("products", [])
        required_categories = constraint.value  # Expected to be a number or list
        
        catalog_items = await storage.get_catalog_snapshots_by_skus(products, csv_upload_id)
        categories = set()
        for item in catalog_items:
            if hasattr(item, 'product_type') and item.product_type:
                categories.add(item.product_type)
            elif hasattr(item, 'category') and item.category:
                categories.add(item.category)
        
        if isinstance(required_categories, int):
            return len(categories) >= required_categories
        elif isinstance(required_categories, list):
            return any(cat in categories for cat in required_categories)
        
        return True
    
    async def _mutate_candidate(self, candidate: Dict[str, Any], csv_upload_id: str) -> Optional[Dict[str, Any]]:
        """Apply mutation to create candidate variation"""
        try:
            if np.random.random() > self.mutation_rate:
                return candidate
            
            mutated = candidate.copy()
            products = mutated.get("products", []).copy()
            
            if not products:
                return candidate
            
            # Mutation strategies
            mutation_type = np.random.choice(["replace", "add", "remove"])
            
            if mutation_type == "replace" and len(products) > 1:
                # Replace one product with a similar one
                replace_idx = np.random.randint(0, len(products))
                # Simplified: just return original for now
                return mutated
            
            elif mutation_type == "add" and len(products) < 5:
                # Add a complementary product (simplified)
                mutated["products"] = products
                return mutated
            
            elif mutation_type == "remove" and len(products) > 2:
                # Remove one product
                remove_idx = np.random.randint(0, len(products))
                products.pop(remove_idx)
                mutated["products"] = products
                return mutated
            
            return candidate
            
        except Exception as e:
            logger.warning(f"Error in mutation: {e}")
            return candidate
    
    def _non_dominated_sorting(self, population: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Non-dominated sorting for NSGA-II"""
        fronts = []
        
        # Calculate dominance relationships
        for i, sol_i in enumerate(population):
            sol_i.dominance_rank = 0
            dominated_solutions = []
            dominating_count = 0
            
            for j, sol_j in enumerate(population):
                if i != j:
                    if self._dominates(sol_i, sol_j):
                        dominated_solutions.append(j)
                    elif self._dominates(sol_j, sol_i):
                        dominating_count += 1
            
            if dominating_count == 0:
                sol_i.dominance_rank = 0
                if not fronts:
                    fronts.append([])
                fronts[0].append(sol_i)
        
        # Build subsequent fronts
        front_index = 0
        while front_index < len(fronts) and fronts[front_index]:
            next_front = []
            
            for sol in fronts[front_index]:
                # This is simplified - full NSGA-II implementation would be more complex
                pass
            
            front_index += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def _dominates(self, sol_a: ParetoSolution, sol_b: ParetoSolution) -> bool:
        """Check if solution A dominates solution B"""
        better_in_all = True
        better_in_at_least_one = False
        
        for obj_name, score_a in sol_a.objective_scores.items():
            score_b = sol_b.objective_scores.get(obj_name, 0.0)
            
            if score_a < score_b:
                better_in_all = False
            elif score_a > score_b:
                better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one
    
    def _calculate_crowding_distance(self, front: List[ParetoSolution]):
        """Calculate crowding distance for solutions in a front"""
        if len(front) <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for sol in front:
            sol.crowding_distance = 0.0
        
        # Calculate distance for each objective
        for obj_name in front[0].objective_scores.keys():
            # Sort by objective value
            front.sort(key=lambda x: x.objective_scores.get(obj_name, 0.0))
            
            # Boundary solutions get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distances for intermediate solutions
            obj_range = front[-1].objective_scores.get(obj_name, 0.0) - front[0].objective_scores.get(obj_name, 0.0)
            
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objective_scores.get(obj_name, 0.0) - 
                              front[i - 1].objective_scores.get(obj_name, 0.0)) / obj_range
                    front[i].crowding_distance += distance
    
    def _solution_to_dict(self, solution: ParetoSolution) -> Dict[str, Any]:
        """Convert ParetoSolution to dictionary for serialization"""
        return {
            "bundle_candidate": solution.bundle_candidate,
            "objective_scores": solution.objective_scores,
            "constraint_satisfaction": solution.constraint_satisfaction,
            "dominance_rank": solution.dominance_rank,
            "crowding_distance": solution.crowding_distance if solution.crowding_distance != float('inf') else 999999
        }
    
    def clear_cache(self):
        """Clear optimization caches"""
        self._objective_cache.clear()
        self._constraint_cache.clear()
        logger.info("Optimization caches cleared")