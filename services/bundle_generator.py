"""
Bundle Generator Service v2
Creates bundle recommendations using comprehensive v2 pipeline with enhanced features
"""
from typing import List, Dict, Any, Optional
import logging
import uuid
from decimal import Decimal
from datetime import datetime, timedelta
import random
import time

from services.storage import storage
from services.ai_copy_generator import AICopyGenerator
from services.data_mapper import DataMapper
from services.objectives import ObjectiveScorer
from services.ml.candidate_generator import CandidateGenerator
from services.pricing import BayesianPricingEngine
from services.ranker import WeightedLinearRanker
from services.deduplication import DeduplicationService
from services.explainability import ExplainabilityEngine
from services.ml.optimization_engine import EnterpriseOptimizationEngine, OptimizationObjective
from services.ml.constraint_manager import EnterpriseConstraintManager
from services.ml.performance_monitor import EnterprisePerformanceMonitor
from services.ml.fallback_ladder import FallbackLadder

# Import observability and feature flag systems (PR-8)
from services.obs.metrics import metrics_collector
from services.feature_flags import feature_flags

logger = logging.getLogger(__name__)

class BundleGenerator:
    """Bundle generator service creating recommendations from association rules"""
    
    def __init__(self):
        # Initialize v1 services
        self.ai_generator = AICopyGenerator()
        
        # Initialize v2 enhanced services
        self.data_mapper = DataMapper()
        self.objective_scorer = ObjectiveScorer()
        self.candidate_generator = CandidateGenerator()
        self.pricing_engine = BayesianPricingEngine()
        self.ranker = WeightedLinearRanker()
        self.deduplicator = DeduplicationService()
        self.explainer = ExplainabilityEngine()
        
        # Initialize enterprise optimization components (PR-4)
        self.optimization_engine = EnterpriseOptimizationEngine()
        self.constraint_manager = EnterpriseConstraintManager()
        self.performance_monitor = EnterprisePerformanceMonitor()
        
        # Initialize small shop fallback system
        self.fallback_ladder = FallbackLadder(storage)
        
        # Bundle configuration
        self.bundle_types = ['FBT', 'VOLUME_DISCOUNT', 'MIX_MATCH', 'BXGY', 'FIXED']
        
        # 8 Objective types for enhanced bundle generation
        self.objectives = {
            'increase_aov': {'priority': 1.0, 'description': 'Increase Average Order Value'},
            'clear_slow_movers': {'priority': 1.2, 'description': 'Clear Slow-Moving Inventory'},
            'seasonal_promo': {'priority': 0.9, 'description': 'Seasonal Promotion'},
            'new_launch': {'priority': 1.1, 'description': 'Promote New Product Launch'},
            'category_bundle': {'priority': 0.8, 'description': 'Cross-Category Bundle'},
            'gift_box': {'priority': 0.7, 'description': 'Gift Box Bundle'},
            'subscription_push': {'priority': 1.0, 'description': 'Subscription Promotion'},
            'margin_guard': {'priority': 1.3, 'description': 'Maintain High Margins'}
        }
        
        # Bundle generation thresholds
        self.min_confidence = 0.3
        self.min_lift = 1.2
        self.min_support = 0.05
        
        # v2 feature flags
        self.enable_v2_pipeline = True  # Enable comprehensive v2 features
        self.enable_data_mapping = True
        self.enable_objective_scoring = True
        self.enable_ml_candidates = True
        self.enable_bayesian_pricing = True
        self.enable_weighted_ranking = True
        self.enable_deduplication = True
        self.enable_explainability = True
        
        # Enterprise optimization feature flags (PR-4)
        self.enable_enterprise_optimization = True
        self.enable_constraint_management = True
        self.enable_performance_monitoring = True
        self.enable_pareto_optimization = True
        
        # Advanced feature flags (PR-5, PR-6, PR-8)
        self.enable_normalized_ranking = True
        self.enable_cold_start_coverage = True
        self.enable_observability = True
        
        # Loop prevention and performance safeguards
        self.max_total_attempts = 500  # Hard cap on total attempts across all objectives/types
        self.max_time_budget_seconds = 300  # 5 minutes maximum processing time
        self.max_attempts_per_objective_type = 50  # Max attempts per objective/bundle_type combo
        self.seen_sku_combinations = set()  # Track already processed SKU combinations
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'skipped_duplicates': 0,
            'failed_attempts': 0,
            'timeout_exits': 0,
            'early_exits': 0
        }
        
        # ARCHITECT FIX: Circuit-breaker pattern to stop runaway behavior
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # Stop after 10 consecutive failures
        self.circuit_breaker_active = False

    def _check_circuit_breaker(self, operation_name: str, success: bool) -> bool:
        """ARCHITECT FIX: Circuit-breaker to detect consecutive failures and stop runaway behavior"""
        if success:
            self.consecutive_failures = 0
            self.circuit_breaker_active = False
            return False  # Continue operation
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                if not self.circuit_breaker_active:
                    logger.error(f"CIRCUIT BREAKER ACTIVATED: {self.consecutive_failures} consecutive failures in {operation_name}")
                    self.circuit_breaker_active = True
                return True  # Stop operation
            else:
                logger.warning(f"Consecutive failure #{self.consecutive_failures} in {operation_name}")
                return False  # Continue operation
    
    def _safe_decimal(self, value, default=None):
        """Safely convert value to Decimal for database storage"""
        from decimal import Decimal, InvalidOperation
        
        if value is None:
            return default
        
        try:
            # Handle string inputs
            if isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in ('null', 'none', ''):
                    return default
            
            # Convert to Decimal
            decimal_value = Decimal(str(value))
            
            # Validate reasonable bounds for confidence, lift, etc.
            if decimal_value < 0:
                return default if default is not None else Decimal('0')
            if decimal_value > 1000:  # Reasonable upper bound
                return Decimal('1000')
            
            return decimal_value
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(f"Could not convert {value} to Decimal, using default: {default}")
            return default if default is not None else Decimal('0')
    
    def _serialize_pricing_for_json(self, pricing_data):
        """Convert pricing data with Decimal values to JSON-safe format"""
        if not pricing_data:
            return {}
        
        serialized = {}
        for key, value in pricing_data.items():
            try:
                if isinstance(value, Decimal):
                    serialized[key] = float(value)
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    serialized[key] = self._serialize_pricing_for_json(value)
                elif isinstance(value, list):
                    # Handle lists that might contain Decimals
                    serialized[key] = [
                        float(item) if isinstance(item, Decimal) else item 
                        for item in value
                    ]
                else:
                    serialized[key] = value
            except (TypeError, ValueError) as e:
                logger.warning(f"Error serializing pricing field {key}: {e}")
                serialized[key] = str(value) if value is not None else None
        
        return serialized
    
    async def generate_bundle_recommendations(self, csv_upload_id: str) -> Dict[str, Any]:
        """Generate bundle recommendations using comprehensive v2 pipeline"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        logger.info(f"Starting v2 bundle generation for upload: {csv_upload_id}")
        
        # Initialize loop prevention tracking
        self.seen_sku_combinations.clear()
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'skipped_duplicates': 0,
            'failed_attempts': 0,
            'timeout_exits': 0,
            'early_exits': 0
        }
        
        # Initialize comprehensive metrics
        metrics = {
            "v2_pipeline_enabled": self.enable_v2_pipeline,
            "data_mapping": {"enabled": False, "metrics": {}},
            "objective_scoring": {"enabled": False, "metrics": {}},
            "ml_candidates": {"enabled": False, "metrics": {}},
            "bayesian_pricing": {"enabled": False, "metrics": {}},
            "weighted_ranking": {"enabled": False, "metrics": {}},
            "deduplication": {"enabled": False, "metrics": {}},
            "explainability": {"enabled": False, "metrics": {}},
            "bundle_counts": {"FBT": 0, "VOLUME_DISCOUNT": 0, "MIX_MATCH": 0, "BXGY": 0, "FIXED": 0},
            "total_recommendations": 0,
            "processing_time_ms": 0,
            "loop_prevention_stats": self.generation_stats
        }
        
        start_time = datetime.now()
        
        # Set hard timeout
        end_time = start_time + timedelta(seconds=self.max_time_budget_seconds)
        
        try:
            # Phase 1: Data Mapping and Enrichment
            if self.enable_data_mapping and csv_upload_id:
                logger.info("Phase 1: Data mapping and enrichment")
                data_mapping_result = await self.data_mapper.enrich_order_lines_with_variants(csv_upload_id)
                metrics["data_mapping"] = {"enabled": True, "metrics": data_mapping_result.get("metrics", {})}
            
            # Phase 2: Objective Scoring
            if self.enable_objective_scoring and csv_upload_id:
                logger.info("Phase 2: Objective scoring")
                objective_result = await self.objective_scorer.compute_objective_flags(csv_upload_id)
                metrics["objective_scoring"] = {"enabled": True, "metrics": objective_result.get("metrics", {})}
            
            # Phase 3: Generate candidates for each objective with loop prevention
            all_recommendations = []
            objectives_processed = 0
            
            for objective_name, objective_config in self.objectives.items():
                # Check time budget
                if datetime.now() >= end_time:
                    logger.warning(f"Time budget exceeded, stopping at objective: {objective_name}")
                    self.generation_stats['timeout_exits'] += 1
                    break
                
                # Check attempt limits
                if self.generation_stats['total_attempts'] >= self.max_total_attempts:
                    logger.warning(f"Max attempts ({self.max_total_attempts}) reached, stopping at objective: {objective_name}")
                    self.generation_stats['early_exits'] += 1
                    break
                
                # Check attempt vs success ratio for early exit
                if (self.generation_stats['total_attempts'] > 100 and 
                    self.generation_stats['successful_generations'] > 0):
                    success_rate = self.generation_stats['successful_generations'] / self.generation_stats['total_attempts']
                    if success_rate < 0.05:  # Less than 5% success rate
                        logger.warning(f"Low success rate ({success_rate:.2%}), early exit at objective: {objective_name}")
                        self.generation_stats['early_exits'] += 1
                        break
                
                logger.info(f"Generating bundles for objective: {objective_name} (attempts: {self.generation_stats['total_attempts']}, successes: {self.generation_stats['successful_generations']})")
                
                for bundle_type in self.bundle_types:
                    # Check time budget again
                    if datetime.now() >= end_time:
                        logger.warning(f"Time budget exceeded, stopping at bundle_type: {bundle_type}")
                        self.generation_stats['timeout_exits'] += 1
                        break
                    
                    objective_recommendations = await self.generate_objective_bundles(
                        csv_upload_id, objective_name, bundle_type, metrics, end_time
                    )
                    all_recommendations.extend(objective_recommendations)
                    
                    # Log progress every few iterations
                    if (objectives_processed * len(self.bundle_types) + self.bundle_types.index(bundle_type)) % 5 == 0:
                        logger.info(f"Progress: {len(all_recommendations)} recommendations, {self.generation_stats['total_attempts']} attempts")
                
                objectives_processed += 1
                
                # Persist partial results periodically
                if len(all_recommendations) >= 20 and len(all_recommendations) % 20 == 0:
                    logger.info(f"Persisting {len(all_recommendations)} partial results")
                    try:
                        await self.store_partial_recommendations(all_recommendations, csv_upload_id)
                    except Exception as e:
                        logger.warning(f"Failed to persist partial results: {e}")
            
            # Phase 4: Deduplication
            if self.enable_deduplication and all_recommendations:
                logger.info("Phase 4: Deduplication")
                dedupe_result = await self.deduplicator.deduplicate_candidates(all_recommendations, csv_upload_id)
                all_recommendations = dedupe_result.get("unique_candidates", all_recommendations)
                metrics["deduplication"] = {"enabled": True, "metrics": dedupe_result.get("metrics", {})}
            
            # Phase 5: Global Enterprise Optimization (across all bundle types)
            if self.enable_enterprise_optimization and all_recommendations and len(all_recommendations) > 10:
                logger.info("Phase 5: Global enterprise optimization across all bundle types")
                optimization_start_time = time.time()
                
                try:
                    # Run global optimization for portfolio-level optimization
                    global_constraints = await self.constraint_manager.get_constraints_for_objective(
                        "increase_aov", csv_upload_id  # Use default objective for global optimization
                    ) if self.enable_constraint_management else []
                    
                    global_optimization_result = await self.optimization_engine.optimize_bundle_portfolio(
                        all_recommendations,
                        [OptimizationObjective.MAXIMIZE_REVENUE, OptimizationObjective.MAXIMIZE_MARGIN],
                        global_constraints,
                        csv_upload_id,
                        "pareto" if self.enable_pareto_optimization else "weighted_sum"
                    )
                    
                    if global_optimization_result.get("optimization_successful", False):
                        pareto_solutions = global_optimization_result.get("pareto_solutions", [])
                        if pareto_solutions:
                            all_recommendations = pareto_solutions
                            logger.info(f"Global optimization: {len(all_recommendations)} portfolio-optimal solutions")
                    
                    metrics["global_enterprise_optimization"] = {
                        "enabled": True,
                        "input_recommendations": len(all_recommendations),
                        "pareto_solutions": len(all_recommendations),
                        "processing_time": (time.time() - optimization_start_time) * 1000,
                        "global_optimization_metrics": global_optimization_result.get("metrics", {})
                    }
                    
                except Exception as e:
                    logger.warning(f"Global enterprise optimization failed: {e}")
                    metrics["global_enterprise_optimization"] = {"enabled": True, "error": str(e)}
            
            # Phase 5b: Weighted Ranking (fallback or when enterprise optimization disabled)
            elif self.enable_weighted_ranking and all_recommendations:
                logger.info("Phase 5b: Weighted ranking (fallback)")
                # Rank all recommendations together for global optimization
                ranked_recommendations = await self.ranker.rank_bundle_recommendations(
                    all_recommendations, "increase_aov", csv_upload_id  # Use default objective for global ranking
                )
                all_recommendations = ranked_recommendations
                metrics["weighted_ranking"] = {"enabled": True}
            
            else:
                metrics["global_enterprise_optimization"] = {"enabled": False}
                metrics["weighted_ranking"] = {"enabled": False}
            
            # Phase 6: Explainability
            if self.enable_explainability and all_recommendations:
                logger.info("Phase 6: Adding explanations")
                for recommendation in all_recommendations:
                    explanation = self.explainer.generate_explanation(recommendation)
                    recommendation["explanation"] = explanation
                    
                    if self.enable_explainability:
                        detailed_explanation = self.explainer.generate_detailed_explanation(recommendation)
                        recommendation["detailed_explanation"] = detailed_explanation
                
                metrics["explainability"] = {"enabled": True}
            
            # Phase 7: Finalization and storage
            final_recommendations = await self.finalize_recommendations(all_recommendations, csv_upload_id, metrics)
            
            # Update metrics
            metrics["total_recommendations"] = len(final_recommendations)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            metrics["processing_time_ms"] = int(processing_time)
            
            # Add loop prevention statistics to metrics
            metrics["loop_prevention_stats"] = self.generation_stats.copy()
            
            # Log final statistics
            logger.info(f"Bundle generation completed. Stats: {self.generation_stats}")
            logger.info(f"Processed {len(self.seen_sku_combinations)} unique SKU combinations")
            
            # Count by bundle type
            for rec in final_recommendations:
                bundle_type = rec.get("bundle_type", "UNKNOWN")
                if bundle_type in metrics["bundle_counts"]:
                    metrics["bundle_counts"][bundle_type] += 1
            
            logger.info(f"v2 Bundle generation completed: {len(final_recommendations)} recommendations in {processing_time:.1f}ms. Stats: {self.generation_stats}")
            
            return {
                "recommendations": final_recommendations,
                "metrics": metrics,
                "v2_pipeline": True,
                "csv_upload_id": csv_upload_id
            }
            
        except Exception as e:
            logger.error(f"Error in v2 bundle generation: {e}")
            # Fallback to v1 pipeline if v2 fails
            if self.enable_v2_pipeline:
                logger.info("Falling back to v1 pipeline due to v2 error")
                return await self.generate_v1_recommendations(csv_upload_id, metrics, error=str(e))
            else:
                raise
    
    async def generate_objective_bundles(self, csv_upload_id: str, objective: str, bundle_type: str, metrics: Dict[str, Any], end_time: datetime = None) -> List[Dict[str, Any]]:
        """Generate bundles for a specific objective and bundle type with loop prevention"""
        objective_type_key = f"{objective}_{bundle_type}"
        attempts_for_this_combo = 0
        
        try:
            # Check if we've exceeded attempts for this specific objective/type combo
            if attempts_for_this_combo >= self.max_attempts_per_objective_type:
                logger.warning(f"Max attempts for {objective_type_key} reached, skipping")
                return []
            
            # Check time budget
            if end_time and datetime.now() >= end_time:
                logger.warning(f"Time budget exceeded in {objective_type_key}")
                self.generation_stats['timeout_exits'] += 1
                return []
            
            recommendations = []
            
            # Phase 3a: ML Candidate Generation with attempt tracking
            self.generation_stats['total_attempts'] += 1
            attempts_for_this_combo += 1
            
            if self.enable_ml_candidates:
                try:
                    candidate_result = await self.candidate_generator.generate_candidates(csv_upload_id, bundle_type, objective)
                    candidates = candidate_result.get("candidates", [])
                    metrics["ml_candidates"] = {"enabled": True, "metrics": candidate_result.get("metrics", {})}
                    
                    if candidates:
                        self.generation_stats['successful_generations'] += 1
                    else:
                        self.generation_stats['failed_attempts'] += 1
                        logger.debug(f"No ML candidates generated for {objective_type_key}")
                        
                except Exception as e:
                    logger.warning(f"ML candidate generation failed for {objective_type_key}: {e}")
                    candidates = []
                    self.generation_stats['failed_attempts'] += 1
            else:
                candidates = []
                logger.debug(f"ML candidates disabled for {objective_type_key}")
            
            # Phase 3a.1: FallbackLadder for Small Shops (when insufficient candidates)
            min_candidates_threshold = 5  # Minimum candidates for adequate recommendations
            if len(candidates) < min_candidates_threshold:
                logger.info(f"Insufficient candidates ({len(candidates)}) for {objective}/{bundle_type}, activating FallbackLadder")
                try:
                    fallback_candidates = await self.fallback_ladder.generate_candidates(
                        csv_upload_id=csv_upload_id,
                        objective=objective,
                        bundle_type=bundle_type,
                        target_n=10
                    )
                    
                    # Convert FallbackCandidate objects to regular dict format
                    for fb_candidate in fallback_candidates:
                        fallback_dict = {
                            "products": fb_candidate.products,
                            "confidence": fb_candidate.features.get("confidence", 0.5),
                            "lift": fb_candidate.features.get("lift", 1.2),
                            "support": fb_candidate.features.get("support", 0.1),
                            "generation_sources": [fb_candidate.source_tier],
                            "generation_method": "fallback_ladder",
                            "tier_weight": fb_candidate.features.get("tier_weight", 0.5),
                            "explanation": fb_candidate.explanation
                        }
                        candidates.append(fallback_dict)
                    
                    logger.info(f"FallbackLadder generated {len(fallback_candidates)} additional candidates")
                    metrics["fallback_ladder"] = {
                        "activated": True,
                        "original_candidates": len(candidates) - len(fallback_candidates),
                        "fallback_candidates": len(fallback_candidates),
                        "total_candidates": len(candidates)
                    }
                except Exception as e:
                    logger.warning(f"FallbackLadder failed for {objective}/{bundle_type}: {e}")
                    metrics["fallback_ladder"] = {"activated": True, "error": str(e)}
            else:
                metrics["fallback_ladder"] = {"activated": False, "reason": "sufficient_candidates"}
            
            # Convert candidates to recommendations format with duplicate checking
            for candidate in candidates:
                # Check for duplicate SKU combinations
                product_set = frozenset(candidate.get("products", []))
                if len(product_set) < 2:  # Skip single-product or empty bundles
                    continue
                    
                sku_combo_key = f"{objective_type_key}:{hash(product_set)}"
                if sku_combo_key in self.seen_sku_combinations:
                    self.generation_stats['skipped_duplicates'] += 1
                    logger.debug(f"Skipping duplicate SKU combination for {objective_type_key}")
                    continue
                
                # ARCHITECT FIX: Count attempt immediately, mark as seen AFTER successful pricing
                self.generation_stats['total_attempts'] += 1
                
                recommendation = {
                    "id": str(uuid.uuid4()),
                    "csv_upload_id": csv_upload_id,
                    "bundle_type": bundle_type,
                    "objective": objective,
                    "products": candidate.get("products", []),
                    "confidence": candidate.get("confidence", 0),
                    "lift": candidate.get("lift", 1),
                    "support": candidate.get("support", 0),
                    "generation_sources": candidate.get("generation_sources", []),
                    "generation_method": candidate.get("generation_method", "unknown"),
                    "sku_combo_key": sku_combo_key
                }
                
                # CRITICAL FIX: Validate SKUs before pricing to prevent infinite loop
                if self.enable_bayesian_pricing and recommendation["products"]:
                    # Filter out invalid SKUs before pricing
                    valid_products = []
                    invalid_skus = []
                    
                    for sku in recommendation["products"]:
                        if (sku and 
                            not sku.startswith("gid://") and 
                            not sku.startswith("no-sku-") and 
                            not sku.startswith("null") and
                            sku.strip() != ""):
                            valid_products.append(sku)
                        else:
                            invalid_skus.append(sku)
                    
                    if invalid_skus:
                        logger.warning(f"Filtered invalid SKUs from pricing: {invalid_skus}")
                    
                    # Only proceed with pricing if we have valid SKUs
                    if len(valid_products) >= 2:  # Need at least 2 products for bundle
                        recommendation["products"] = valid_products  # Update with only valid SKUs
                        try:
                            pricing_result = await self.pricing_engine.compute_bundle_pricing(
                                valid_products, objective, csv_upload_id, bundle_type
                            )
                            
                            # Check if pricing actually succeeded
                            if pricing_result.get("success", False):
                                recommendation["pricing"] = pricing_result.get("pricing", {})
                                metrics["bayesian_pricing"] = {"enabled": True}
                            else:
                                # Pricing failed, set fallback pricing to prevent retry loop
                                logger.warning(f"Pricing failed for {objective_type_key}, using fallback")
                                recommendation["pricing"] = {
                                    "original_total": 0,
                                    "bundle_price": 0, 
                                    "discount_amount": 0,
                                    "fallback_used": True
                                }
                                
                        except Exception as e:
                            logger.warning(f"Pricing failed for {objective_type_key}: {e}")
                            # Use fallback pricing to prevent infinite retry
                            recommendation["pricing"] = {
                                "error": str(e),
                                "original_total": 0,
                                "bundle_price": 0,
                                "discount_amount": 0,
                                "fallback_used": True
                            }
                        
                        # ARCHITECT FIX: Mark as seen ONLY after processing (success or failure)
                        self.seen_sku_combinations.add(sku_combo_key)
                        
                    else:
                        # Skip this recommendation if insufficient valid SKUs
                        logger.warning(f"Insufficient valid SKUs for bundle, skipping: valid={len(valid_products)}, invalid={len(invalid_skus)}")
                        self.generation_stats['failed_attempts'] += 1
                        # Still mark as seen to prevent retry
                        self.seen_sku_combinations.add(sku_combo_key)
                        continue
                
                # ARCHITECT FIX: Mark as seen if not marked above (non-pricing path)
                if sku_combo_key not in self.seen_sku_combinations:
                    self.seen_sku_combinations.add(sku_combo_key)
                
                recommendations.append(recommendation)
                
                # Check time budget periodically
                if end_time and datetime.now() >= end_time:
                    logger.warning(f"Time budget exceeded during candidate processing for {objective_type_key}")
                    break
            
            # Phase 4: Enterprise Optimization (PR-4)
            if self.enable_enterprise_optimization and recommendations:
                logger.info(f"Phase 4: Enterprise optimization for {objective}/{bundle_type}")
                opt_start_time = time.time()
                
                try:
                    # Start performance monitoring
                    operation_id = f"opt_{objective}_{bundle_type}_{uuid.uuid4().hex[:6]}"
                    if self.enable_performance_monitoring:
                        await self.performance_monitor.start_operation_monitoring(
                            operation_id, "optimization", csv_upload_id, len(recommendations)
                        )
                    
                    # Generate constraints for this objective
                    constraints = []
                    if self.enable_constraint_management:
                        constraints = await self.constraint_manager.get_constraints_for_objective(
                            objective, csv_upload_id
                        )
                    
                    # Map business objectives to optimization objectives
                    objective_mapping = {
                        "increase_aov": [OptimizationObjective.MAXIMIZE_REVENUE, OptimizationObjective.MAXIMIZE_CROSS_SELL],
                        "clear_slow_movers": [OptimizationObjective.MINIMIZE_INVENTORY_RISK, OptimizationObjective.MAXIMIZE_MARGIN],
                        "margin_guard": [OptimizationObjective.MAXIMIZE_MARGIN, OptimizationObjective.MINIMIZE_CANNIBALIZATION],
                        "seasonal_promo": [OptimizationObjective.MAXIMIZE_CUSTOMER_SATISFACTION, OptimizationObjective.MAXIMIZE_CROSS_SELL],
                        "new_launch": [OptimizationObjective.MAXIMIZE_CUSTOMER_SATISFACTION, OptimizationObjective.MAXIMIZE_REVENUE]
                    }
                    
                    optimization_objectives = objective_mapping.get(objective, [
                        OptimizationObjective.MAXIMIZE_REVENUE, OptimizationObjective.MAXIMIZE_MARGIN
                    ])
                    
                    # Run enterprise optimization
                    optimization_method = "pareto" if self.enable_pareto_optimization else "weighted_sum"
                    
                    optimization_result = await self.optimization_engine.optimize_bundle_portfolio(
                        recommendations,
                        optimization_objectives,
                        constraints,
                        csv_upload_id,
                        optimization_method
                    )
                    
                    # Replace recommendations with optimized results
                    if optimization_result.get("optimization_successful", False):
                        pareto_solutions = optimization_result.get("pareto_solutions", [])
                        if pareto_solutions:
                            recommendations = pareto_solutions
                            logger.info(f"Optimization successful: {len(recommendations)} Pareto-optimal solutions for {objective}/{bundle_type}")
                        
                        # Finish performance monitoring with metrics
                        if self.enable_performance_monitoring:
                            ml_metrics = {
                                "pareto_solutions": len(recommendations),
                                "optimization_successful": True,
                                "constraints_applied": len(constraints),
                                "objective": objective,
                                "bundle_type": bundle_type
                            }
                            await self.performance_monitor.finish_operation_monitoring(
                                operation_id, len(recommendations), True, None, ml_metrics
                            )
                    
                    # Update metrics with optimization details
                    opt_metrics = optimization_result.get("metrics", {})
                    opt_metrics.update({
                        "processing_time": (time.time() - opt_start_time) * 1000,
                        "optimization_method": optimization_method,
                        "objectives": [obj.value for obj in optimization_objectives],
                        "constraints_count": len(constraints)
                    })
                    metrics[f"enterprise_optimization_{bundle_type}"] = opt_metrics
                    
                except Exception as e:
                    logger.warning(f"Enterprise optimization failed for {objective}/{bundle_type}: {e}")
                    # Continue with original recommendations on failure
                    metrics[f"enterprise_optimization_{bundle_type}"] = {"enabled": True, "error": str(e)}
                    
                    if self.enable_performance_monitoring:
                        await self.performance_monitor.finish_operation_monitoring(
                            operation_id, len(recommendations), False, str(e)
                        )
            
            # Log stats for this objective/bundle type
            logger.info(f"Completed {objective_type_key}: {len(recommendations)} recommendations, {attempts_for_this_combo} attempts")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating objective bundles for {objective_type_key}: {e}")
            self.generation_stats['failed_attempts'] += 1
            return []
    
    async def finalize_recommendations(self, recommendations: List[Dict[str, Any]], csv_upload_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Finalize recommendations and prepare for storage"""
        try:
            final_recommendations = []
            
            # Limit recommendations per objective/type to avoid overwhelming merchants
            max_per_type = 20
            recommendations_by_type = {}
            
            for rec in recommendations:
                key = f"{rec.get('objective', 'unknown')}_{rec.get('bundle_type', 'unknown')}"
                if key not in recommendations_by_type:
                    recommendations_by_type[key] = []
                
                if len(recommendations_by_type[key]) < max_per_type:
                    recommendations_by_type[key].append(rec)
            
            # Flatten back to single list
            for type_recs in recommendations_by_type.values():
                final_recommendations.extend(type_recs)
            
            # Add AI-generated copy if available
            if final_recommendations:
                for rec in final_recommendations[:10]:  # Limit AI copy generation for cost
                    try:
                        ai_copy = await self.ai_generator.generate_bundle_copy(rec)
                        rec["ai_copy"] = ai_copy
                    except Exception as e:
                        logger.warning(f"Error generating AI copy: {e}")
                        rec["ai_copy"] = {"title": "Bundle Deal", "description": "Great products bundled together"}
            
            # Store recommendations in database
            if final_recommendations and csv_upload_id:
                await self.store_recommendations(final_recommendations, csv_upload_id)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error finalizing recommendations: {e}")
            return recommendations  # Return original recommendations if finalization fails
    
    async def store_partial_recommendations(self, recommendations: List[Dict[str, Any]], csv_upload_id: str) -> None:
        """Store partial recommendations during processing to prevent data loss"""
        try:
            if not recommendations:
                return
                
            # Store a subset of recommendations as partial results
            partial_recs = recommendations[-20:]  # Store last 20 recommendations
            
            db_recommendations = []
            for rec in partial_recs:
                try:
                    # Use same conversion logic as main store method
                    confidence = self._safe_decimal(rec.get("confidence", 0), 0.5)
                    predicted_lift = self._safe_decimal(rec.get("predicted_lift", rec.get("lift", 1)), 1.2)
                    ranking_score = self._safe_decimal(rec.get("ranking_score", confidence * predicted_lift), confidence * predicted_lift)
                    
                    # Default structures for partial saves
                    pricing = rec.get("pricing", {
                        "bundle_price": 0.0,
                        "discount_amount": 0.0,
                        "discount_percentage": 0.0,
                        "individual_total": 0.0,
                        "pricing_strategy": "percentage_discount"
                    })
                    
                    ai_copy = rec.get("ai_copy", {
                        "title": "Bundle Deal (Partial)",
                        "description": "Partial result during processing",
                        "value_proposition": "Bundle in progress"
                    })
                    
                    db_rec = {
                        "id": rec["id"],
                        "csv_upload_id": csv_upload_id,
                        "bundle_type": rec.get("bundle_type", "FBT"),
                        "objective": rec.get("objective", "increase_aov"),
                        "products": rec.get("products", []),
                        "pricing": pricing,
                        "ai_copy": ai_copy,
                        "confidence": confidence,
                        "predicted_lift": predicted_lift,
                        "support": self._safe_decimal(rec.get("support", 0), None),
                        "lift": self._safe_decimal(rec.get("lift", 1), None),
                        "ranking_score": ranking_score,
                        "is_approved": False,
                        "is_used": False
                    }
                    
                    # Remove None values for optional fields
                    db_rec = {k: v for k, v in db_rec.items() if v is not None}
                    db_recommendations.append(db_rec)
                    
                except Exception as rec_error:
                    logger.warning(f"Error processing partial recommendation {rec.get('id', 'unknown')}: {rec_error}")
                    continue
            
            if db_recommendations:
                await storage.create_bundle_recommendations(db_recommendations)
                logger.info(f"Stored {len(db_recommendations)} partial recommendations")
            
        except Exception as e:
            logger.warning(f"Error storing partial recommendations: {e}")
    
    async def store_recommendations(self, recommendations: List[Dict[str, Any]], csv_upload_id: str) -> None:
        """Store recommendations in database"""
        try:
            if not recommendations:
                logger.info("No recommendations to store")
                return
                
            # Convert to database format with proper field mapping and data types
            db_recommendations = []
            for rec in recommendations:
                try:
                    # Convert numeric values to Decimal for database storage
                    confidence = self._safe_decimal(rec.get("confidence", 0), 0.5)
                    predicted_lift = self._safe_decimal(rec.get("predicted_lift", rec.get("lift", 1)), 1.2)
                    support = self._safe_decimal(rec.get("support", 0), None)
                    lift = self._safe_decimal(rec.get("lift", 1), None)
                    ranking_score = self._safe_decimal(rec.get("ranking_score", confidence * predicted_lift), confidence * predicted_lift)
                    
                    # Ensure JSON-serializable pricing data
                    pricing = rec.get("pricing", {})
                    if pricing:
                        pricing = self._serialize_pricing_for_json(pricing)
                    else:
                        # Default pricing structure
                        pricing = {
                            "bundle_price": 0.0,
                            "discount_amount": 0.0,
                            "discount_percentage": 0.0,
                            "individual_total": 0.0,
                            "pricing_strategy": "percentage_discount"
                        }
                    
                    # Ensure AI copy exists
                    ai_copy = rec.get("ai_copy", {})
                    if not ai_copy:
                        ai_copy = {
                            "title": "Bundle Deal",
                            "description": "Great products bundled together for savings",
                            "value_proposition": "Save money with this bundle"
                        }
                    
                    db_rec = {
                        "id": rec["id"],
                        "csv_upload_id": csv_upload_id,
                        "bundle_type": rec.get("bundle_type", "FBT"),
                        "objective": rec.get("objective", "increase_aov"),
                        "products": rec.get("products", []),
                        "pricing": pricing,
                        "ai_copy": ai_copy,
                        "confidence": confidence,
                        "predicted_lift": predicted_lift,
                        "support": support,
                        "lift": lift,
                        "ranking_score": ranking_score,
                        "discount_reference": rec.get("discount_reference"),
                        "is_approved": False,
                        "is_used": False,
                        "rank_position": rec.get("rank_position")
                    }
                    
                    # Remove None values for optional fields
                    db_rec = {k: v for k, v in db_rec.items() if v is not None}
                    db_recommendations.append(db_rec)
                    
                except Exception as rec_error:
                    logger.warning(f"Error processing recommendation {rec.get('id', 'unknown')}: {rec_error}")
                    continue
            
            if db_recommendations:
                # Store in database with detailed error handling
                logger.info(f"Attempting to store {len(db_recommendations)} bundle recommendations")
                await storage.create_bundle_recommendations(db_recommendations)
                logger.info(f"Successfully stored {len(db_recommendations)} bundle recommendations for upload {csv_upload_id}")
            else:
                logger.warning("No valid recommendations to store after processing")
            
        except Exception as e:
            logger.error(f"Error storing recommendations for upload {csv_upload_id}: {e}")
            logger.error(f"Recommendation sample: {recommendations[:1] if recommendations else 'None'}")
            # Re-raise to ensure calling code knows about the failure
            raise
    
    async def generate_v1_recommendations(self, csv_upload_id: Optional[str], metrics: Dict[str, Any], error: str = "") -> Dict[str, Any]:
        """Fallback v1 bundle generation pipeline"""
        logger.info("Using v1 fallback pipeline")
        
        try:
            # Get association rules (v1 approach)
            association_rules = await storage.get_association_rules(csv_upload_id)
            if not association_rules:
                logger.warning("No association rules found for v1 fallback")
                return {
                    "recommendations": [],
                    "metrics": {**metrics, "v1_fallback": True, "v2_error": error},
                    "v2_pipeline": False
                }
            
            # Basic v1 bundle generation
            recommendations = []
            
            for rule in association_rules[:50]:  # Limit for performance
                try:
                    if (rule.confidence >= self.min_confidence and 
                        rule.lift >= self.min_lift and 
                        rule.support >= self.min_support):
                        
                        recommendation = {
                            "id": str(uuid.uuid4()),
                            "csv_upload_id": csv_upload_id,
                            "bundle_type": "FBT",  # Default to FBT in v1
                            "objective": "increase_aov",  # Default objective
                            "products": [rule.antecedent, rule.consequent] if isinstance(rule.antecedent, str) else rule.antecedent + [rule.consequent],
                            "confidence": float(rule.confidence),
                            "lift": float(rule.lift),
                            "support": float(rule.support),
                            "explanation": f"Customers who buy these items together {rule.confidence:.0%} of the time",
                            "v1_fallback": True
                        }
                        
                        recommendations.append(recommendation)
                        
                except Exception as e:
                    logger.warning(f"Error processing rule in v1 fallback: {e}")
                    continue
            
            return {
                "recommendations": recommendations,
                "metrics": {**metrics, "v1_fallback": True, "v2_error": error, "total_recommendations": len(recommendations)},
                "v2_pipeline": False
            }
            
        except Exception as e:
            logger.error(f"Error in v1 fallback: {e}")
            return {
                "recommendations": [],
                "metrics": {**metrics, "v1_fallback": True, "v1_error": str(e), "v2_error": error},
                "v2_pipeline": False
            }