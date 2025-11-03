"""
Bundle Generator Service v2
Creates bundle recommendations using comprehensive v2 pipeline with enhanced features
"""
from typing import List, Dict, Any, Optional
import asyncio
import logging
import uuid
from decimal import Decimal
from datetime import datetime, timedelta
import random
import time
from collections import defaultdict
import os
import contextlib

from services.storage import storage
from services.ai_copy_generator import AICopyGenerator
from services.data_mapper import DataMapper
from services.objectives import ObjectiveScorer
from services.ml.candidate_generator import CandidateGenerator, CandidateGenerationContext
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
from services.notifications import notify_bundle_ready, notify_partial_ready
from services.progress_tracker import (
    update_generation_progress,
    get_generation_checkpoint,
)

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
except ImportError:  # pragma: no cover - optional dependency
    trace = None
    SpanKind = None

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

        self._tracer = trace.get_tracer(__name__) if trace else None
        self._initialize_configuration()

    @contextlib.contextmanager
    def _start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self._tracer or not SpanKind:
            yield None
            return
        with self._tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def _time_budget_exceeded(self, end_time: Optional[datetime]) -> bool:
        """Check whether the global time budget has been exhausted."""
        return bool(end_time and datetime.now() >= end_time)

    def _build_dataset_profile(
        self,
        context: Optional[CandidateGenerationContext],
        recommendations: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        transactions = len(getattr(context, "transactions", []) or [])
        sku_count = len(getattr(context, "valid_skus", []) or [])
        candidate_count = len(recommendations or [])
        tier = "small"
        if (
            transactions >= self.large_dataset_txn_threshold
            or sku_count >= self.large_dataset_sku_threshold
        ):
            tier = "large"
        elif (
            transactions >= self.medium_dataset_txn_threshold
            or sku_count >= self.medium_dataset_sku_threshold
        ):
            tier = "medium"

        avg_txn_per_sku = (
            float(transactions) / float(sku_count) if transactions and sku_count else 0.0
        )
        return {
            "tier": tier,
            "transaction_count": transactions,
            "unique_sku_count": sku_count,
            "candidate_count": candidate_count,
            "avg_transactions_per_sku": avg_txn_per_sku,
        }

    def _should_defer_async(
        self,
        dataset_profile: Dict[str, Any],
        end_time: Optional[datetime],
        resume_used: bool,
    ) -> bool:
        if not self.async_defer_enabled or resume_used:
            return False

        time_remaining: Optional[float] = None
        if end_time:
            time_remaining = (end_time - datetime.now()).total_seconds()
            if time_remaining is not None and time_remaining < 0:
                time_remaining = 0.0
        dataset_profile["time_remaining_seconds"] = time_remaining

        tier = dataset_profile.get("tier")
        candidate_count = dataset_profile.get("candidate_count", 0)

        if tier == "large":
            return True

        if self.async_defer_large_tier_only:
            return False

        if candidate_count >= self.async_defer_candidate_threshold:
            return True

        if (
            time_remaining is not None
            and time_remaining <= max(0, self.async_defer_min_time_remaining)
        ):
            return True

        return False

    def _derive_allocation_plan(self, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        plan = {
            "phase3_concurrency": self.phase3_concurrency_limit,
            "llm_candidate_target": 20,
        }

        tier = dataset_profile.get("tier")
        if tier == "small":
            plan["phase3_concurrency"] = max(2, min(self.phase3_concurrency_limit, 3))
            plan["llm_candidate_target"] = 12
        elif tier == "medium":
            plan["phase3_concurrency"] = min(self.phase3_concurrency_limit, 5)
            plan["llm_candidate_target"] = 18
        elif tier == "large":
            plan["phase3_concurrency"] = max(self.phase3_concurrency_limit, 8)
            plan["llm_candidate_target"] = 24
        return plan

    def _warn_if_time_low(self, end_time: Optional[datetime], csv_upload_id: str, phase_label: str) -> None:
        if not end_time or self.soft_timeout_warning_seconds <= 0:
            return
        remaining = (end_time - datetime.now()).total_seconds()
        if remaining is None:
            return
        if remaining <= self.soft_timeout_warning_seconds:
            warned = getattr(self, "_soft_timeout_warned", set())
            if phase_label in warned:
                return
            warned.add(phase_label)
            self._soft_timeout_warned = warned
            logger.warning(
                "[%s] Soft timeout warning | phase=%s remaining=%.1fs threshold=%ss",
                csv_upload_id,
                phase_label,
                max(remaining, 0),
                self.soft_timeout_warning_seconds,
            )

    async def _emit_heartbeat(
        self,
        csv_upload_id: str,
        *,
        step: str,
        progress: int,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.heartbeat_interval_seconds <= 0:
            return
        now = time.time()
        last = getattr(self, "_last_heartbeat_ts", 0.0)
        if now - last < self.heartbeat_interval_seconds:
            return
        self._last_heartbeat_ts = now
        try:
            await update_generation_progress(
                csv_upload_id,
                step=step,
                progress=progress,
                status="in_progress",
                message=message,
                metadata=metadata,
            )
            logger.debug(
                "[%s] Heartbeat emitted | step=%s progress=%d",
                csv_upload_id,
                step,
                progress,
            )
        except Exception as exc:  # pragma: no cover - heartbeat best-effort
            logger.debug("[%s] Heartbeat update failed: %s", csv_upload_id, exc)

    async def _finalize_soft_timeout(
        self,
        csv_upload_id: str,
        phase_name: str,
        metrics: Dict[str, Any],
        pipeline_start: float,
        partial_recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Gracefully terminate when the soft time budget is exceeded.
        Returns a result payload without attempting further heavy processing.
        """
        total_pipeline_duration = int((time.time() - pipeline_start) * 1000)
        metrics["processing_time_ms"] = total_pipeline_duration
        metrics["timeout_error"] = True
        metrics["timeout_phase"] = phase_name
        metrics["soft_timeout"] = True
        metrics["total_recommendations"] = len(partial_recommendations or [])

        logger.warning(
            f"[{csv_upload_id}] Soft timeout reached during {phase_name}. "
            f"Elapsed={total_pipeline_duration/1000:.1f}s, "
            f"candidates_retained={metrics['total_recommendations']}"
        )

        # Persist latest partial recommendations for potential resume attempts.
        await self.store_partial_recommendations(
            partial_recommendations or [],
            csv_upload_id,
            stage="soft_timeout",
        )

        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=f"Bundle generation stopped during {phase_name} due to time budget.",
            bundle_count=metrics["total_recommendations"] or None,
            metadata={"checkpoint": {"phase": phase_name, "timestamp": datetime.utcnow().isoformat()}},
        )

        # Persist the latest checkpoint details on the upload record for resumability.
        await self._update_upload_checkpoint(
            csv_upload_id,
            {
                "phase": phase_name,
                "soft_timeout": True,
                "timestamp": datetime.utcnow().isoformat(),
                "bundle_count": metrics["total_recommendations"],
            },
            metrics_snapshot={
                "total_recommendations": metrics["total_recommendations"],
                "timeout_phase": phase_name,
            },
        )

        return {
            "recommendations": partial_recommendations or [],
            "metrics": metrics,
            "v2_pipeline": True,
            "csv_upload_id": csv_upload_id,
        }

    async def _update_upload_checkpoint(
        self,
        csv_upload_id: str,
        checkpoint: Dict[str, Any],
        metrics_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist checkpoint data to the CSV upload record while preserving history."""
        try:
            upload = await storage.get_csv_upload(csv_upload_id)
            existing_metrics: Dict[str, Any] = {}
            if upload and getattr(upload, "bundle_generation_metrics", None):
                existing_metrics = dict(upload.bundle_generation_metrics)

            checkpoints = list(existing_metrics.get("checkpoints", []))
            checkpoints.append(checkpoint)
            # Keep only the most recent 10 checkpoints to avoid unbounded growth.
            existing_metrics["checkpoints"] = checkpoints[-10:]
            existing_metrics["last_checkpoint"] = checkpoint
            if metrics_snapshot:
                existing_metrics["latest_metrics_snapshot"] = metrics_snapshot

            await storage.update_csv_upload(
                csv_upload_id, {"bundle_generation_metrics": existing_metrics}
            )
        except Exception as exc:
            logger.warning(
                f"[{csv_upload_id}] Failed to persist checkpoint metadata: {exc}"
            )

    async def _record_checkpoint(
        self,
        csv_upload_id: str,
        phase: str,
        *,
        bundle_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct and persist a checkpoint entry."""
        checkpoint: Dict[str, Any] = {
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if bundle_count is not None:
            checkpoint["bundle_count"] = bundle_count
        if metadata:
            checkpoint.update(metadata)

        await self._update_upload_checkpoint(
            csv_upload_id, checkpoint, metrics_snapshot=metrics_snapshot
        )
        return checkpoint

    async def _load_resume_state(
        self, csv_upload_id: str
    ) -> Optional[Dict[str, Any]]:
        """Inspect saved checkpoints/partials to decide if we can resume mid-pipeline."""
        try:
            checkpoint = await get_generation_checkpoint(csv_upload_id)
        except Exception as exc:
            logger.warning(
                f"[{csv_upload_id}] Unable to load generation checkpoint: {exc}"
            )
            return None

        if not checkpoint:
            return None

        phase = checkpoint.get("phase")
        if phase not in {"phase_3_candidates", "phase_4_dedup"}:
            # Only resume in phases where we know how to rehydrate state.
            return None

        partial_recs = await storage.get_partial_bundle_recommendations(csv_upload_id)
        if not partial_recs:
            return None

        recommendations: List[Dict[str, Any]] = []
        for rec in partial_recs:
            try:
                recommendations.append(self._db_recommendation_to_internal(rec))
            except Exception as exc:
                logger.warning(
                    f"[{csv_upload_id}] Failed to convert partial recommendation {getattr(rec, 'id', 'unknown')}: {exc}"
                )

        if not recommendations:
            return None

        logger.info(
            f"[{csv_upload_id}] Resuming bundle generation from checkpoint {phase} "
            f"with {len(recommendations)} persisted candidates."
        )
        return {"phase": phase, "recommendations": recommendations}

    def _db_recommendation_to_internal(self, record: Any) -> Dict[str, Any]:
        """Convert a BundleRecommendation ORM instance into the internal dict format."""
        return {
            "id": record.id,
            "bundle_type": record.bundle_type,
            "objective": record.objective,
            "products": record.products,
            "pricing": record.pricing,
            "ai_copy": record.ai_copy,
            "confidence": float(record.confidence) if record.confidence is not None else 0.0,
            "predicted_lift": float(record.predicted_lift) if record.predicted_lift is not None else 0.0,
            "support": float(record.support) if record.support is not None else None,
            "lift": float(record.lift) if record.lift is not None else None,
            "ranking_score": float(record.ranking_score) if record.ranking_score is not None else 0.0,
            "discount_reference": None,
            "is_resumed": True,
        }

    def _initialize_configuration(self) -> None:
        # Bundle configuration
        self.bundle_types = ['FBT', 'VOLUME_DISCOUNT', 'MIX_MATCH', 'BXGY', 'FIXED']

        # 8 Objective types for enhanced bundle generation (all defined for backward compatibility)
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

        # PARETO OPTIMIZATION: Map each objective to best-fit bundle type(s)
        # This reduces 8 objectives × 5 types (40 tasks) to 3-4 objectives × 1-2 types (3-8 tasks)
        self.objective_to_bundle_types = {
            # Top priority objectives (Pareto 80/20)
            'margin_guard': ['FBT', 'FIXED'],              # Protect margins: FBT works best
            'clear_slow_movers': ['VOLUME_DISCOUNT', 'BXGY'],  # Move inventory: Volume discounts
            'increase_aov': ['MIX_MATCH', 'FBT'],          # Boost AOV: Mix & Match maximizes cart

            # Secondary objectives (only for large datasets)
            'new_launch': ['FBT', 'FIXED'],                # Promote new products: FBT exposure
            'seasonal_promo': ['BXGY', 'FIXED'],           # Seasonal: BXGY promotions

            # Low priority (rarely used)
            'category_bundle': ['MIX_MATCH'],
            'gift_box': ['FIXED'],
            'subscription_push': ['VOLUME_DISCOUNT'],
        }

        # Early termination thresholds
        self.min_transactions_for_ml = 10  # Skip ML phase if < 10 transactions
        self.min_products_for_ml = 5       # Skip ML phase if < 5 unique products
        self.min_transactions_for_llm_only = int(os.getenv("MIN_TXNS_FOR_LLM_ONLY", "1"))
        
        # Bundle generation thresholds
        self.base_min_support = 0.05
        self.base_min_confidence = 0.3
        self.base_min_lift = 1.2
        self.min_support = self.base_min_support
        self.min_confidence = self.base_min_confidence
        self.min_lift = self.base_min_lift
        self._last_threshold_signature: Optional[tuple] = None
        self._refresh_thresholds()
        
        # v2 feature flags
        self.enable_v2_pipeline = True  # Enable comprehensive v2 features
        self.enable_data_mapping = True
        self.enable_objective_scoring = False  # DISABLED: 60s+ timeout bottleneck
        self.enable_ml_candidates = True
        self.enable_bayesian_pricing = True
        self.enable_weighted_ranking = True
        self.enable_deduplication = True
        self.enable_explainability = False  # DISABLED: Non-critical, saves time
        
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
        self.phase3_concurrency_limit = max(1, int(os.getenv("PHASE3_CONCURRENCY_LIMIT", "6")))
        
        # ARCHITECT FIX: Circuit-breaker pattern to stop runaway behavior
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # Stop after 10 consecutive failures
        self.circuit_breaker_active = False

        # New caps for optimization and diversity safeguards
        self.min_candidates_for_optimization = 5
        self.max_bundles_per_pair = 2
        self.medium_dataset_txn_threshold = int(os.getenv("DATASET_MEDIUM_TXN_THRESHOLD", "200"))
        self.large_dataset_txn_threshold = int(os.getenv("DATASET_LARGE_TXN_THRESHOLD", "500"))
        self.medium_dataset_sku_threshold = int(os.getenv("DATASET_MEDIUM_SKU_THRESHOLD", "150"))
        self.large_dataset_sku_threshold = int(os.getenv("DATASET_LARGE_SKU_THRESHOLD", "350"))
        self.async_defer_enabled = os.getenv("BUNDLING_ASYNC_DEFER", "true").lower() != "false"
        self.async_defer_min_time_remaining = int(os.getenv("ASYNC_DEFER_MIN_TIME", "120"))
        self.async_defer_candidate_threshold = int(os.getenv("ASYNC_DEFER_CANDIDATE_THRESHOLD", "120"))
        self.async_defer_large_tier_only = os.getenv("ASYNC_DEFER_LARGE_ONLY", "false").lower() == "true"
        self.heartbeat_interval_seconds = int(os.getenv("BUNDLE_HEARTBEAT_SECONDS", "30"))
        self.soft_timeout_warning_seconds = int(os.getenv("BUNDLE_SOFT_TIMEOUT_WARNING", "45"))

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

    def _refresh_thresholds(self) -> None:
        """Pull threshold overrides from feature flags with safe fallbacks."""
        use_relaxed = feature_flags.get_flag("bundling.relaxed_thresholds", True)

        def _safe_numeric(flag_key: str, default: float) -> float:
            value = feature_flags.get_flag(flag_key, default)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        if use_relaxed:
            self.min_support = max(0.0, _safe_numeric("bundling.relaxed_min_support", self.base_min_support))
            self.min_confidence = max(0.0, _safe_numeric("bundling.relaxed_min_confidence", self.base_min_confidence))
            self.min_lift = max(0.0, _safe_numeric("bundling.relaxed_min_lift", self.base_min_lift))
        else:
            self.min_support = self.base_min_support
            self.min_confidence = self.base_min_confidence
            self.min_lift = self.base_min_lift

        signature = (round(self.min_support, 4), round(self.min_confidence, 4), round(self.min_lift, 4), bool(use_relaxed))
        if signature != self._last_threshold_signature:
            logger.info(
                "Bundle thresholds updated: support=%.3f confidence=%.3f lift=%.3f (relaxed=%s)",
                self.min_support,
                self.min_confidence,
                self.min_lift,
                use_relaxed
            )
            self._last_threshold_signature = signature

    async def _gather_with_concurrency(self, limit: int, coroutines: List[Any]) -> List[Any]:
        """Run coroutines with a concurrency cap to avoid exhausting DB connections."""
        if not coroutines:
            return []
        semaphore = asyncio.Semaphore(max(1, limit))

        async def _runner(coro):
            async with semaphore:
                return await coro

        return await asyncio.gather(*(_runner(coro) for coro in coroutines), return_exceptions=True)

    def _should_skip_ml_phase(self, context: CandidateGenerationContext, csv_upload_id: str) -> tuple[bool, str]:
        """
        Check if ML phase should be skipped due to insufficient data.
        Returns (should_skip: bool, reason: str)
        """
        try:
            # Check transaction count
            txn_count = len(context.transactions) if context and context.transactions else 0
            product_count = len(context.valid_skus) if context and context.valid_skus else 0

            logger.info(
                f"[{csv_upload_id}] PARETO: Early termination check | "
                f"txn_count={txn_count} (threshold={self.min_transactions_for_ml}), "
                f"product_count={product_count} (threshold={self.min_products_for_ml})"
            )

            if txn_count < self.min_transactions_for_ml:
                # Allow LLM-only generation for sparse datasets when embeddings exist
                if (
                    context
                    and context.embeddings
                    and len(context.embeddings) >= 2
                    and txn_count >= self.min_transactions_for_llm_only
                ):
                    context.llm_only = True
                    logger.info(
                        f"[{csv_upload_id}] PARETO: Enabling LLM-only candidate generation | "
                        f"txn_count={txn_count} embeddings={len(context.embeddings)}"
                    )
                    return (False, "")

                reason = f"Only {txn_count} transactions (need {self.min_transactions_for_ml}+)"
                logger.warning(
                    f"[{csv_upload_id}] PARETO: Skipping ML phase - {reason} | "
                    f"This will save ~195s of wasted computation"
                )
                return (True, reason)

            # Check unique products
            if product_count < self.min_products_for_ml:
                if context and getattr(context, "llm_only", False) and product_count >= 2:
                    logger.info(
                        f"[{csv_upload_id}] PARETO: Proceeding with LLM-only flow despite small catalog | "
                        f"product_count={product_count}"
                    )
                else:
                    reason = f"Only {product_count} unique products (need {self.min_products_for_ml}+)"
                    logger.warning(
                        f"[{csv_upload_id}] PARETO: Skipping ML phase - {reason} | "
                        f"Insufficient product catalog for meaningful bundles"
                    )
                    return (True, reason)

            logger.info(
                f"[{csv_upload_id}] PARETO: ML phase proceeding | "
                f"txn_count={txn_count}, product_count={product_count} | "
                f"Both thresholds passed"
            )
            return (False, "")

        except Exception as e:
            logger.error(
                f"[{csv_upload_id}] PARETO: Error in early termination check: {e} | "
                f"Defaulting to proceed with ML phase",
                exc_info=True
            )
            return (False, "")

    def _select_objectives_for_dataset(self, context: CandidateGenerationContext) -> List[str]:
        """
        Dynamically select objectives based on dataset size using Pareto principle.
        Returns top objectives that cover 80% of business value.
        """
        try:
            if context and getattr(context, "llm_only", False):
                logger.info(
                    "PARETO: LLM-only mode active | returning default objective set ['increase_aov']"
                )
                return ['increase_aov']

            txn_count = len(context.transactions) if context and context.transactions else 0
            product_count = len(context.valid_skus) if context and context.valid_skus else 0
    
            logger.info(
                f"PARETO: Objective selection | "
                f"txn_count={txn_count}, product_count={product_count}"
            )

            # Tiny dataset (<10 txns): Skip ML entirely (handled by _should_skip_ml_phase)
            # Defensive check retained for clarity
            if txn_count < 10:
                logger.info(
                    f"PARETO: Tiny dataset ({txn_count} txns) without LLM-only flag | "
                    f"falling back to minimal objective set ['increase_aov']"
                )
                return ['increase_aov']

            # Small dataset (10-50 txns): Focus on top 2 objectives (Pareto 80%)
            if txn_count < 50:
                objectives = ['margin_guard', 'increase_aov']  # 2 objectives × 2 types = 4 tasks
                logger.info(
                    f"PARETO: Small dataset tier | "
                    f"txn_count={txn_count} | "
                    f"selected_objectives={objectives} ({len(objectives)} objectives) | "
                    f"expected_tasks={len(objectives) * 2} (vs 40 baseline) | "
                    f"reduction={(1 - (len(objectives) * 2) / 40) * 100:.0f}%"
                )
                return objectives

            # Medium dataset (50-200 txns): Top 3 objectives (Pareto 80%)
            elif txn_count < 200:
                objectives = ['margin_guard', 'clear_slow_movers', 'increase_aov']  # 3 objectives × 2 types = 6 tasks
                logger.info(
                    f"PARETO: Medium dataset tier | "
                    f"txn_count={txn_count} | "
                    f"selected_objectives={objectives} ({len(objectives)} objectives) | "
                    f"expected_tasks={len(objectives) * 2} (vs 40 baseline) | "
                    f"reduction={(1 - (len(objectives) * 2) / 40) * 100:.0f}%"
                )
                return objectives

            # Large dataset (200+ txns): Top 4 objectives
            else:
                objectives = ['margin_guard', 'clear_slow_movers', 'increase_aov', 'new_launch']  # 4 objectives × 2 types = 8 tasks
                logger.info(
                    f"PARETO: Large dataset tier | "
                    f"txn_count={txn_count} | "
                    f"selected_objectives={objectives} ({len(objectives)} objectives) | "
                    f"expected_tasks={len(objectives) * 2} (vs 40 baseline) | "
                    f"reduction={(1 - (len(objectives) * 2) / 40) * 100:.0f}%"
                )
                return objectives

        except Exception as e:
            logger.error(
                f"PARETO: Error in objective selection: {e} | "
                f"Falling back to minimal safe objectives",
                exc_info=True
            )
            return ['margin_guard', 'increase_aov']  # Safe fallback

    def _get_bundle_types_for_objective(self, objective: str) -> List[str]:
        """
        Get the best-fit bundle types for a given objective.
        Returns 1-2 bundle types instead of all 5.
        """
        try:
            types = self.objective_to_bundle_types.get(objective, ['FBT'])

            if objective not in self.objective_to_bundle_types:
                logger.warning(
                    f"PARETO: Unknown objective '{objective}' | "
                    f"Falling back to default bundle type ['FBT'] | "
                    f"Known objectives: {list(self.objective_to_bundle_types.keys())}"
                )
            else:
                logger.info(
                    f"PARETO: Bundle type mapping | "
                    f"objective={objective} -> types={types} | "
                    f"Reduced from 5 types to {len(types)} type(s)"
                )

            return types

        except Exception as e:
            logger.error(
                f"PARETO: Error getting bundle types for objective '{objective}': {e} | "
                f"Falling back to safe default ['FBT']",
                exc_info=True
            )
            return ['FBT']  # Safe fallback

    async def _apply_forced_pair_fallbacks(self, recommendations: List[Dict[str, Any]], csv_upload_id: str) -> List[Dict[str, Any]]:
        """Inject top association rule pairs when coverage is too low."""
        if not feature_flags.get_flag("bundling.fallback_force_top_pairs", True):
            return recommendations

        target_total = feature_flags.get_flag("bundling.fallback_force_pair_limit", 12)
        try:
            target_total = int(target_total)
        except (TypeError, ValueError):
            target_total = 12

        if target_total <= 0 or len(recommendations) >= target_total:
            return recommendations

        needed = target_total - len(recommendations)
        if needed <= 0:
            return recommendations

        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            rule_limit = max(needed * 3, 20)
            if run_id:
                rules = await storage.get_association_rules_by_run(run_id, limit=rule_limit)
            else:
                rules = await storage.get_association_rules(csv_upload_id, limit=rule_limit)
        except Exception as exc:
            logger.warning(f"Forced pair fallback skipped: unable to load rules ({exc})")
            return recommendations

        existing_signatures = {
            tuple(sorted(rec.get("products", [])))
            for rec in recommendations
            if rec.get("products")
        }

        injected = []
        for rule in rules:
            antecedent = rule.antecedent if isinstance(rule.antecedent, list) else [rule.antecedent]
            consequent = rule.consequent if isinstance(rule.consequent, list) else [rule.consequent]
            products = [p for p in (antecedent + consequent) if p]
            if len(products) != 2:
                continue
            signature = tuple(sorted(products))
            if signature in existing_signatures:
                continue

            fallback_rec = {
                "id": str(uuid.uuid4()),
                "csv_upload_id": csv_upload_id,
                "bundle_type": "FBT",
                "objective": "increase_aov",
                "products": products,
                "confidence": float(getattr(rule, "confidence", 0.0) or 0.0),
                "lift": float(getattr(rule, "lift", 1.0) or 1.0),
                "support": float(getattr(rule, "support", 0.0) or 0.0),
                "generation_sources": ["association_rule_fallback"],
                "generation_method": "forced_top_pair",
                "is_fallback": True,
                "fallback_reason": "top_pair_injection"
            }
            # Provide a baseline ranking score so downstream ordering remains deterministic
            fallback_rec["ranking_score"] = fallback_rec["confidence"] * max(fallback_rec["lift"], 1.0)
            injected.append(fallback_rec)
            existing_signatures.add(signature)
            if len(injected) >= needed:
                break

        if injected:
            recommendations.extend(injected)
            recommendations.sort(key=lambda rec: rec.get("ranking_score", 0), reverse=True)
            logger.info(f"Forced pair fallback injected {len(injected)} bundles to reach minimum coverage")

        return recommendations
    
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

        await update_generation_progress(
            csv_upload_id,
            step="enrichment",
            progress=5,
            status="in_progress",
            message="Starting enrichment…",
        )

        pipeline_start = time.time()
        logger.info(f"[{csv_upload_id}] ========== BUNDLE GENERATION PIPELINE STARTED ==========")
        logger.info(f"[{csv_upload_id}] Configuration: timeout={self.max_time_budget_seconds}s max_attempts={self.max_total_attempts}")
        self._refresh_thresholds()

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

        # Initialize comprehensive metrics with timing
        metrics = {
            "v2_pipeline_enabled": self.enable_v2_pipeline,
            "data_mapping": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "objective_scoring": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "ml_candidates": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "bayesian_pricing": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "weighted_ranking": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "deduplication": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "explainability": {"enabled": False, "metrics": {}, "duration_ms": 0},
            "bundle_counts": {"FBT": 0, "VOLUME_DISCOUNT": 0, "MIX_MATCH": 0, "BXGY": 0, "FIXED": 0},
            "total_recommendations": 0,
            "processing_time_ms": 0,
            "loop_prevention_stats": self.generation_stats,
            "phase_timings": {}
        }

        resume_state = await self._load_resume_state(csv_upload_id)
        resume_recommendations = resume_state["recommendations"] if resume_state else None
        resume_phase = resume_state["phase"] if resume_state else None

        start_time = datetime.now()
        self._last_heartbeat_ts = time.time()
        self._soft_timeout_warned = set()

        feature_flag_snapshot = {
            "data_mapping": self.enable_data_mapping,
            "objective_scoring": self.enable_objective_scoring,
            "ml_candidates": self.enable_ml_candidates,
            "bayesian_pricing": self.enable_bayesian_pricing,
            "weight_ranking": self.enable_weighted_ranking,
            "deduplication": self.enable_deduplication,
            "explainability": self.enable_explainability,
        }
        pipeline_run_id = f"bundle:{csv_upload_id}"
        pipeline_started = False
        pipeline_finished = False
        if not resume_recommendations:
            try:
                metrics_collector.start_pipeline(
                    pipeline_run_id,
                    csv_upload_id,
                    "bundle_generation",
                    feature_flag_snapshot,
                )
                pipeline_started = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[%s] Unable to start pipeline metrics tracking: %s", csv_upload_id, exc)

        # Set hard timeout
        end_time = start_time + timedelta(seconds=self.max_time_budget_seconds)

        try:
            # Phase 1: Data Mapping and Enrichment
            if self.enable_data_mapping and csv_upload_id:
                phase_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 1: Data Mapping & Enrichment - STARTED")
                data_mapping_result = await self.data_mapper.enrich_order_lines_with_variants(csv_upload_id)
                phase_duration = int((time.time() - phase_start) * 1000)
                enrichment_metrics = data_mapping_result.get("metrics", {})
                total_order_lines = enrichment_metrics.get('total_order_lines', 0)
                logger.info(f"[{csv_upload_id}] Phase 1: Data Mapping & Enrichment - COMPLETED in {phase_duration}ms | "
                           f"total_lines={total_order_lines} "
                           f"resolved={enrichment_metrics.get('resolved_variants', 0)} "
                           f"unresolved={enrichment_metrics.get('unresolved_skus', 0)}")
                metrics["data_mapping"] = {"enabled": True, "metrics": enrichment_metrics, "duration_ms": phase_duration}
                metrics["total_order_lines"] = total_order_lines  # Track for Phase 3 FallbackLadder decision
                metrics["phase_timings"]["phase_1_data_mapping"] = phase_duration
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_1_enrichment_completed",
                    bundle_count=total_order_lines,
                    metrics_snapshot={"data_mapping": enrichment_metrics},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="enrichment",
                    progress=25,
                    status="in_progress",
                    message=(
                        "Enrichment complete – "
                        f"{enrichment_metrics.get('resolved_variants', 0)} variants resolved."
                    ),
                    bundle_count=total_order_lines if total_order_lines else None,
                    metadata={"checkpoint": checkpoint},
                )
            else:
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_1_enrichment_skipped",
                    metrics_snapshot={"data_mapping": {"enabled": False}},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="enrichment",
                    progress=25,
                    status="in_progress",
                    message="Enrichment skipped (disabled).",
                    metadata={"checkpoint": checkpoint},
                )
            
            await update_generation_progress(
                csv_upload_id,
                step="scoring",
                progress=30,
                status="in_progress",
                message="Scoring objectives…",
            )

            # Phase 2: Objective Scoring
            if self.enable_objective_scoring and csv_upload_id:
                phase_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 2: Objective Scoring - STARTED")

                # Add timeout for objective scoring (60 seconds max)
                try:
                    objective_result = await asyncio.wait_for(
                        self.objective_scorer.compute_objective_flags(csv_upload_id),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"[{csv_upload_id}] Phase 2: Objective Scoring TIMEOUT after 60s - continuing with empty flags")
                    objective_result = {"metrics": {"timeout": True, "total_items": 0}, "updated_items": 0}

                phase_duration = int((time.time() - phase_start) * 1000)
                objective_metrics = objective_result.get("metrics", {})
                logger.info(f"[{csv_upload_id}] Phase 2: Objective Scoring - COMPLETED in {phase_duration}ms | "
                           f"products_scored={objective_metrics.get('products_scored', 0)} "
                           f"objectives_computed={len(self.objectives)}")
                metrics["objective_scoring"] = {"enabled": True, "metrics": objective_metrics, "duration_ms": phase_duration}
                metrics["phase_timings"]["phase_2_objective_scoring"] = phase_duration
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_2_objective_scoring_completed",
                    metrics_snapshot={"objective_scoring": objective_metrics},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="scoring",
                    progress=45,
                    status="in_progress",
                    message="Objective scoring complete.",
                    metadata={"checkpoint": checkpoint},
                )
            else:
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_2_objective_scoring_skipped",
                    metrics_snapshot={"objective_scoring": {"enabled": False}},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="scoring",
                    progress=45,
                    status="in_progress",
                    message="Objective scoring skipped (disabled).",
                    metadata={"checkpoint": checkpoint},
                )
            
            phase3_checkpoint_start = await self._record_checkpoint(
                csv_upload_id,
                "phase_3_candidates_started",
                metrics_snapshot={"resume_phase": resume_phase} if resume_phase else None,
            )
            await update_generation_progress(
                csv_upload_id,
                step="ml_generation",
                progress=50,
                status="in_progress",
                message="Generating ML candidates…",
                metadata={"checkpoint": phase3_checkpoint_start},
            )

            # Phase 3: Generate candidates for each objective with loop prevention
            phase3_start = time.time()
            all_recommendations: List[Dict[str, Any]] = []
            objectives_processed = 0
            resume_used = bool(resume_recommendations)

            candidate_context: Optional[CandidateGenerationContext] = None

            if resume_recommendations:
                all_recommendations = resume_recommendations
                metrics["ml_candidates"] = {
                    "enabled": True,
                    "resume_used": True,
                    "duration_ms": 0,
                    "candidates_generated": len(all_recommendations),
                }
                metrics["phase_timings"]["phase_3_ml_candidates"] = 0
                candidate_count = len(all_recommendations)
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_3_candidates_resumed",
                    bundle_count=candidate_count,
                    metadata={"resume_phase": resume_phase},
                    metrics_snapshot={"resume_candidates": candidate_count},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="ml_generation",
                    progress=65,
                    status="in_progress",
                    message=f"Resumed from saved candidates ({candidate_count} ready).",
                    bundle_count=candidate_count,
                    metadata={"checkpoint": checkpoint, "partial_bundle_count": candidate_count},
                )
            else:
                candidate_context = await self.candidate_generator.prepare_context(csv_upload_id)

                # Early termination check
                should_skip, skip_reason = self._should_skip_ml_phase(candidate_context, csv_upload_id)
                if should_skip:
                    logger.warning(f"[{csv_upload_id}] Phase 3: ML Candidate Generation - SKIPPED | reason={skip_reason}")
                    metrics["ml_candidates"] = {
                        "enabled": False,
                        "skipped": True,
                        "skip_reason": skip_reason,
                        "duration_ms": 0
                    }
                    metrics["phase_timings"]["phase_3_ml_candidates"] = 0
                    checkpoint = await self._record_checkpoint(
                        csv_upload_id,
                        "phase_3_candidates_skipped",
                        metadata={"skip_reason": skip_reason},
                    )
                    await update_generation_progress(
                        csv_upload_id,
                        step="ml_generation",
                        progress=70,
                        status="in_progress",
                        message=f"ML generation skipped: {skip_reason}",
                        metadata={"checkpoint": checkpoint},
                    )
                else:
                    # PARETO OPTIMIZATION: Select top objectives dynamically based on dataset size
                    selected_objectives = self._select_objectives_for_dataset(candidate_context)
                    logger.info(f"[{csv_upload_id}] Phase 3: ML Candidate Generation - STARTED | "
                               f"selected_objectives={len(selected_objectives)} (Pareto optimized from {len(self.objectives)})")
                    dataset_profile_phase3 = self._build_dataset_profile(candidate_context, [])
                    allocation_plan = self._derive_allocation_plan(dataset_profile_phase3)
                    metrics["dataset_profile_phase3"] = dataset_profile_phase3
                    metrics["allocation_plan"] = allocation_plan
                    if candidate_context:
                        candidate_context.llm_candidate_target = allocation_plan.get("llm_candidate_target", 20)
                    phase3_concurrency_limit = allocation_plan.get("phase3_concurrency", self.phase3_concurrency_limit)

                    # PARALLEL EXECUTION...
                    generation_tasks = []
                    try:
                        for objective_name in selected_objectives:
                            bundle_types_for_objective = self._get_bundle_types_for_objective(objective_name)
                            for bundle_type in bundle_types_for_objective:
                                task = self.generate_objective_bundles(
                                    csv_upload_id,
                                    objective_name,
                                    bundle_type,
                                    metrics,
                                    end_time,
                                    candidate_context,
                                )
                                generation_tasks.append((objective_name, bundle_type, task))
                                logger.debug(
                                    f"PARETO: Task created | "
                                    f"objective={objective_name}, bundle_type={bundle_type}"
                                )

                        old_task_count = len(self.objectives) * len(self.bundle_types)
                        new_task_count = len(generation_tasks)
                        reduction_pct = int((1 - new_task_count / old_task_count) * 100) if old_task_count > 0 else 0

                        logger.info(
                            f"[{csv_upload_id}] PARETO: Task creation complete | "
                            f"old_task_count={old_task_count} (8 objectives × 5 types) → "
                            f"new_task_count={new_task_count} | "
                            f"reduction={reduction_pct}% | "
                            f"selected_objectives={selected_objectives}"
                        )
                        logger.info(
                            f"[{csv_upload_id}] PARETO: Starting parallel execution | "
                            f"tasks={new_task_count}, concurrency_limit={phase3_concurrency_limit}"
                        )

                    except Exception as e:
                        logger.error(
                            f"[{csv_upload_id}] PARETO: Error building generation tasks: {e} | "
                            f"Proceeding with {len(generation_tasks)} tasks created so far",
                            exc_info=True
                        )

                    parallel_start = time.time()

                    try:
                        tasks_only = [task for _, _, task in generation_tasks]

                        logger.info(
                            f"[{csv_upload_id}] PARETO: Executing parallel tasks | "
                            f"task_count={len(tasks_only)}"
                        )

                        await self._emit_heartbeat(
                            csv_upload_id,
                            step="ml_generation",
                            progress=58,
                            message="Generating ML candidates…",
                            metadata={"active_tasks": len(tasks_only)},
                        )

                        results = await self._gather_with_concurrency(phase3_concurrency_limit, tasks_only)

                        parallel_duration = int((time.time() - parallel_start) * 1000)
                        logger.info(
                            f"[{csv_upload_id}] PARETO: Parallel execution complete | "
                            f"wall_clock_time={parallel_duration}ms, "
                            f"avg_time_per_task={parallel_duration // len(tasks_only) if tasks_only else 0}ms"
                        )

                        success_count = 0
                        failure_count = 0
                        empty_count = 0

                        for (objective_name, bundle_type, _), result in zip(generation_tasks, results):
                            if isinstance(result, Exception):
                                logger.warning(
                                    f"[{csv_upload_id}] PARETO: Task failed | "
                                    f"objective={objective_name}, bundle_type={bundle_type}, "
                                    f"error={str(result)[:100]}"
                                )
                                self.generation_stats['failed_attempts'] += 1
                                failure_count += 1
                            elif isinstance(result, list):
                                all_recommendations.extend(result)
                                if len(result) > 0:
                                    objectives_processed += 1
                                    success_count += 1
                                    logger.info(
                                        f"[{csv_upload_id}] PARETO: Task succeeded | "
                                        f"objective={objective_name}, bundle_type={bundle_type}, "
                                        f"bundles_generated={len(result)}"
                                    )
                                else:
                                    empty_count += 1
                                    logger.debug(
                                        f"PARETO: Task completed but generated 0 bundles | "
                                        f"objective={objective_name}, bundle_type={bundle_type}"
                                    )

                        logger.info(
                            f"[{csv_upload_id}] PARETO: Result processing complete | "
                            f"total_candidates={len(all_recommendations)}, "
                            f"successful_combinations={objectives_processed}, "
                            f"success={success_count}, empty={empty_count}, failed={failure_count}"
                        )

                    except Exception as e:
                        parallel_duration = int((time.time() - parallel_start) * 1000)
                        logger.error(
                            f"[{csv_upload_id}] PARETO: Error during parallel execution: {e} | "
                            f"partial_results={len(all_recommendations)} candidates | "
                            f"duration={parallel_duration}ms",
                            exc_info=True
                        )

                    phase3_duration = int((time.time() - phase3_start) * 1000)
                    logger.info(f"[{csv_upload_id}] Phase 3: ML Candidate Generation - COMPLETED in {phase3_duration}ms | "
                               f"candidates_generated={len(all_recommendations)} "
                               f"objectives_processed={objectives_processed} "
                               f"attempts={self.generation_stats['total_attempts']} "
                               f"successes={self.generation_stats['successful_generations']} "
                               f"duplicates_skipped={self.generation_stats['skipped_duplicates']}")
                    metrics["ml_candidates"] = {"enabled": True, "duration_ms": phase3_duration,
                                                "candidates_generated": len(all_recommendations)}
                    metrics["phase_timings"]["phase_3_ml_candidates"] = phase3_duration
                    candidate_count = len(all_recommendations)
                    checkpoint = await self._record_checkpoint(
                        csv_upload_id,
                        "phase_3_candidates_completed",
                        bundle_count=candidate_count,
                        metrics_snapshot={"ml_candidates": metrics["ml_candidates"]},
                    )
                    await update_generation_progress(
                        csv_upload_id,
                        step="ml_generation",
                        progress=70,
                        status="in_progress",
                        message=f"ML candidate generation complete – {candidate_count} candidates ready.",
                        bundle_count=candidate_count if candidate_count else None,
                        metadata={"checkpoint": checkpoint},
                    )

            if resume_recommendations:
                candidate_count = len(all_recommendations)
                await self.store_partial_recommendations(
                    all_recommendations, csv_upload_id, stage="phase_3_resume"
                )
            elif not resume_used and metrics["ml_candidates"]["enabled"]:
                candidate_count = len(all_recommendations)
                await self.store_partial_recommendations(
                    all_recommendations, csv_upload_id, stage="phase_3"
                )
            else:
                candidate_count = len(all_recommendations)

            if resume_recommendations:
                await update_generation_progress(
                    csv_upload_id,
                    step="ml_generation",
                    progress=70,
                    status="in_progress",
                    message=f"Resumed candidates ready – {candidate_count} available.",
                    bundle_count=candidate_count,
                    metadata={"checkpoint": checkpoint},
                )
            elif not metrics["ml_candidates"].get("enabled"):
                candidate_count = 0
            if not resume_recommendations and metrics["ml_candidates"].get("enabled"):
                await update_generation_progress(
                    csv_upload_id,
                    step="ml_generation",
                    progress=70,
                    status="in_progress",
                    message=f"ML candidate generation complete – {candidate_count} candidates ready.",
                    bundle_count=candidate_count if candidate_count else None,
                    metadata={"partial_bundle_count": candidate_count, "checkpoint": checkpoint},
                )
            await update_generation_progress(
                csv_upload_id,
                step="optimization",
                progress=75,
                status="in_progress",
                message="Optimizing bundle candidates…",
            )

            dataset_profile = self._build_dataset_profile(candidate_context, all_recommendations)
            metrics["dataset_profile"] = dataset_profile

            if self._should_defer_async(dataset_profile, end_time, resume_used):
                logger.info(
                    "[%s] Async deferral engaged | tier=%s candidates=%s time_remaining=%s",
                    csv_upload_id,
                    dataset_profile.get("tier"),
                    dataset_profile.get("candidate_count"),
                    dataset_profile.get("time_remaining_seconds"),
                )
                await self.store_partial_recommendations(
                    all_recommendations, csv_upload_id, stage="phase_3_deferred"
                )
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_3_async_deferred",
                    bundle_count=len(all_recommendations),
                    metadata={"dataset_profile": dataset_profile},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="optimization",
                    progress=78,
                    status="in_progress",
                    message="Phase 3 complete. Continuing heavy phases asynchronously…",
                    bundle_count=len(all_recommendations) if all_recommendations else None,
                    metadata={"checkpoint": checkpoint, "dataset_profile": dataset_profile},
                )
                metrics["async_deferred"] = True
                metrics["processing_time_ms"] = int((time.time() - pipeline_start) * 1000)
                await notify_partial_ready(csv_upload_id, len(all_recommendations))
                return {
                    "recommendations": all_recommendations,
                    "metrics": metrics,
                    "async_deferred": True,
                    "dataset_profile": dataset_profile,
                }

            self._warn_if_time_low(end_time, csv_upload_id, "Phase 4 (pre-deduplication)")
            if self._time_budget_exceeded(end_time):
                result = await self._finalize_soft_timeout(
                    csv_upload_id,
                    "Phase 4 (pre-deduplication)",
                    metrics,
                    pipeline_start,
                    all_recommendations,
                )
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(result.get("recommendations", [])),
                        success=False,
                    )
                    pipeline_finished = True
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("[%s] Failed to finish pipeline metrics on soft timeout: %s", csv_upload_id, exc)
                return result

            # Phase 4: Deduplication
            if self.enable_deduplication and all_recommendations:
                phase_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 4: Deduplication - STARTED | input_candidates={len(all_recommendations)}")
                dedupe_result = await self.deduplicator.deduplicate_candidates(all_recommendations, csv_upload_id)
                all_recommendations = dedupe_result.get("unique_candidates", all_recommendations)
                phase_duration = int((time.time() - phase_start) * 1000)
                dedupe_metrics = dedupe_result.get("metrics", {})
                logger.info(f"[{csv_upload_id}] Phase 4: Deduplication - COMPLETED in {phase_duration}ms | "
                           f"unique_candidates={len(all_recommendations)} "
                           f"duplicates_removed={dedupe_metrics.get('duplicates_removed', 0)}")
                metrics["deduplication"] = {"enabled": True, "metrics": dedupe_metrics, "duration_ms": phase_duration}
                metrics["phase_timings"]["phase_4_deduplication"] = phase_duration
                await self.store_partial_recommendations(
                    all_recommendations, csv_upload_id, stage="phase_4"
                )
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_4_deduplication_completed",
                    bundle_count=len(all_recommendations),
                    metrics_snapshot={"deduplication": dedupe_metrics},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="optimization",
                    progress=80,
                    status="in_progress",
                    message="Deduplication complete.",
                    bundle_count=len(all_recommendations) if all_recommendations else None,
                    metadata={"checkpoint": checkpoint},
                )
                await self._emit_heartbeat(
                    csv_upload_id,
                    step="optimization",
                    progress=80,
                    message="Post-deduplication heartbeat",
                    metadata={"remaining_bundles": len(all_recommendations)},
                )
            
            self._warn_if_time_low(end_time, csv_upload_id, "Phase 5 (pre-enterprise optimization)")
            if self._time_budget_exceeded(end_time):
                result = await self._finalize_soft_timeout(
                    csv_upload_id,
                    "Phase 5 (pre-enterprise optimization)",
                    metrics,
                    pipeline_start,
                    all_recommendations,
                )
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(result.get("recommendations", [])),
                        success=False,
                    )
                    pipeline_finished = True
                except Exception as exc:
                    logger.warning("[%s] Failed to finish pipeline metrics on soft timeout: %s", csv_upload_id, exc)
                return result

            # Phase 5a: Global Enterprise Optimization (across all bundle types)
            if self.enable_enterprise_optimization and all_recommendations and len(all_recommendations) > 10:
                phase_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 5a: Enterprise Optimization - STARTED | input_bundles={len(all_recommendations)}")

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

                    phase_duration = int((time.time() - phase_start) * 1000)
                    logger.info(f"[{csv_upload_id}] Phase 5a: Enterprise Optimization - COMPLETED in {phase_duration}ms | "
                               f"output_bundles={len(all_recommendations)} "
                               f"constraints_applied={len(global_constraints)}")

                    metrics["global_enterprise_optimization"] = {
                        "enabled": True,
                        "input_recommendations": len(all_recommendations),
                        "pareto_solutions": len(all_recommendations),
                        "duration_ms": phase_duration,
                        "global_optimization_metrics": global_optimization_result.get("metrics", {})
                    }
                    metrics["phase_timings"]["phase_5a_optimization"] = phase_duration

                except Exception as e:
                    phase_duration = int((time.time() - phase_start) * 1000)
                    logger.warning(f"[{csv_upload_id}] Phase 5a: Enterprise Optimization - FAILED in {phase_duration}ms | error={str(e)}")
                    metrics["global_enterprise_optimization"] = {"enabled": True, "error": str(e), "duration_ms": phase_duration}
            
            # Phase 5b: Weighted Ranking (fallback or when enterprise optimization disabled)
            elif self.enable_weighted_ranking and all_recommendations:
                phase_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 5b: Weighted Ranking - STARTED (fallback) | bundles={len(all_recommendations)}")
                # Rank all recommendations together for global optimization
                ranked_recommendations = await self.ranker.rank_bundle_recommendations(
                    all_recommendations, "increase_aov", csv_upload_id  # Use default objective for global ranking
                )
                all_recommendations = ranked_recommendations
                phase_duration = int((time.time() - phase_start) * 1000)
                logger.info(f"[{csv_upload_id}] Phase 5b: Weighted Ranking - COMPLETED in {phase_duration}ms")
                metrics["weighted_ranking"] = {"enabled": True, "duration_ms": phase_duration}
                metrics["phase_timings"]["phase_5b_ranking"] = phase_duration

            else:
                metrics["global_enterprise_optimization"] = {"enabled": False}
                metrics["weighted_ranking"] = {"enabled": False}

            await self.store_partial_recommendations(
                all_recommendations, csv_upload_id, stage="phase_5"
            )
            checkpoint = await self._record_checkpoint(
                csv_upload_id,
                "phase_5_optimization_completed",
                bundle_count=len(all_recommendations),
                metrics_snapshot={
                    "weighted_ranking": metrics.get("weighted_ranking"),
                    "enterprise_optimization": metrics.get("global_enterprise_optimization"),
                },
            )
            await update_generation_progress(
                csv_upload_id,
                step="optimization",
                progress=82,
                status="in_progress",
                message="Optimization pass complete.",
                bundle_count=len(all_recommendations) if all_recommendations else None,
                metadata={"checkpoint": checkpoint},
            )

            self._warn_if_time_low(end_time, csv_upload_id, "Phase 5c (pre-fallback injection)")
            if self._time_budget_exceeded(end_time):
                result = await self._finalize_soft_timeout(
                    csv_upload_id,
                    "Phase 5c (pre-fallback injection)",
                    metrics,
                    pipeline_start,
                    all_recommendations,
                )
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(result.get("recommendations", [])),
                        success=False,
                    )
                    pipeline_finished = True
                except Exception as exc:
                    logger.warning("[%s] Failed to finish pipeline metrics on soft timeout: %s", csv_upload_id, exc)
                return result

            # Phase 5c: Ensure minimum pair coverage via forced fallbacks
            phase_start = time.time()
            bundles_before = len(all_recommendations)
            all_recommendations = await self._apply_forced_pair_fallbacks(all_recommendations, csv_upload_id)
            injected = len(all_recommendations) - bundles_before
            if injected > 0:
                phase_duration = int((time.time() - phase_start) * 1000)
                logger.info(f"[{csv_upload_id}] Phase 5c: Fallback Injection - COMPLETED in {phase_duration}ms | injected={injected}")
                metrics["phase_timings"]["phase_5c_fallback"] = phase_duration
                await self.store_partial_recommendations(
                    all_recommendations, csv_upload_id, stage="phase_5c"
                )
                checkpoint = await self._record_checkpoint(
                    csv_upload_id,
                    "phase_5_fallback_completed",
                    bundle_count=len(all_recommendations),
                    metrics_snapshot={"fallback_injected": injected},
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="optimization",
                    progress=83,
                    status="in_progress",
                    message=f"Fallback bundles injected (+{injected}).",
                    bundle_count=len(all_recommendations),
                    metadata={"checkpoint": checkpoint},
                )

            self._warn_if_time_low(end_time, csv_upload_id, "Phase 6 (pre-explainability)")
            if self._time_budget_exceeded(end_time):
                result = await self._finalize_soft_timeout(
                    csv_upload_id,
                    "Phase 6 (pre-explainability)",
                    metrics,
                    pipeline_start,
                    all_recommendations,
                )
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(result.get("recommendations", [])),
                        success=False,
                    )
                    pipeline_finished = True
                except Exception as exc:
                    logger.warning("[%s] Failed to finish pipeline metrics on soft timeout: %s", csv_upload_id, exc)
                return result

            # Phase 6: Explainability
            if self.enable_explainability and all_recommendations:
                phase_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 6: Explainability - STARTED | bundles={len(all_recommendations)}")
                for recommendation in all_recommendations:
                    explanation = self.explainer.generate_explanation(recommendation)
                    recommendation["explanation"] = explanation

                    if self.enable_explainability:
                        detailed_explanation = self.explainer.generate_detailed_explanation(recommendation)
                        recommendation["detailed_explanation"] = detailed_explanation

                phase_duration = int((time.time() - phase_start) * 1000)
                logger.info(f"[{csv_upload_id}] Phase 6: Explainability - COMPLETED in {phase_duration}ms")
                metrics["explainability"] = {"enabled": True, "duration_ms": phase_duration}
                metrics["phase_timings"]["phase_6_explainability"] = phase_duration
            
            await update_generation_progress(
                csv_upload_id,
                step="optimization",
                progress=85,
                status="in_progress",
                message="Optimization complete. Preparing AI descriptions…",
            )
            await self._emit_heartbeat(
                csv_upload_id,
                step="optimization",
                progress=85,
                message="Preparing AI descriptions…",
            )

            self._warn_if_time_low(end_time, csv_upload_id, "Phase 7 (pre-pricing/finalization)")
            if self._time_budget_exceeded(end_time):
                result = await self._finalize_soft_timeout(
                    csv_upload_id,
                    "Phase 7 (pre-pricing/finalization)",
                    metrics,
                    pipeline_start,
                    all_recommendations,
                )
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(result.get("recommendations", [])),
                        success=False,
                    )
                except Exception as exc:
                    logger.warning("[%s] Failed to finish pipeline metrics on soft timeout: %s", csv_upload_id, exc)
                pipeline_finished = True
                return result

            # Phase 7: Pricing (part of finalization)
            pricing_start = time.time()
            logger.info(f"[{csv_upload_id}] Phase 7: Pricing & Finalization - STARTED | bundles={len(all_recommendations)}")

            # Phase 8: AI Copy Generation & Phase 9: Storage happen in finalize_recommendations
            final_recommendations = await self.finalize_recommendations(
                all_recommendations,
                csv_upload_id,
                metrics,
                end_time=end_time,
            )

            finalization_duration = int((time.time() - pricing_start) * 1000)
            logger.info(f"[{csv_upload_id}] Phase 7-9: Pricing, AI Copy & Storage - COMPLETED in {finalization_duration}ms")
            metrics["phase_timings"]["phase_7_9_finalization"] = finalization_duration
            
            # Update metrics
            metrics["total_recommendations"] = len(final_recommendations)
            total_pipeline_duration = int((time.time() - pipeline_start) * 1000)
            metrics["processing_time_ms"] = total_pipeline_duration

            # Add loop prevention statistics to metrics
            metrics["loop_prevention_stats"] = self.generation_stats.copy()

            # Count by bundle type
            for rec in final_recommendations:
                bundle_type = rec.get("bundle_type", "UNKNOWN")
                if bundle_type in metrics["bundle_counts"]:
                    metrics["bundle_counts"][bundle_type] += 1

            # Log final comprehensive summary
            logger.info(f"[{csv_upload_id}] ========== BUNDLE GENERATION PIPELINE COMPLETED ==========")
            logger.info(f"[{csv_upload_id}] Total Duration: {total_pipeline_duration}ms ({total_pipeline_duration/1000:.1f}s)")
            logger.info(f"[{csv_upload_id}] Bundles Generated: {len(final_recommendations)} total")
            logger.info(f"[{csv_upload_id}] Bundle Types: FBT={metrics['bundle_counts']['FBT']} "
                       f"VOLUME={metrics['bundle_counts']['VOLUME_DISCOUNT']} "
                       f"MIX_MATCH={metrics['bundle_counts']['MIX_MATCH']} "
                       f"BXGY={metrics['bundle_counts']['BXGY']} "
                       f"FIXED={metrics['bundle_counts']['FIXED']}")
            logger.info(f"[{csv_upload_id}] Generation Stats: attempts={self.generation_stats['total_attempts']} "
                       f"successes={self.generation_stats['successful_generations']} "
                       f"duplicates_skipped={self.generation_stats['skipped_duplicates']} "
                       f"failures={self.generation_stats['failed_attempts']}")
            logger.info(f"[{csv_upload_id}] Unique SKU Combinations Processed: {len(self.seen_sku_combinations)}")

            # Log phase breakdown
            logger.info(f"[{csv_upload_id}] Phase Timing Breakdown:")
            for phase_name, duration in sorted(metrics["phase_timings"].items()):
                percentage = (duration / total_pipeline_duration * 100) if total_pipeline_duration > 0 else 0
                logger.info(f"[{csv_upload_id}]   - {phase_name}: {duration}ms ({percentage:.1f}%)")

            if not pipeline_finished:
                try:
                    metrics_collector.record_phase_timings(metrics.get("phase_timings", {}))
                except Exception as exc:
                    logger.warning("[%s] Failed to record phase timings: %s", csv_upload_id, exc)
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(final_recommendations),
                        success=True,
                    )
                    pipeline_finished = True
                except Exception as exc:
                    logger.warning("[%s] Failed to record final pipeline metrics: %s", csv_upload_id, exc)

            await notify_bundle_ready(csv_upload_id, len(final_recommendations), resume_used)

            return {
                "recommendations": final_recommendations,
                "metrics": metrics,
                "v2_pipeline": True,
                "csv_upload_id": csv_upload_id
            }
            
        except Exception as e:
            # Calculate total time even on failure
            total_pipeline_duration = int((time.time() - pipeline_start) * 1000)

            logger.error(f"[{csv_upload_id}] ========== BUNDLE GENERATION PIPELINE FAILED ==========")
            logger.error(f"[{csv_upload_id}] Error: {str(e)}")
            logger.error(f"[{csv_upload_id}] Total Duration Before Failure: {total_pipeline_duration}ms ({total_pipeline_duration/1000:.1f}s)")

            # Log phase breakdown even on failure
            if metrics.get("phase_timings"):
                logger.info(f"[{csv_upload_id}] Phase Timing Breakdown (Before Failure):")
                for phase_name, duration in sorted(metrics["phase_timings"].items()):
                    percentage = (duration / total_pipeline_duration * 100) if total_pipeline_duration > 0 else 0
                    logger.info(f"[{csv_upload_id}]   - {phase_name}: {duration}ms ({percentage:.1f}%)")

            # Log generation stats
            logger.info(f"[{csv_upload_id}] Generation Stats (Before Failure): attempts={self.generation_stats['total_attempts']} "
                       f"successes={self.generation_stats['successful_generations']} "
                       f"duplicates_skipped={self.generation_stats['skipped_duplicates']} "
                       f"failures={self.generation_stats['failed_attempts']}")

            logger.error(f"[{csv_upload_id}] ================================================================")

            if not pipeline_finished:
                try:
                    if metrics.get("phase_timings"):
                        metrics_collector.record_phase_timings(metrics["phase_timings"])
                except Exception as exc:
                    logger.warning("[%s] Failed to record phase timings on failure: %s", csv_upload_id, exc)
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        metrics.get("total_recommendations", 0),
                        success=False,
                    )
                    pipeline_finished = True
                except Exception as exc:
                    logger.warning("[%s] Failed to record failure metrics: %s", csv_upload_id, exc)

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=100,
                status="failed",
                message=f"Bundle generation failed: {e}",
            )
            # Fallback to v1 pipeline if v2 fails
            if self.enable_v2_pipeline:
                logger.info("Falling back to v1 pipeline due to v2 error")
                return await self.generate_v1_recommendations(csv_upload_id, metrics, error=str(e))
            else:
                raise
    
    async def generate_objective_bundles(
        self,
        csv_upload_id: str,
        objective: str,
        bundle_type: str,
        metrics: Dict[str, Any],
        end_time: datetime = None,
        context: Optional[CandidateGenerationContext] = None,
    ) -> List[Dict[str, Any]]:
        """Generate bundles for a specific objective and bundle type with loop prevention"""
        objective_type_key = f"{objective}_{bundle_type}"
        attempts_for_this_combo = 0

        # DETAILED TIMING: Track this individual task
        task_start = time.time()
        logger.info(f"[{csv_upload_id}] Task {objective_type_key} - STARTED")

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

            ml_start = time.time()
            if self.enable_ml_candidates:
                try:
                    logger.info(f"[{csv_upload_id}] Task {objective_type_key} - ML candidate generation STARTED")
                    candidate_result = await self.candidate_generator.generate_candidates(
                        csv_upload_id,
                        bundle_type,
                        objective,
                        context=context,
                    )
                    ml_duration = int((time.time() - ml_start) * 1000)
                    logger.info(f"[{csv_upload_id}] Task {objective_type_key} - ML candidate generation COMPLETED in {ml_duration}ms")

                    candidates = candidate_result.get("candidates", [])
                    metrics["ml_candidates"] = {"enabled": True, "metrics": candidate_result.get("metrics", {})}

                    if candidates:
                        self.generation_stats['successful_generations'] += 1
                        logger.info(f"[{csv_upload_id}] Task {objective_type_key} - Generated {len(candidates)} ML candidates")
                    else:
                        self.generation_stats['failed_attempts'] += 1
                        logger.debug(f"No ML candidates generated for {objective_type_key}")

                except Exception as e:
                    ml_duration = int((time.time() - ml_start) * 1000)
                    logger.warning(f"[{csv_upload_id}] Task {objective_type_key} - ML candidate generation FAILED after {ml_duration}ms: {e}")
                    candidates = []
                    self.generation_stats['failed_attempts'] += 1
            else:
                candidates = []
                logger.debug(f"ML candidates disabled for {objective_type_key}")
            
            # Phase 3a.1: FallbackLadder for Small Shops (when insufficient candidates)
            min_candidates_threshold = 5  # Minimum candidates for adequate recommendations
            total_order_lines = metrics.get("total_order_lines", 0)

            # Skip FallbackLadder for very small datasets (< 10 order lines) as it's too slow and won't generate useful bundles
            if len(candidates) < min_candidates_threshold and total_order_lines >= 10:
                fallback_start = time.time()
                logger.info(f"[{csv_upload_id}] Task {objective_type_key} - FallbackLadder STARTED | current_candidates={len(candidates)} order_lines={total_order_lines}")
                try:
                    fallback_candidates = await self.fallback_ladder.generate_candidates(
                        csv_upload_id=csv_upload_id,
                        objective=objective,
                        bundle_type=bundle_type,
                        target_n=10
                    )
                    fallback_duration = int((time.time() - fallback_start) * 1000)
                    logger.info(f"[{csv_upload_id}] Task {objective_type_key} - FallbackLadder COMPLETED in {fallback_duration}ms | generated={len(fallback_candidates)}")

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
                    fallback_duration = int((time.time() - fallback_start) * 1000)
                    logger.warning(f"[{csv_upload_id}] Task {objective_type_key} - FallbackLadder FAILED after {fallback_duration}ms: {e}")
                    metrics["fallback_ladder"] = {"activated": True, "error": str(e)}
            elif len(candidates) < min_candidates_threshold and total_order_lines < 10:
                logger.info(f"[{csv_upload_id}] Task {objective_type_key} - Skipping FallbackLadder | dataset_too_small={total_order_lines} order_lines")
                metrics["fallback_ladder"] = {"activated": False, "reason": "dataset_too_small", "order_lines": total_order_lines}
            else:
                logger.info(f"[{csv_upload_id}] Task {objective_type_key} - Skipping FallbackLadder | sufficient_candidates={len(candidates)}")
                metrics["fallback_ladder"] = {"activated": False, "reason": "sufficient_candidates"}
            
            # Convert candidates to recommendations format with duplicate checking
            conversion_start = time.time()
            if candidates:
                logger.info(f"[{csv_upload_id}] Task {objective_type_key} - Converting {len(candidates)} candidates to recommendations")
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
            if (self.enable_enterprise_optimization and recommendations and
                    len(recommendations) >= self.min_candidates_for_optimization):
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
            elif self.enable_enterprise_optimization and recommendations:
                metrics[f"enterprise_optimization_{bundle_type}"] = {
                    "enabled": False,
                    "reason": "not_enough_candidates",
                    "candidate_count": len(recommendations)
                }
            
            # DETAILED TIMING: Task completion with full breakdown
            task_duration = int((time.time() - task_start) * 1000)
            conversion_duration = int((time.time() - conversion_start) * 1000) if candidates else 0

            logger.info(f"[{csv_upload_id}] Task {objective_type_key} - COMPLETED in {task_duration}ms | "
                       f"recommendations={len(recommendations)} attempts={attempts_for_this_combo} "
                       f"conversion_time={conversion_duration}ms")

            return recommendations

        except Exception as e:
            task_duration = int((time.time() - task_start) * 1000)
            logger.error(f"[{csv_upload_id}] Task {objective_type_key} - FAILED after {task_duration}ms: {e}")
            self.generation_stats['failed_attempts'] += 1
            return []
    
    async def finalize_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        metrics: Dict[str, Any],
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Finalize recommendations and prepare for storage"""
        try:
            final_recommendations = []
            
            # Limit recommendations per objective/type to avoid overwhelming merchants
            max_per_type_flag = feature_flags.get_flag("bundling.max_per_bundle_type", 15)
            try:
                max_per_type = int(max_per_type_flag)
            except (TypeError, ValueError):
                max_per_type = 15
            if max_per_type <= 0:
                max_per_type = float('inf')
            recommendations_by_type = {}
            
            for rec in recommendations:
                key = f"{rec.get('objective', 'unknown')}_{rec.get('bundle_type', 'unknown')}"
                if key not in recommendations_by_type:
                    recommendations_by_type[key] = []
                
                if len(recommendations_by_type[key]) < max_per_type:
                    recommendations_by_type[key].append(rec)
            
            # Flatten back to single list while limiting bundles per SKU pair
            pair_cap_flag = feature_flags.get_flag("bundling.max_per_pair", self.max_bundles_per_pair)
            try:
                pair_cap = int(pair_cap_flag)
            except (TypeError, ValueError):
                pair_cap = self.max_bundles_per_pair
            if pair_cap <= 0:
                pair_cap = float('inf')

            pair_usage = defaultdict(int)

            for type_recs in recommendations_by_type.values():
                for rec in type_recs:
                    products = rec.get("products", [])
                    if isinstance(products, list) and products:
                        pair_key = tuple(sorted(products))
                    else:
                        pair_key = (rec.get("id"),)

                    if pair_usage[pair_key] >= pair_cap:
                        continue

                    pair_usage[pair_key] += 1
                    final_recommendations.append(rec)
            
            bundle_total = len(final_recommendations)
            await update_generation_progress(
                csv_upload_id,
                step="ai_descriptions",
                progress=90,
                status="in_progress",
                message="Generating AI descriptions…" if bundle_total else "No bundles to describe.",
                bundle_count=bundle_total if bundle_total else None,
            )
            
            # Add AI-generated copy if available (BATCHED for speed)
            if final_recommendations and not self._time_budget_exceeded(end_time):
                ai_copy_start = time.time()
                batch_size = 5  # Process 5 bundles at a time to avoid rate limits
                bundles_for_copy = final_recommendations[:10]  # Limit AI copy generation for cost

                logger.info(f"[{csv_upload_id}] Phase 8: AI Copy Generation - STARTED | bundles={len(bundles_for_copy)} batch_size={batch_size}")

                def _normalize_products(raw_products: Any) -> List[Dict[str, Any]]:
                    if isinstance(raw_products, list):
                        return raw_products
                    if raw_products:
                        return [raw_products]
                    return []

                for i in range(0, len(bundles_for_copy), batch_size):
                    batch_start = time.time()
                    batch = bundles_for_copy[i:i+batch_size]
                    span_attrs = {
                        "csv_upload_id": csv_upload_id,
                        "batch_index": i // batch_size,
                        "batch_size": len(batch),
                    }
                    with self._start_span("bundle_generator.ai_copy_batch", span_attrs):
                        # Generate AI copy for batch in parallel
                        try:
                            copy_tasks = []
                            copy_inputs = []
                            for rec in batch:
                                bundle_type = rec.get("bundle_type") or rec.get("type") or "MIX_MATCH"
                                products = _normalize_products(rec.get("products"))
                                objective = rec.get("objective", "")
                                copy_tasks.append(
                                    self.ai_generator.generate_bundle_copy(
                                        products,
                                        bundle_type,
                                        objective,
                                    )
                                )
                                copy_inputs.append((rec, products, bundle_type))
                            copy_results = await asyncio.gather(*copy_tasks, return_exceptions=True)
                        except Exception as exc:
                            logger.exception("Error scheduling AI copy generation batch")
                            copy_inputs = [
                                (
                                    rec,
                                    _normalize_products(rec.get("products")),
                                    rec.get("bundle_type") or rec.get("type") or "MIX_MATCH",
                                )
                                for rec in batch
                            ]
                            copy_results = [exc] * len(copy_inputs)

                        # Assign results back to recommendations
                        for (rec, products, bundle_type), result in zip(copy_inputs, copy_results):
                            if isinstance(result, Exception):
                                logger.warning(f"Error generating AI copy for bundle type {bundle_type}: {result}")
                                rec["ai_copy"] = self.ai_generator.generate_fallback_copy(products, bundle_type)
                            elif not result:
                                logger.warning(f"Empty AI copy response for bundle type {bundle_type}; using fallback copy")
                                rec["ai_copy"] = self.ai_generator.generate_fallback_copy(products, bundle_type)
                            else:
                                rec["ai_copy"] = result

                        batch_duration = int((time.time() - batch_start) * 1000)
                        logger.info(f"[{csv_upload_id}] AI Copy batch {i//batch_size + 1}/{(len(bundles_for_copy) + batch_size - 1)//batch_size} completed in {batch_duration}ms")

                ai_copy_duration = int((time.time() - ai_copy_start) * 1000)
                logger.info(f"[{csv_upload_id}] Phase 8: AI Copy Generation - COMPLETED in {ai_copy_duration}ms")
                metrics["phase_timings"]["phase_8_ai_copy"] = ai_copy_duration

                await update_generation_progress(
                    csv_upload_id,
                    step="ai_descriptions",
                    progress=92,
                    status="in_progress",
                    message="AI descriptions ready.",
                    bundle_count=bundle_total,
                )
            elif final_recommendations:
                logger.warning(
                    f"[{csv_upload_id}] Skipping AI copy generation due to soft time budget proximity."
                )
                metrics["phase_timings"]["phase_8_ai_copy"] = 0
                await update_generation_progress(
                    csv_upload_id,
                    step="ai_descriptions",
                    progress=92,
                    status="in_progress",
                    message="Skipped AI descriptions due to time budget.",
                    bundle_count=bundle_total,
                )
            
            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=95,
                status="in_progress",
                message="Finalizing bundle recommendations…",
                bundle_count=bundle_total if bundle_total else None,
            )

            # Remove any partial recommendations before persisting the final set
            if csv_upload_id:
                await storage.delete_partial_bundle_recommendations(csv_upload_id)

            # Store recommendations in database
            if final_recommendations and csv_upload_id:
                storage_start = time.time()
                logger.info(f"[{csv_upload_id}] Phase 9: Database Storage - STARTED | bundles={len(final_recommendations)}")
                with self._start_span(
                    "bundle_generator.persist_recommendations",
                    {"csv_upload_id": csv_upload_id, "bundle_count": len(final_recommendations)},
                ):
                    await self.store_recommendations(final_recommendations, csv_upload_id)
                storage_duration = int((time.time() - storage_start) * 1000)
                logger.info(f"[{csv_upload_id}] Phase 9: Database Storage - COMPLETED in {storage_duration}ms")
                metrics["phase_timings"]["phase_9_storage"] = storage_duration

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=100,
                status="completed",
                message="Bundle generation complete.",
                bundle_count=bundle_total if bundle_total else None,
                time_remaining=0,
            )
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error finalizing recommendations: {e}")
            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=100,
                status="failed",
                message=f"Finalization error: {e}",
            )
            return recommendations  # Return original recommendations if finalization fails
    
    async def store_partial_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        stage: str = "phase_3",
    ) -> None:
        """Store partial recommendations during processing to prevent data loss"""
        try:
            if not recommendations:
                return

            # Select up to 50 top recommendations based on ranking/confidence
            sorted_recs = sorted(
                recommendations,
                key=lambda rec: rec.get("ranking_score", rec.get("confidence", 0.0)),
                reverse=True,
            )
            partial_recs = sorted_recs[:50]
            if not partial_recs:
                return

            await storage.delete_partial_bundle_recommendations(csv_upload_id)

            db_recommendations = []
            for index, rec in enumerate(partial_recs):
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
                        "shop_id": rec.get("shop_id"),
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
                        "discount_reference": f"__partial__:{stage}",
                        "is_approved": False,
                        "is_used": False,
                        "rank_position": index + 1,
                    }
                    
                    # Remove None values for optional fields
                    db_rec = {k: v for k, v in db_rec.items() if v is not None}
                    db_recommendations.append(db_rec)
                    
                except Exception as rec_error:
                    logger.warning(f"Error processing partial recommendation {rec.get('id', 'unknown')}: {rec_error}")
                    continue
            
            if db_recommendations:
                await storage.create_bundle_recommendations(db_recommendations)
                logger.info(
                    f"Stored {len(db_recommendations)} partial recommendations for {csv_upload_id} (stage={stage})"
                )
            
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
