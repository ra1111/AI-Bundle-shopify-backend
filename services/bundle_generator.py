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
import math
import time
from collections import defaultdict, Counter
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
from services.deadlines import Deadline

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
        self._initialize_caps_and_safeguards()

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
        if self._current_deadline and self._current_deadline.expired:
            return True
        return bool(end_time and datetime.now() >= end_time)

    def _deadline_remaining(self) -> Optional[float]:
        if not self._current_deadline:
            return None
        return self._current_deadline.remaining()

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
        time_remaining: Optional[float] = None
        if end_time:
            time_remaining = (end_time - datetime.now()).total_seconds()
            if time_remaining is not None and time_remaining < 0:
                time_remaining = 0.0
        dataset_profile["time_remaining_seconds"] = time_remaining

        tier = dataset_profile.get("tier")
        candidate_count = dataset_profile.get("candidate_count", 0)

        should_defer = False
        if tier == "large":
            should_defer = True
        elif not self.async_defer_large_tier_only:
            if candidate_count >= self.async_defer_candidate_threshold:
                should_defer = True
            elif (
                time_remaining is not None
                and time_remaining <= max(0, self.async_defer_min_time_remaining)
            ):
                should_defer = True

        dataset_profile["defer_candidate"] = should_defer

        if not self.async_defer_enabled or resume_used:
            return False

        return should_defer

    def _derive_allocation_plan(self, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
        plan = {
            "phase3_concurrency": self.phase3_concurrency_limit,
            "llm_candidate_target": 20,
        }

        tier = dataset_profile.get("tier")
        if tier == "small":
            plan["phase3_concurrency"] = max(2, min(self.phase3_concurrency_limit, 3))
            plan["llm_candidate_target"] = 9
        elif tier == "medium":
            plan["phase3_concurrency"] = min(self.phase3_concurrency_limit, 5)
            plan["llm_candidate_target"] = 15
        elif tier == "large":
            plan["phase3_concurrency"] = max(self.phase3_concurrency_limit, 8)
            plan["llm_candidate_target"] = 24
        return plan

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return default

    @staticmethod
    def _coerce_int(value: Any, default: int = 0, minimum: Optional[int] = None) -> int:
        result = default
        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default
        if minimum is not None and result < minimum:
            result = minimum
        return result

    @staticmethod
    def _coerce_float(
        value: Any,
        default: float = 0.0,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = float(default)
        if minimum is not None and result < minimum:
            result = minimum
        if maximum is not None and result > maximum:
            result = maximum
        return result

    def _extract_sku_list(self, recommendation: Dict[str, Any]) -> List[str]:
        """
        Normalize the products payload on a recommendation into a list of SKUs.
        Handles both simple string lists and richer dict payloads.
        """
        skus: List[str] = []
        products = recommendation.get("products") or []
        for entry in products:
            candidate: Optional[str] = None
            if isinstance(entry, dict):
                candidate = (
                    entry.get("sku")
                    or entry.get("id")
                    or entry.get("variant_id")
                    or entry.get("product_id")
                )
            else:
                candidate = str(entry)

            if candidate:
                candidate_str = str(candidate).strip()
                if candidate_str:
                    skus.append(candidate_str)
        return skus

    def _parse_stage_thresholds(self, raw: Any) -> List[int]:
        if isinstance(raw, list):
            candidates = raw
        elif isinstance(raw, str):
            stripped = raw.strip().strip("[]")
            if not stripped:
                candidates = []
            else:
                candidates = [token.strip() for token in stripped.split(",")]
        else:
            candidates = []

        thresholds: List[int] = []
        for token in candidates:
            try:
                num = int(token)
                if num > 0:
                    thresholds.append(num)
            except (TypeError, ValueError):
                continue

        if not thresholds:
            thresholds = [3, 5, 10, 20, 40]
        thresholds = sorted(set(thresholds))
        return thresholds

    async def _load_staged_wave_state(
        self,
        csv_upload_id: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        default_state: Dict[str, Any] = {
            "version": 1,
            "waves": [],
            "totals": {"published": 0, "dropped": 0},
            "cursor": {"stage_idx": 0, "published": 0, "last_bundle_id": None},
            "backpressure": {"active": False, "reason": None, "last_event": None},
            "resume": {},
            "last_finalize_tx": None,
        }
        metrics_state: Dict[str, Any] = {}
        try:
            upload = await storage.get_csv_upload(csv_upload_id)
        except Exception as exc:
            logger.warning(
                "[%s] Unable to load staged wave state (get_csv_upload failed): %s",
                csv_upload_id,
                exc,
            )
            return default_state, metrics_state

        raw_metrics = {}
        if upload and getattr(upload, "bundle_generation_metrics", None):
            try:
                raw_metrics = dict(upload.bundle_generation_metrics)
            except Exception:
                raw_metrics = upload.bundle_generation_metrics or {}
        metrics_state = raw_metrics

        staged_section = raw_metrics.get("staged_wave_state") or {}
        merged_state = default_state.copy()
        for key, value in staged_section.items():
            if key in {"waves", "totals", "cursor", "backpressure", "resume"} and isinstance(
                value, dict | list
            ):
                merged_state[key] = value
            elif key in {"version", "last_finalize_tx"}:
                merged_state[key] = value

        # Ensure waves list sorted by index
        waves = merged_state.get("waves", [])
        if isinstance(waves, list):
            try:
                waves = sorted(waves, key=lambda item: item.get("index", 0))
            except (TypeError, AttributeError) as e:
                # Wave items might not be dicts or might not have 'get' method
                logger.warning(f"Could not sort waves by index: {e}. Using unsorted list.")
            except Exception as e:
                logger.error(f"Unexpected error sorting waves: {e}")
        else:
            waves = []
            if waves != []:  # Only log if waves was not already empty
                logger.debug(f"Waves is not a list (type: {type(waves)}), using empty list")
        merged_state["waves"] = waves

        # Normalise totals / cursor
        totals = merged_state.get("totals") or {}
        merged_state["totals"] = {
            "published": int(totals.get("published", 0) or 0),
            "dropped": int(totals.get("dropped", 0) or 0),
        }
        cursor = merged_state.get("cursor") or {}
        merged_state["cursor"] = {
            "stage_idx": int(cursor.get("stage_idx", len(waves)) or 0),
            "published": int(cursor.get("published", merged_state["totals"]["published"]) or 0),
            "last_bundle_id": cursor.get("last_bundle_id"),
        }
        backpressure = merged_state.get("backpressure") or {}
        merged_state["backpressure"] = {
            "active": bool(backpressure.get("active", False)),
            "reason": backpressure.get("reason"),
            "last_event": backpressure.get("last_event"),
        }

        try:
            logger.info(
                "[%s] Loaded staged state | waves=%d published=%d dropped=%d cursor=%s",
                csv_upload_id,
                len(merged_state["waves"]),
                merged_state["totals"]["published"],
                merged_state["totals"]["dropped"],
                merged_state["cursor"],
            )
        except Exception:
            logger.debug("[%s] Loaded staged state (summary logging failed)", csv_upload_id)

        return merged_state, metrics_state

    async def _persist_staged_wave_state(
        self,
        csv_upload_id: str,
        staged_state: Dict[str, Any],
        metrics_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        metrics_payload = dict(metrics_state or {})
        metrics_payload["staged_wave_state"] = staged_state
        try:
            await storage.update_csv_upload(
                csv_upload_id,
                {"bundle_generation_metrics": metrics_payload},
            )
            logger.info(
                "[%s] Persisted staged state | waves=%d totals=%s cursor=%s finalize_tx=%s",
                csv_upload_id,
                len(staged_state.get("waves", [])),
                staged_state.get("totals"),
                staged_state.get("cursor"),
                staged_state.get("last_finalize_tx"),
            )
        except Exception as exc:
            logger.warning(
                "[%s] Failed to persist staged wave state: %s", csv_upload_id, exc
            )

    def _normalize_drop_reason(self, raw_reason: str, origin: str) -> str:
        if not raw_reason:
            return "FINALIZE_TX_FAIL"
        key = raw_reason.lower()
        origin_key = origin.lower()

        if key in {"out_of_stock", "inventory_out"}:
            return "OUT_OF_STOCK"
        if key in {"missing_catalog", "missing_variant"}:
            return "FINALIZE_TX_FAIL"
        if key in {"inactive_product", "policy_violation"}:
            return "POLICY_BLOCK"
        if key in {"below_margin", "margin_violation"}:
            return "BELOW_MARGIN"
        if key in {"duplicate_sku", "duplicate"}:
            return "DUPLICATE_SKU"
        if key in {"score_low", "low_signal"}:
            return "LOW_SCORE"
        if key in {"pricing_error", "pricing_fallback"}:
            return "PRICE_ANOMALY"
        if key in {"copy_error", "llm_failure"} or origin_key == "copy":
            return "COPY_FAIL"
        if key in {"tx_fail", "persist_fail"}:
            return "FINALIZE_TX_FAIL"
        return "FINALIZE_TX_FAIL"

    def _normalize_drop_reasons(
        self,
        drop_map: Dict[str, int],
        origin: str,
    ) -> Dict[str, int]:
        normalized: Dict[str, int] = defaultdict(int)
        for reason, count in (drop_map or {}).items():
            normalized_reason = self._normalize_drop_reason(reason, origin)
            normalized[normalized_reason] += int(count or 0)
        return dict(normalized)

    def _build_staged_progress_payload(
        self,
        run_id: str,
        staged_state: Dict[str, Any],
        *,
        next_eta_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        waves_payload: List[Dict[str, Any]] = []
        for wave in staged_state.get("waves", []) or []:
            waves_payload.append(
                {
                    "index": wave.get("index"),
                    "target": wave.get("target"),
                    "published": wave.get("published"),
                    "drops": wave.get("drops", {}),
                    "took_ms": wave.get("duration_ms"),
                    "finalize_tx": wave.get("finalize_tx"),
                }
            )
        totals = staged_state.get("totals", {"published": 0, "dropped": 0})
        cursor = staged_state.get("cursor", {})

        payload = {
            "run_id": run_id,
            "staged": True,
            "waves": waves_payload,
            "totals": {
                "published": totals.get("published", 0),
                "dropped": totals.get("dropped", 0),
            },
            "cursor": {
                "stage_idx": cursor.get("stage_idx"),
                "published": cursor.get("published"),
                "last_bundle_id": cursor.get("last_bundle_id"),
            },
            "backpressure": staged_state.get(
                "backpressure", {"active": False, "reason": None}
            ),
            "next_wave_eta_sec": next_eta_seconds,
        }
        resume = staged_state.get("resume")
        if resume:
            payload["resume"] = resume
        return payload

    def _warn_if_time_low(self, end_time: Optional[datetime], csv_upload_id: str, phase_label: str) -> None:
        if self.soft_timeout_warning_seconds <= 0:
            return

        remaining: Optional[float] = None
        if self._current_deadline and not self._current_deadline.expired:
            remaining = self._current_deadline.remaining()
        elif end_time:
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

        # Create user-friendly timeout message based on phase
        phase_messages = {
            "enrichment": "We couldn't complete data analysis in time. Your store may have a large catalog.",
            "scoring": "Product scoring took longer than expected due to catalog size.",
            "ml_generation": "ML analysis timed out. This can happen with large transaction histories.",
            "optimization": "Bundle optimization took too long. Try again or reduce catalog size.",
            "ai_descriptions": "AI description generation timed out.",
            "staged_publish": "Publishing bundles took longer than expected.",
            "finalization": "Final processing timed out.",
        }
        user_message = phase_messages.get(
            phase_name,
            f"Bundle generation stopped during {phase_name}. Please try again."
        )
        if metrics["total_recommendations"] > 0:
            user_message += f" However, we managed to generate {metrics['total_recommendations']} bundle(s) before timeout."

        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=user_message,
            bundle_count=metrics["total_recommendations"] or None,
            metadata={
                "checkpoint": {"phase": phase_name, "timestamp": datetime.utcnow().isoformat()},
                "timeout_error": True,
                "timeout_phase": phase_name,
            },
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
        # Bundle configuration - MUST match frontend expected types
        # Frontend expects exactly: FBT, VOLUME, BOGO
        self.bundle_types = ['FBT', 'VOLUME', 'BOGO']

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
        # Frontend bundle types: FBT (Frequently Bought Together), VOLUME (quantity discounts), BOGO (Buy X Get Y)
        self.objective_to_bundle_types = {
            # Top priority objectives (Pareto 80/20)
            'margin_guard': ['FBT'],                       # Protect margins: FBT works best
            'clear_slow_movers': ['VOLUME', 'BOGO'],       # Move inventory: Volume discounts & BOGO
            'increase_aov': ['FBT', 'VOLUME'],             # Boost AOV: FBT & volume bundles

            # Secondary objectives (only for large datasets)
            'new_launch': ['FBT'],                         # Promote new products: FBT exposure
            'seasonal_promo': ['BOGO'],                    # Seasonal: BOGO promotions

            # Low priority (rarely used)
            'category_bundle': ['FBT'],                    # Cross-category: FBT bundles
            'gift_box': ['FBT'],                           # Gift boxes: FBT bundles
            'subscription_push': ['VOLUME'],               # Subscription: Volume discounts
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
        self.enable_pareto_optimization = feature_flags.get_flag("advanced.pareto_optimization", False)  # MODERN: Disabled by default for speed
        
        # Advanced feature flags (PR-5, PR-6, PR-8)
        self.enable_normalized_ranking = True
        self.enable_cold_start_coverage = True
        self.enable_observability = True
        
        # Loop prevention and performance safeguards
        self.max_total_attempts = 500  # Hard cap on total attempts across all objectives/types
        self.max_time_budget_seconds = 300  # 5 minutes maximum processing time
        self.soft_timeout_seconds = max(
            1,
            min(
                self.max_time_budget_seconds,
                int(os.getenv("SOFT_TIMEOUT_SECONDS", str(self.max_time_budget_seconds - 30))),
            ),
        )
        self._current_deadline: Optional[Deadline] = None
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

    def _enrich_bundle_with_type_structure(
        self,
        recommendation: Dict[str, Any],
        catalog: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Enrich a bundle recommendation with type-specific structure.

        Frontend expects:
        - FBT: products array with product details
        - VOLUME: volume_tiers array with tier configuration
        - BOGO: bogo_config, qualifiers, rewards

        This ensures full generation outputs match quick-start structure.
        """
        bundle_type = recommendation.get("bundle_type", "FBT")
        products = recommendation.get("products", [])
        pricing = recommendation.get("pricing", {})

        # Ensure ai_copy exists
        if "ai_copy" not in recommendation:
            recommendation["ai_copy"] = {
                "title": f"{bundle_type} Bundle",
                "description": f"AI-generated {bundle_type} bundle recommendation",
            }

        if bundle_type == "VOLUME":
            # Add volume_tiers structure for VOLUME bundles
            volume_tiers = [
                {"min_qty": 1, "discount_type": "NONE", "discount_value": 0, "label": None, "type": "percentage", "value": 0},
                {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5, "label": "Starter Pack", "type": "percentage", "value": 5},
                {"min_qty": 3, "discount_type": "PERCENTAGE", "discount_value": 10, "label": "Popular", "type": "percentage", "value": 10},
                {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15, "label": "Best Value", "type": "percentage", "value": 15},
            ]

            # Structure products for VOLUME
            if isinstance(products, list) and products:
                first_product = products[0] if isinstance(products[0], dict) else {"sku": products[0]}
                recommendation["products"] = {
                    "items": [first_product] if isinstance(first_product, dict) else [{"sku": first_product}],
                    "volume_tiers": volume_tiers,
                }

            # Add volume_tiers to pricing
            pricing["discount_type"] = "tiered"
            pricing["volume_tiers"] = volume_tiers
            recommendation["pricing"] = pricing

            # Top-level volume_tiers for frontend compatibility
            recommendation["volume_tiers"] = volume_tiers

        elif bundle_type == "BOGO":
            # Add BOGO structure
            bogo_config = {
                "buy_qty": 2,
                "get_qty": 1,
                "discount_type": "free",
                "discount_percent": 100,
                "same_product": True,
                "mode": "free_same_variant",
            }

            # Structure products for BOGO
            if isinstance(products, list) and products:
                first_product = products[0] if isinstance(products[0], dict) else {"sku": products[0]}
                product_data = first_product if isinstance(first_product, dict) else {"sku": first_product}

                recommendation["products"] = {
                    "items": [product_data],
                    "qualifiers": [{"quantity": 2, **product_data}],
                    "rewards": [{"quantity": 1, "discount_type": "free", "discount_percent": 100, **product_data}],
                }

            # Add bogo_config to pricing
            pricing["discount_type"] = "bogo"
            pricing["bogo_config"] = bogo_config
            recommendation["pricing"] = pricing

            # Top-level fields for frontend compatibility
            recommendation["bogo_config"] = bogo_config
            recommendation["qualifiers"] = recommendation.get("products", {}).get("qualifiers", [])
            recommendation["rewards"] = recommendation.get("products", {}).get("rewards", [])

        else:  # FBT (default)
            # FBT keeps products as array of product objects
            if isinstance(products, list):
                enriched_products = []
                for p in products:
                    if isinstance(p, dict):
                        enriched_products.append(p)
                    elif isinstance(p, str):
                        # SKU string - try to enrich from catalog
                        product_data = {"sku": p}
                        if catalog and p in catalog:
                            snap = catalog[p]
                            product_data.update({
                                "title": getattr(snap, "product_title", p),
                                "price": float(getattr(snap, "price", 0) or 0),
                                "image_url": getattr(snap, "image_url", None),
                                "variant_id": getattr(snap, "variant_id", None),
                            })
                        enriched_products.append(product_data)
                recommendation["products"] = enriched_products

        return recommendation

    def _initialize_caps_and_safeguards(self) -> None:
        """Initialize caps for optimization and diversity safeguards - called from _initialize_configuration"""
        # New caps for optimization and diversity safeguards
        self.min_candidates_for_optimization = 5
        self.max_bundles_per_pair = 2
        self.medium_dataset_txn_threshold = int(os.getenv("DATASET_MEDIUM_TXN_THRESHOLD", "200"))
        self.large_dataset_txn_threshold = int(os.getenv("DATASET_LARGE_TXN_THRESHOLD", "500"))
        self.medium_dataset_sku_threshold = int(os.getenv("DATASET_MEDIUM_SKU_THRESHOLD", "150"))
        self.large_dataset_sku_threshold = int(os.getenv("DATASET_LARGE_SKU_THRESHOLD", "350"))
        self.async_defer_enabled = os.getenv("BUNDLING_ASYNC_DEFER", "true").lower() != "false"
        self.async_defer_min_time_remaining = int(os.getenv("ASYNC_DEFER_MIN_TIME", "45"))
        self.async_defer_candidate_threshold = int(os.getenv("ASYNC_DEFER_CANDIDATE_THRESHOLD", "120"))
        self.async_defer_large_tier_only = os.getenv("ASYNC_DEFER_LARGE_ONLY", "false").lower() == "true"
        self.heartbeat_interval_seconds = int(os.getenv("BUNDLE_HEARTBEAT_SECONDS", "30"))
        self.soft_timeout_warning_seconds = int(os.getenv("BUNDLE_SOFT_TIMEOUT_WARNING", "45"))
        self.staged_publish_enabled = self._coerce_bool(
            feature_flags.get_flag("bundling.staged_publish_enabled", True), True
        )
        self.staged_thresholds = self._parse_stage_thresholds(
            feature_flags.get_flag("bundling.staged_thresholds", default=[3, 5, 10, 20, 40])
        )
        self.staged_hard_cap = self._coerce_int(
            feature_flags.get_flag("bundling.staged_hard_cap", 40), default=40, minimum=0
        )
        self.staged_prefer_high_score = self._coerce_bool(
            feature_flags.get_flag("bundling.staged_prefer_high_score", True), True
        )
        self.staged_cycle_interval_seconds = self._coerce_int(
            feature_flags.get_flag("bundling.staged_cycle_interval_seconds", 0),
            default=0,
            minimum=0,
        )
        wave_cooldown_ms = self._coerce_int(
            feature_flags.get_flag(
                "bundling.staged.wave_cooldown_ms",
                self.staged_cycle_interval_seconds * 1000,
            ),
            default=self.staged_cycle_interval_seconds * 1000,
            minimum=0,
        )
        self.staged_wave_cooldown_seconds = wave_cooldown_ms / 1000.0
        self.staged_auto_shrink_threshold = self._coerce_float(
            feature_flags.get_flag("bundling.staged.auto_shrink_drop_threshold", 0.6),
            default=0.6,
            minimum=0.0,
            maximum=1.0,
        )
        self.staged_auto_shrink_factor = self._coerce_float(
            feature_flags.get_flag("bundling.staged.auto_shrink_factor", 0.5),
            default=0.5,
            minimum=0.1,
            maximum=1.0,
        )
        self.staged_soft_guard_seconds = self._coerce_int(
            feature_flags.get_flag("bundling.staged.soft_guard_seconds", 45),
            default=45,
            minimum=0,
        )
        self.staged_wave_batch_size = self._coerce_int(
            feature_flags.get_flag("bundling.staged.wave_batch_size", 10),
            default=10,
            minimum=1,
        )
        self.staged_backpressure_queue_threshold = self._coerce_int(
            feature_flags.get_flag("bundling.staged.backpressure_queue_threshold", 0),
            default=0,
            minimum=0,
        )
        self.staged_backpressure_cooldown_waves = self._coerce_int(
            feature_flags.get_flag("bundling.staged.backpressure_cooldown_waves", 1),
            default=1,
            minimum=1,
        )
        self.finalize_track_concurrency = {
            "copy": self._coerce_int(
                feature_flags.get_flag("bundling.finalize.concurrent_tracks.copy", 3),
                default=3,
                minimum=1,
            ),
            "pricing": self._coerce_int(
                feature_flags.get_flag("bundling.finalize.concurrent_tracks.pricing", 2),
                default=2,
                minimum=1,
            ),
            "inventory": self._coerce_int(
                feature_flags.get_flag(
                    "bundling.finalize.concurrent_tracks.inventory", 3
                ),
                default=3,
                minimum=1,
            ),
        }
        if self.staged_cycle_interval_seconds == 0 and self.staged_wave_cooldown_seconds > 0:
            self.staged_cycle_interval_seconds = int(
                max(1, round(self.staged_wave_cooldown_seconds))
            )

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

    def _ensure_ai_copy_present(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure a recommendation has ai_copy; supply fallback if missing/invalid."""
        ai_copy = recommendation.get("ai_copy")
        if not ai_copy or not isinstance(ai_copy, dict):
            ai_copy = self.ai_generator.generate_fallback_copy(
                recommendation.get("products", []),
                recommendation.get("bundle_type", "FBT"),
            )
        # Ensure core fields exist for downstream rendering
        ai_copy.setdefault("title", "Bundle Deal")
        ai_copy.setdefault("description", "Great products bundled together for savings.")
        ai_copy.setdefault("valueProposition", "Save with this bundle.")
        recommendation["ai_copy"] = ai_copy
        return ai_copy

    async def _canonical_upload_id(self, upload_id: str) -> str:
        """
        Prefer the most recent orders upload for the run as canonical, else fall back to the provided id.
        This keeps downstream catalog/variant lookups aligned with ingestion that writes to the orders upload id.
        """
        try:
            run_id = await storage.get_run_id_for_upload(upload_id)
            if not run_id:
                return upload_id
            orders_upload = await storage.get_latest_orders_upload_for_run(run_id)
            if orders_upload and getattr(orders_upload, "id", None):
                orders_id = getattr(orders_upload, "id")
                if orders_id != upload_id:
                    logger.info(
                        "[%s] Canonical upload alignment: using orders upload %s for run %s",
                        upload_id,
                        orders_id,
                        run_id,
                    )
                return orders_id
        except Exception as exc:
            logger.warning("[%s] Canonical upload lookup failed: %s", upload_id, exc)
        return upload_id
    async def generate_quick_start_bundles(
        self,
        csv_upload_id: str,
        max_products: int = 50,
        max_bundles: int = 10,
        timeout_seconds: int = 120
    ) -> Dict[str, Any]:
        """Generate quick preview bundles for first-time installations.

        This is an optimized fast-path that:
        - Limits to top 50 products by sales volume
        - Uses only 1-2 high-priority objectives
        - Has aggressive 2-minute timeout
        - Generates 10 bundles for immediate preview
        - Skips expensive ML phases

        Args:
            csv_upload_id: Upload ID to process
            max_products: Maximum products to consider (default: 50)
            max_bundles: Maximum bundles to generate (default: 10)
            timeout_seconds: Hard timeout in seconds (default: 120)

        Returns:
            Dict with recommendations, metrics, and quick_start flag
        """
        import traceback
        
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")

        logger.info(
            f"[{csv_upload_id}] ========== QUICK-START BUNDLE GENERATION ==========\n"
            f"  Max products: {max_products}\n"
            f"  Max bundles: {max_bundles}\n"
            f"  Timeout: {timeout_seconds}s"
        )

        pipeline_start = time.time()

        # Set aggressive deadline
        self._current_deadline = Deadline(timeout_seconds)

        # Clean up any orphaned partial recommendations from previous failed runs
        try:
            await storage.delete_partial_bundle_recommendations(csv_upload_id)
            logger.debug(f"[{csv_upload_id}] Cleaned up any orphaned partial recommendations")
        except Exception as cleanup_err:
            logger.warning(f"[{csv_upload_id}] Failed to cleanup orphaned partials: {cleanup_err}")

        await update_generation_progress(
            csv_upload_id,
            step="enrichment",
            progress=10,
            status="in_progress",
            message="Quick-start: Loading top products…",
        )

        try:
            # PHASE 1: Load and enrich order data (simplified)
            logger.info(f"[{csv_upload_id}] 📦 PHASE 1: Loading order lines from database...")
            phase1_start = time.time()
            
            try:
                order_lines = await storage.get_order_lines(csv_upload_id)
                phase1_db_duration = (time.time() - phase1_start) * 1000
                logger.info(
                    f"[{csv_upload_id}] ✅ Database query completed in {phase1_db_duration:.0f}ms\n"
                    f"  Order lines loaded: {len(order_lines)}"
                )

                # EXTENSIVE LOGGING: What did we load from database?
                sku_list = [getattr(line, 'sku', None) for line in order_lines]
                variant_id_list = [getattr(line, 'variant_id', None) for line in order_lines]
                unique_skus_loaded = set(filter(None, sku_list))
                unique_variant_ids_loaded = set(filter(None, variant_id_list))
                none_sku_count = sum(1 for s in sku_list if not s)

                logger.info(f"[{csv_upload_id}] 📊 QUICK-START PHASE 1 - SKU ANALYSIS:")
                logger.info(f"[{csv_upload_id}]   Total order lines: {len(order_lines)}")
                logger.info(f"[{csv_upload_id}]   Order lines with SKU: {len(order_lines) - none_sku_count}")
                logger.info(f"[{csv_upload_id}]   Order lines with NULL/empty SKU: {none_sku_count}")
                logger.info(f"[{csv_upload_id}]   Unique SKUs found: {len(unique_skus_loaded)}")
                logger.info(f"[{csv_upload_id}]   Unique variant_ids found: {len(unique_variant_ids_loaded)}")
                logger.info(f"[{csv_upload_id}]   All SKUs from DB: {sorted(unique_skus_loaded)}")

                # Show SKU type breakdown
                no_sku_prefix = [s for s in unique_skus_loaded if s and s.startswith('no-sku-')]
                acc_prefix = [s for s in unique_skus_loaded if s and s.startswith('ACC-')]
                other_skus = [s for s in unique_skus_loaded if s and not s.startswith('no-sku-') and not s.startswith('ACC-')]

                if no_sku_prefix:
                    logger.warning(f"[{csv_upload_id}] ⚠️  {len(no_sku_prefix)} SKUs start with 'no-sku-' prefix: {no_sku_prefix}")
                if acc_prefix:
                    logger.info(f"[{csv_upload_id}]   {len(acc_prefix)} SKUs start with 'ACC-' prefix: {acc_prefix}")
                if other_skus:
                    logger.info(f"[{csv_upload_id}]   {len(other_skus)} SKUs with other formats: {other_skus}")

                # Log order structure
                order_ids = list(set(filter(None, [getattr(line, 'order_id', None) for line in order_lines])))
                logger.info(f"[{csv_upload_id}]   Unique orders: {len(order_ids)}")
                if len(order_ids) > 0 and len(order_ids) <= 5:
                    for oid in order_ids[:5]:
                        order_skus = [getattr(line, 'sku', None) for line in order_lines if getattr(line, 'order_id', None) == oid]
                        logger.info(f"[{csv_upload_id}]     Order {oid}: {order_skus}")

            except Exception as db_e:
                phase1_db_duration = (time.time() - phase1_start) * 1000
                logger.error(
                    f"[{csv_upload_id}] ❌ PHASE 1 FAILED: Database query failed after {phase1_db_duration:.0f}ms!\n"
                    f"  Error type: {type(db_e).__name__}\n"
                    f"  Error message: {str(db_e)}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                raise

            logger.info(
                f"[{csv_upload_id}] ✅ Quick-start Phase 1: Loaded {len(order_lines)} order lines"
            )

            # Early exit check: insufficient data
            logger.info(f"[{csv_upload_id}] 🔍 Checking data sufficiency (minimum: 10 order lines)...")
            if len(order_lines) < 10:
                logger.warning(
                    f"[{csv_upload_id}] Quick-start: Insufficient data - only {len(order_lines)} order lines. "
                    f"Need at least 10 for meaningful bundles."
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="finalization",
                    progress=100,
                    status="completed",
                    message=f"Insufficient order data to generate bundles. Found {len(order_lines)} order lines, need at least 10.",
                    metadata={
                        "exit_reason": "insufficient_order_lines",
                        "order_lines_count": len(order_lines),
                        "min_required": 10,
                        "data_issue": True,
                    },
                )
                return {
                    "recommendations": [],
                    "metrics": {
                        "quick_start_mode": True,
                        "total_recommendations": 0,
                        "exit_reason": "insufficient_order_lines",
                        "order_lines_count": len(order_lines),
                        "data_issue": True,
                    },
                    "quick_start": True,
                    "csv_upload_id": csv_upload_id,
                }

            # Limit to top products by sales volume
            from collections import Counter
            variant_sales = Counter()  # Track sales by variant_id
            unique_variants = set()     # Track unique variant_ids
            for line in order_lines:
                variant_id = getattr(line, 'variant_id', None)
                quantity = getattr(line, 'quantity', 0) or 0

                # ARCHITECTURE: Use variant_id as primary key (always exists, immutable, unique)
                # SKU is stored in catalog for display/merchant reference only
                if variant_id:
                    unique_variants.add(variant_id)
                    variant_sales[variant_id] += quantity

            logger.info(
                f"[{csv_upload_id}] Quick-start: Found {len(unique_variants)} unique products (by variant_id), "
                f"total quantity sold: {sum(variant_sales.values())}"
            )

            # Early exit check: insufficient product variety
            if len(unique_variants) < 2:
                logger.warning(
                    f"[{csv_upload_id}] Quick-start: Insufficient product variety - only {len(unique_variants)} unique products. "
                    f"Need at least 2 for bundles."
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="finalization",
                    progress=100,
                    status="completed",
                    message=f"Insufficient product variety for bundles. Found {len(unique_variants)} unique product(s), need at least 2.",
                    metadata={
                        "exit_reason": "insufficient_product_variety",
                        "unique_products": len(unique_variants),
                        "min_required": 2,
                        "data_issue": True,
                    },
                )
                return {
                    "recommendations": [],
                    "metrics": {
                        "quick_start_mode": True,
                        "total_recommendations": 0,
                        "exit_reason": "insufficient_product_variety",
                        "unique_variants": len(unique_variants),
                        "data_issue": True,
                    },
                    "quick_start": True,
                    "csv_upload_id": csv_upload_id,
                }

            # Get top N products by sales
            top_variants = [variant_id for variant_id, _ in variant_sales.most_common(max_products)]
            logger.info(
                f"[{csv_upload_id}] Quick-start: Selected top {len(top_variants)} products from {len(unique_variants)} total"
            )

            # Filter order lines to only include top products
            top_variants_set = set(top_variants)
            filtered_lines = []
            for line in order_lines:
                variant_id = getattr(line, 'variant_id', None)
                # Use variant_id as primary key (consistent with above)
                if variant_id and variant_id in top_variants_set:
                    filtered_lines.append(line)

            logger.info(
                f"[{csv_upload_id}] Quick-start: Filtered {len(order_lines)} → {len(filtered_lines)} order lines "
                f"(kept {len(filtered_lines)/len(order_lines)*100:.1f}%)"
            )

            phase1_duration = (time.time() - phase1_start) * 1000

            await update_generation_progress(
                csv_upload_id,
                step="scoring",
                progress=40,
                status="in_progress",
                message=f"Quick-start: Analyzing {len(top_variants)} products…",
            )

            # PHASE 2: Simple objective scoring (only high-priority objectives)
            logger.info(f"[{csv_upload_id}] 📊 PHASE 2: Starting product scoring...")
            phase2_start = time.time()

            # Use only 2 high-priority objectives for speed
            quick_objectives = ['increase_aov', 'clear_slow_movers']
            logger.info(f"[{csv_upload_id}] Using objectives: {quick_objectives}")

            logger.info(f"[{csv_upload_id}] 🗂️ Loading catalog snapshots map...")
            catalog_start = time.time()
            try:
                canonical_upload_id = await self._canonical_upload_id(csv_upload_id)
                run_id = await storage.get_run_id_for_upload(canonical_upload_id)

                # Prefer the latest catalog upload for the run; fall back to canonical orders upload if not found.
                catalog_upload_id = canonical_upload_id
                try:
                    catalog_upload = await storage.get_latest_upload_for_run(run_id, "catalog_joined") if run_id else None
                    if catalog_upload and getattr(catalog_upload, "id", None):
                        catalog_upload_id = catalog_upload.id
                        if catalog_upload_id != canonical_upload_id:
                            logger.info(
                                "[%s] Using latest catalog upload %s for run %s (orders upload=%s)",
                                csv_upload_id,
                                catalog_upload_id,
                                run_id,
                                canonical_upload_id,
                            )
                except Exception as catalog_lookup_exc:
                    logger.warning(
                        "[%s] Failed to resolve latest catalog upload for run %s: %s",
                        csv_upload_id,
                        run_id,
                        catalog_lookup_exc,
                    )

                catalog = await storage.get_catalog_snapshots_map_by_variant(catalog_upload_id)

                # If still empty and we have a run_id, try run-scoped map as a fallback.
                if not catalog and run_id:
                    logger.warning(
                        "[%s] Catalog map empty for upload %s; retrying by run_id=%s",
                        csv_upload_id,
                        catalog_upload_id,
                        run_id,
                    )
                    catalog = await storage.get_catalog_snapshots_map_by_variant_and_run(run_id)
                    if catalog:
                        logger.info(
                            "[%s] Fallback catalog load by run_id succeeded with %d entries",
                            csv_upload_id,
                            len(catalog),
                        )

                catalog_duration = (time.time() - catalog_start) * 1000
                logger.info(
                    f"[{csv_upload_id}] ✅ Catalog loaded in {catalog_duration:.0f}ms\n"
                    f"  Catalog entries: {len(catalog)} (source upload={catalog_upload_id})"
                )

                # EXTENSIVE LOGGING: Catalog contents
                # NOTE: Catalog is keyed by variant_id (primary key), but we log SKUs for debugging
                catalog_variant_ids = list(catalog.keys())
                catalog_skus = [getattr(snap, 'sku', '') for snap in catalog.values()]

                logger.info(f"[{csv_upload_id}] 📋 QUICK-START PHASE 2 - CATALOG ANALYSIS:")
                logger.info(f"[{csv_upload_id}]   Products in catalog (by variant_id): {len(catalog_variant_ids)}")
                logger.info(f"[{csv_upload_id}]   All catalog variant_ids: {sorted(catalog_variant_ids)}")
                logger.info(f"[{csv_upload_id}]   All catalog SKUs (for reference): {sorted(filter(None, catalog_skus))}")

                # Show SKU type breakdown in catalog (for debugging data quality)
                no_sku_catalog = [s for s in catalog_skus if s and s.startswith('no-sku-')]
                acc_catalog = [s for s in catalog_skus if s and s.startswith('ACC-')]
                other_catalog = [s for s in catalog_skus if s and not s.startswith('no-sku-') and not s.startswith('ACC-')]

                logger.info(f"[{csv_upload_id}]   Catalog SKUs with 'no-sku-' prefix: {len(no_sku_catalog)}")
                logger.info(f"[{csv_upload_id}]   Catalog SKUs with 'ACC-' prefix: {len(acc_catalog)}")
                logger.info(f"[{csv_upload_id}]   Catalog SKUs with other formats: {len(other_catalog)}")

                # Check for price data
                variants_with_price = [vid for vid, snap in catalog.items() if getattr(snap, 'price', 0) and float(getattr(snap, 'price', 0)) > 0]
                variants_without_price = [vid for vid in catalog_variant_ids if vid not in variants_with_price]

                logger.info(f"[{csv_upload_id}]   Products with valid prices (>$0): {len(variants_with_price)}")
                logger.info(f"[{csv_upload_id}]   Products with invalid/zero prices: {len(variants_without_price)}")

                if variants_with_price:
                    logger.info(f"[{csv_upload_id}] 💰 CATALOG PRICES (all entries):")
                    for variant_id in sorted(variants_with_price):
                        price = float(getattr(catalog[variant_id], 'price', 0))
                        sku = getattr(catalog[variant_id], 'sku', '')
                        logger.info(f"[{csv_upload_id}]     variant_id='{variant_id}' | SKU='{sku}' | price=${price}")

                if variants_without_price:
                    logger.warning(f"[{csv_upload_id}] ⚠️  Variant IDs in catalog with ZERO/NULL prices: {variants_without_price}")

                # Cross-check with order variant_ids (primary key comparison)
                order_variant_ids = set(filter(None, [getattr(line, 'variant_id', None) for line in order_lines]))
                catalog_variant_id_set = set(catalog_variant_ids)
                missing_in_catalog = order_variant_ids - catalog_variant_id_set
                missing_in_orders = catalog_variant_id_set - order_variant_ids

                logger.info(f"[{csv_upload_id}] 🔍 VARIANT_ID CROSS-CHECK (Orders vs Catalog):")
                logger.info(f"[{csv_upload_id}]   Variant IDs in orders: {len(order_variant_ids)}")
                logger.info(f"[{csv_upload_id}]   Variant IDs in catalog: {len(catalog_variant_id_set)}")
                logger.info(f"[{csv_upload_id}]   Variant IDs in BOTH: {len(order_variant_ids & catalog_variant_id_set)}")

                if missing_in_catalog:
                    logger.warning(
                        "[%s] ⚠️ Variant IDs in orders but NOT in catalog (%d): %s",
                        csv_upload_id,
                        len(missing_in_catalog),
                        sorted(list(missing_in_catalog))[:20],
                    )
                    logger.warning(
                        "[%s] Missing catalog rows will drop some FBT pairs but will not block all bundles.",
                        csv_upload_id,
                    )
                else:
                    logger.info(f"[{csv_upload_id}] ✅ All order variant_ids found in catalog!")

                if missing_in_orders:
                    logger.info(f"[{csv_upload_id}] ℹ️  Variant IDs in catalog but not in orders ({len(missing_in_orders)}): {sorted(missing_in_orders)}")

            except Exception as cat_e:
                catalog_duration = (time.time() - catalog_start) * 1000
                logger.error(
                    f"[{csv_upload_id}] ❌ PHASE 2 FAILED: Catalog loading failed after {catalog_duration:.0f}ms!\n"
                    f"  Error type: {type(cat_e).__name__}\n"
                    f"  Error message: {str(cat_e)}\n"
                    f"  Traceback:\n{traceback.format_exc()}"
                )
                raise

            # Simple scoring based on inventory flags
            product_scores = {}
            for variant_id in top_variants:
                snapshot = catalog.get(variant_id)
                if not snapshot:
                    continue

                score = 0.5  # Base score

                # Boost slow movers
                if getattr(snapshot, 'is_slow_mover', False):
                    score += 0.3

                # Boost high-margin products
                if getattr(snapshot, 'is_high_margin', False):
                    score += 0.2

                product_scores[variant_id] = score

            phase2_duration = (time.time() - phase2_start) * 1000

            await update_generation_progress(
                csv_upload_id,
                step="ml_generation",
                progress=60,
                status="in_progress",
                message="Quick-start: Building co-visitation graph…",
            )

            # PHASE 2.5: Build co-visitation graph (MODERN ML)
            # This gives us Item2Vec-style similarity without training
            phase2_5_start = time.time()

            from services.ml.pseudo_item2vec import build_covis_vectors

            # NOTE: filtered_lines already contains only top products, so no need for top_skus filter
            # build_covis_vectors extracts SKUs from order_lines internally
            covis_vectors = build_covis_vectors(
                order_lines=filtered_lines,
                top_skus=None,  # No additional filtering needed - filtered_lines is already filtered
                min_co_visits=1,
                max_neighbors=50,
                weighting="lift",  # Lift weighting: down-weights ubiquitous products
                min_lift=0.0,  # No filtering - let downstream ranking decide
            )

            logger.info(
                f"[{csv_upload_id}] Quick-start: Built co-visitation graph for {len(covis_vectors)} products"
            )

            phase2_5_duration = (time.time() - phase2_5_start) * 1000

            await update_generation_progress(
                csv_upload_id,
                step="ml_generation",
                progress=70,
                status="in_progress",
                message="Quick-start: Generating bundles…",
            )

            # PHASE 3: Multi-type bundle generation (FBT + BOGO + Volume)
            logger.info(f"[{csv_upload_id}] 🔗 PHASE 3: Multi-type bundle generation starting...")
            phase3_start = time.time()
            bundle_creation_start = time.time()

            # Decide how many of each type (respect overall max_bundles)
            # Priority: FBT (3-5) > BOGO (2-3) > Volume (1-2)
            max_fbt_bundles = max(3, min(5, max_bundles))
            remaining = max(0, max_bundles - max_fbt_bundles)
            max_bogo_bundles = min(3, remaining)
            remaining -= max_bogo_bundles
            max_volume_bundles = min(2, remaining)

            logger.info(
                f"[{csv_upload_id}] 📊 Bundle targets - "
                f"FBT={max_fbt_bundles}, BOGO={max_bogo_bundles}, VOLUME={max_volume_bundles}"
            )

            # ✅ OPTIMIZATION: Parallel bundle generation (FBT + BOGO + VOLUME)
            # Run all three bundle types concurrently using asyncio.gather()
            # This saves 2-4 seconds by running FBT, BOGO, VOLUME in parallel instead of sequentially
            import asyncio

            async def generate_fbt():
                """FBT bundles in separate thread"""
                return await asyncio.to_thread(
                    _build_quick_start_fbt_bundles,
                    csv_upload_id, filtered_lines, catalog, product_scores, max_fbt_bundles, covis_vectors
                )

            async def generate_bogo():
                """BOGO bundles in separate thread"""
                if max_bogo_bundles > 0:
                    return await asyncio.to_thread(
                        _build_quick_start_bogo_bundles,
                        csv_upload_id, catalog, product_scores, max_bogo_bundles
                    )
                return []

            async def generate_volume():
                """VOLUME bundles in separate thread"""
                if max_volume_bundles > 0:
                    return await asyncio.to_thread(
                        _build_quick_start_volume_bundles,
                        csv_upload_id, variant_sales, catalog, product_scores, max_volume_bundles
                    )
                return []

            logger.info(f"[{csv_upload_id}] ⚡ Running FBT, BOGO, VOLUME generation in parallel...")

            # Execute all three in parallel
            fbt_bundles, bogo_bundles, volume_bundles = await asyncio.gather(
                generate_fbt(),
                generate_bogo(),
                generate_volume(),
            )

            logger.info(
                f"[{csv_upload_id}] ✅ Parallel generation complete - "
                f"FBT: {len(fbt_bundles)}, BOGO: {len(bogo_bundles)}, VOLUME: {len(volume_bundles)}"
            )

            # Combine all bundle types
            recommendations = fbt_bundles + bogo_bundles + volume_bundles

            # Enforce global cap just in case
            recommendations = recommendations[:max_bundles]

            # Ensure ai_copy is always present to satisfy DB NOT NULL constraint.
            for rec in recommendations:
                self._ensure_ai_copy_present(rec)

            bundle_creation_duration = (time.time() - bundle_creation_start) * 1000
            logger.info(
                f"[{csv_upload_id}] ✅ PHASE 3 complete in {(time.time() - phase3_start) * 1000:.0f}ms\n"
                f"  Total bundles: {len(recommendations)}\n"
                f"  FBT: {len(fbt_bundles)}, BOGO: {len(bogo_bundles)}, VOLUME: {len(volume_bundles)}\n"
                f"  Bundle creation time: {bundle_creation_duration:.0f}ms"
            )

            phase3_duration = (time.time() - phase3_start) * 1000

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=90,
                status="in_progress",
                message="Quick-start: Saving bundles…",
            )

            # PHASE 4: Save recommendations
            logger.info(f"[{csv_upload_id}] 💾 PHASE 4: Saving {len(recommendations)} bundle recommendations...")
            phase4_start = time.time()

            if recommendations:
                try:
                    await storage.create_bundle_recommendations(recommendations)
                    save_duration = (time.time() - phase4_start) * 1000
                    logger.info(
                        f"[{csv_upload_id}] ✅ Saved {len(recommendations)} preview bundles in {save_duration:.0f}ms"
                    )
                except Exception as save_e:
                    save_duration = (time.time() - phase4_start) * 1000
                    logger.error(
                        f"[{csv_upload_id}] ❌ PHASE 4 FAILED: Database save failed after {save_duration:.0f}ms!\n"
                        f"  Recommendations to save: {len(recommendations)}\n"
                        f"  Error type: {type(save_e).__name__}\n"
                        f"  Error message: {str(save_e)}\n"
                        f"  Traceback:\n{traceback.format_exc()}"
                    )
                    raise
            else:
                logger.info(f"[{csv_upload_id}] ⚠️ No recommendations to save (0 bundles generated)")

            phase4_duration = (time.time() - phase4_start) * 1000

            total_duration = (time.time() - pipeline_start) * 1000

            # Safety check: Mark as failed if no bundles were generated
            bundle_count = len(recommendations)
            if bundle_count == 0:
                status = "failed"
                message = "No bundle patterns detected"
                logger.warning(f"[{csv_upload_id}] Quick-start completed with 0 bundles - marking as failed")
            else:
                status = "completed"
                message = f"Quick-start complete: {bundle_count} bundles in {total_duration/1000:.1f}s"

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=100,
                status=status,
                message=message,
            )

            # Build metrics
            metrics = {
                "quick_start_mode": True,
                "max_products_limit": max_products,
                "max_bundles_limit": max_bundles,
                "timeout_seconds": timeout_seconds,
                "total_recommendations": len(recommendations),
                "bundle_type_counts": {
                    "fbt": len(fbt_bundles),
                    "bogo": len(bogo_bundles),
                    "volume": len(volume_bundles),
                },
                "processing_time_ms": total_duration,
                "phase_timings": {
                    "phase1_data_loading_ms": phase1_duration,
                    "phase2_scoring_ms": phase2_duration,
                    "phase2_5_covis_graph_ms": phase2_5_duration,
                    "phase3_generation_ms": phase3_duration,
                    "phase4_persistence_ms": phase4_duration,
                },
                # Detailed filtering funnel metrics
                "funnel": {
                    "order_lines_loaded": len(order_lines),
                    "unique_variants_found": len(unique_variants),
                    "top_variants_selected": len(top_variants),
                    "order_lines_after_filter": len(filtered_lines),
                    "final_bundles": len(recommendations),
                },
                "products_analyzed": len(top_variants),
            }

            logger.info(
                f"[{csv_upload_id}] ========== QUICK-START COMPLETE ==========\n"
                f"  Total Bundles: {len(recommendations)}\n"
                f"  - FBT: {len(fbt_bundles)}\n"
                f"  - BOGO: {len(bogo_bundles)}\n"
                f"  - Volume: {len(volume_bundles)}\n"
                f"  Duration: {total_duration/1000:.2f}s\n"
                f"  Products Analyzed: {len(top_variants)}"
            )

            # Cleanup intermediate data to save storage costs
            try:
                run_id = await storage.get_run_id_for_upload(csv_upload_id)
                if run_id:
                    cleanup_result = await storage.cleanup_intermediate_data(run_id)
                    logger.info(f"[{csv_upload_id}] Post-generation cleanup: {cleanup_result}")
            except Exception as cleanup_error:
                logger.warning(f"[{csv_upload_id}] Cleanup failed (non-fatal): {cleanup_error}")

            return {
                "recommendations": recommendations,
                "metrics": metrics,
                "v2_pipeline": False,  # Mark as quick-start, not full v2
                "quick_start": True,
                "csv_upload_id": csv_upload_id,
            }

        except Exception as e:
            logger.error(f"[{csv_upload_id}] Quick-start generation failed: {e}", exc_info=True)

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=100,
                status="failed",
                message=f"Quick-start failed: {str(e)}",
            )

            raise
        finally:
            self._current_deadline = None

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

        if self.soft_timeout_seconds:
            self._current_deadline = Deadline(self.soft_timeout_seconds)
            logger.info(
                "[%s] Soft deadline set to %.1fs (remaining=%.1fs)",
                csv_upload_id,
                self.soft_timeout_seconds,
                self._deadline_remaining() or 0.0,
            )
        else:
            self._current_deadline = None

        try:
            # Initialize variables used in finally/except blocks to avoid NameError
            initial_bundle_count = 0
            final_bundle_count = 0
            drop_count = 0
            ai_metadata: Dict[str, Any] = {}
            final_recommendations = []

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

            should_async_defer = self._should_defer_async(dataset_profile, end_time, resume_used)
            if should_async_defer:
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
                await notify_partial_ready(
                    csv_upload_id,
                    len(all_recommendations),
                    details={
                        "phase": "phase_3_async_deferred",
                        "dataset_profile": dataset_profile,
                    },
                )
                return {
                    "recommendations": all_recommendations,
                    "metrics": metrics,
                    "async_deferred": True,
                    "dataset_profile": dataset_profile,
                }
            if (
                not should_async_defer
                and dataset_profile.get("defer_candidate")
                and not self.async_defer_enabled
                and not resume_used
            ):
                logger.warning(
                    "[%s] Async deferral disabled but conditions met; performing graceful pause.",
                    csv_upload_id,
                )
                result = await self._finalize_soft_timeout(
                    csv_upload_id,
                    "Phase 3 (async disabled)",
                    metrics,
                    pipeline_start,
                    all_recommendations,
                )
                await notify_partial_ready(
                    csv_upload_id,
                    len(all_recommendations),
                    details={
                        "phase": "phase_3_soft_timeout",
                        "dataset_profile": dataset_profile,
                    },
                )
                try:
                    metrics_collector.finish_pipeline(
                        pipeline_run_id,
                        len(result.get("recommendations", [])),
                        success=False,
                    )
                    pipeline_finished = True
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("[%s] Failed to finish pipeline metrics on soft exit: %s", csv_upload_id, exc)
                return result

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
                run_id=pipeline_run_id,
            )

            # Calculate counts for final notification (these were previously undefined)
            initial_bundle_count = len(all_recommendations)
            final_bundle_count = len(final_recommendations)
            drop_count = initial_bundle_count - final_bundle_count
            ai_metadata = {}  # Populated inside finalize_recommendations but not returned

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

            final_notify_details = {
                "initial_bundle_count": initial_bundle_count,
                "final_bundle_count": final_bundle_count,
                "drops": drop_count,
                "staged_publish": False,
            }
            if ai_metadata.get("drop_reasons"):
                final_notify_details["drop_reasons"] = ai_metadata["drop_reasons"]
            logger.info(
                "[%s] Finalization complete | bundles=%d drops=%d staged=False",
                csv_upload_id,
                final_bundle_count,
                drop_count,
            )
            await notify_bundle_ready(
                csv_upload_id,
                len(final_recommendations),
                resume_used,
                details=final_notify_details,
            )

            # Cleanup intermediate data to save storage costs
            try:
                run_id = await storage.get_run_id_for_upload(csv_upload_id)
                if run_id:
                    cleanup_result = await storage.cleanup_intermediate_data(run_id)
                    logger.info(f"[{csv_upload_id}] Post-generation cleanup: {cleanup_result}")
            except Exception as cleanup_error:
                logger.warning(f"[{csv_upload_id}] Cleanup failed (non-fatal): {cleanup_error}")

            return {
                "recommendations": final_recommendations,
                "metrics": metrics,
                "v2_pipeline": True,
                "csv_upload_id": csv_upload_id
            }

        except Exception as e:
            import traceback as tb_module
            # Calculate total time even on failure
            total_pipeline_duration = int((time.time() - pipeline_start) * 1000)

            logger.error(
                f"[{csv_upload_id}] ========== BUNDLE GENERATION PIPELINE FAILED ==========\n"
                f"  Duration until failure: {total_pipeline_duration}ms ({total_pipeline_duration/1000:.1f}s)\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Full traceback:\n{tb_module.format_exc()}"
            )
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

                # Enrich with type-specific structure for frontend compatibility
                recommendation = self._enrich_bundle_with_type_structure(recommendation)

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
        run_id: Optional[str] = None,
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

            staged_enabled = self._coerce_bool(
                feature_flags.get_flag("bundling.staged_publish_enabled", self.staged_publish_enabled),
                self.staged_publish_enabled,
            )
            logger.info(
                "[%s] Finalization entry | run=%s staged_enabled=%s initial_candidates=%d",
                csv_upload_id,
                run_id or f"bundle:{csv_upload_id}",
                staged_enabled,
                len(final_recommendations),
            )

            if staged_enabled and csv_upload_id:
                return await self._publish_in_stages(
                    final_recommendations,
                    csv_upload_id,
                    metrics,
                    end_time=end_time,
                    run_id=run_id,
                )

            initial_bundle_count = len(final_recommendations)
            final_bundle_count = initial_bundle_count
            drop_count = 0
            ai_metadata: Dict[str, Any] = {}
            ai_progress_message = "No bundles to describe."

            if final_recommendations:
                logger.info(
                    "[%s] Finalization pre-tracks | run=%s bundle_count=%d time_remaining=%s",
                    csv_upload_id,
                    run_id or f"bundle:{csv_upload_id}",
                    initial_bundle_count,
                    None
                    if not end_time
                    else f"{(end_time - datetime.now()).total_seconds():.1f}s",
                )
                metrics.setdefault("finalization", {})
                metrics["finalization"]["initial_bundle_count"] = initial_bundle_count

                await self.store_partial_recommendations(
                    final_recommendations,
                    csv_upload_id,
                    stage="phase_6_pre_copy",
                )

                await update_generation_progress(
                    csv_upload_id,
                    step="ai_descriptions",
                    progress=88,
                    status="in_progress",
                    message="Preparing AI descriptions and pricing…",
                    bundle_count=initial_bundle_count,
                    metadata={"initial_bundle_count": initial_bundle_count},
                )

                if not self._time_budget_exceeded(end_time):
                    processed_recommendations, track_summary = await self._run_post_filter_tracks(
                        final_recommendations,
                        csv_upload_id,
                        metrics,
                        end_time,
                        stage_index=0,
                        total_stages=1,
                        staged_mode=False,
                    )
                    drop_count = len(track_summary.get("dropped", []))
                    final_recommendations = processed_recommendations
                    final_bundle_count = len(final_recommendations)
                    logger.info(
                        "[%s] Finalization tracks complete | kept=%d dropped=%d drop_reasons=%s",
                        csv_upload_id,
                        final_bundle_count,
                        drop_count,
                        track_summary.get("drop_reasons"),
                    )

                    metrics["phase_timings"]["phase_8_ai_copy"] = track_summary.get("copy_ms", 0)
                    metrics.setdefault("finalization", {}).update(
                        {
                            "drops": drop_count,
                            "pricing_ms": track_summary.get("pricing_ms", 0),
                            "inventory_ms": track_summary.get("inventory_ms", 0),
                            "compliance_ms": track_summary.get("compliance_ms", 0),
                        }
                    )
                    metrics["total_recommendations"] = final_bundle_count

                    if drop_count:
                        ai_progress_message = f"AI descriptions ready. Filtered {drop_count} bundles."
                    else:
                        ai_progress_message = "AI descriptions ready."

                    ai_metadata = {
                        "initial_bundle_count": initial_bundle_count,
                        "final_bundle_count": final_bundle_count,
                        "dropped": drop_count,
                        "drop_reasons": track_summary.get("drop_reasons"),
                    }
                    try:
                        metrics_collector.record_drop_summary(
                            track_summary.get("drop_reasons", {}),
                            namespace="finalization",
                        )
                    except Exception as exc:
                        logger.debug("[%s] Failed to record finalization drop summary: %s", csv_upload_id, exc)
                else:
                    logger.warning(
                        f"[{csv_upload_id}] Skipping AI copy/pricing due to soft time budget proximity."
                    )
                    metrics["phase_timings"]["phase_8_ai_copy"] = 0
                    ai_progress_message = "Skipped AI descriptions due to time budget."
                    ai_metadata = {
                        "initial_bundle_count": initial_bundle_count,
                        "skipped": True,
                    }
            else:
                metrics["phase_timings"]["phase_8_ai_copy"] = 0
                metrics["total_recommendations"] = 0

            await update_generation_progress(
                csv_upload_id,
                step="ai_descriptions",
                progress=92,
                status="in_progress",
                message=ai_progress_message,
                bundle_count=final_bundle_count if final_bundle_count else None,
                metadata=ai_metadata if ai_metadata else None,
            )

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=95,
                status="in_progress",
                message="Finalizing bundle recommendations…",
                bundle_count=final_bundle_count if final_bundle_count else None,
                metadata={
                    "initial_bundle_count": initial_bundle_count,
                    "final_bundle_count": final_bundle_count,
                    "dropped": drop_count,
                },
            )

            # Remove any partial recommendations before persisting the final set
            if csv_upload_id:
                await storage.delete_partial_bundle_recommendations(csv_upload_id)

            # Store recommendations in database
            if final_recommendations and csv_upload_id:
                logger.info(
                    f"[{csv_upload_id}] Phase 9: Database Storage - STARTED | bundles={len(final_recommendations)}"
                )
                publish_result = await self._publish_wave(final_recommendations, csv_upload_id)
                storage_duration = publish_result["duration_ms"]
                logger.info(
                    f"[{csv_upload_id}] Phase 9: Database Storage - COMPLETED in {storage_duration}ms"
                )
                metrics["phase_timings"]["phase_9_storage"] = storage_duration
            elif final_bundle_count == 0:
                metrics["phase_timings"]["phase_9_storage"] = 0

            # Safety check: Mark as failed if no bundles were generated
            if final_bundle_count == 0:
                status = "failed"
                message = "No bundle patterns detected"
                logger.warning(f"[{csv_upload_id}] Bundle generation completed with 0 bundles - marking as failed")
            else:
                status = "completed"
                message = "Bundle generation complete."

            await update_generation_progress(
                csv_upload_id,
                step="finalization",
                progress=100,
                status=status,
                message=message,
                bundle_count=final_bundle_count if final_bundle_count else None,
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

    async def _publish_wave(
        self,
        wave: List[Dict[str, Any]],
        csv_upload_id: str,
        stage_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Persist a single wave and return IDs with timing."""
        if not wave:
            return {"ids": [], "duration_ms": 0}

        deduped_wave: List[Dict[str, Any]] = []
        seen_ids = set()
        for record in wave:
            rec_id = record.get("id")
            if rec_id and rec_id in seen_ids:
                continue
            if rec_id:
                seen_ids.add(rec_id)
            deduped_wave.append(record)

        if not deduped_wave:
            return {"ids": [], "duration_ms": 0}

        storage_start = time.time()
        with self._start_span(
            "bundle_generator.publish_wave",
            {
                "csv_upload_id": csv_upload_id,
                "stage": stage_index + 1 if stage_index is not None else None,
                "stage_size": len(deduped_wave),
            },
        ):
            await self.store_recommendations(deduped_wave, csv_upload_id)

        duration_ms = int((time.time() - storage_start) * 1000)
        published_ids = [rec.get("id") for rec in deduped_wave if rec.get("id")]
        return {"ids": published_ids, "duration_ms": duration_ms}

    async def _publish_in_stages(
        self,
        recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        metrics: Dict[str, Any],
        *,
        end_time: Optional[datetime] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Publish bundles in micro-batches so merchants see results sooner."""
        run_identifier = run_id or f"bundle:{csv_upload_id}"
        logger.info(
            "[finalize.enter] run=%s staged=%s wave_targets=%s hard_cap=%s remaining=%.2fs",
            run_identifier,
            True,
            self.staged_thresholds,
            self.staged_hard_cap,
            self._deadline_remaining() or -1.0,
        )

        if self._time_budget_exceeded(end_time):
            logger.warning(
                "[%s] TIMEOUT GUARD: skipping staged publish because deadline already exceeded",
                csv_upload_id,
            )
            return recommendations

        staged_state, metrics_state = await self._load_staged_wave_state(csv_upload_id)
        metrics.setdefault("phase_timings", {})
        metrics.setdefault("staged_publish", {})["enabled"] = True

        if not recommendations:
            staged_state.update(
                {
                    "waves": [],
                    "totals": {"published": 0, "dropped": 0},
                    "cursor": {"stage_idx": 0, "published": 0, "last_bundle_id": None},
                    "backpressure": {"active": False, "reason": None, "last_event": None},
                    "resume": {},
                }
            )
            await self._persist_staged_wave_state(csv_upload_id, staged_state, metrics_state)
            progress_payload = self._build_staged_progress_payload(
                run_identifier, staged_state, next_eta_seconds=None
            )
            await update_generation_progress(
                csv_upload_id,
                step="staged_publish",
                progress=100,
                status="completed",
                message="No bundles to publish.",
                bundle_count=0,
                metadata=progress_payload,
            )
            await storage.delete_partial_bundle_recommendations(csv_upload_id)
            metrics["total_recommendations"] = 0
            metrics["staged_publish"]["state"] = staged_state
            await notify_bundle_ready(
                csv_upload_id,
                0,
                resume_run=False,
                details={"staged_publish": staged_state},
            )
            return []

        prefer_high_score = self.staged_prefer_high_score
        ordered: List[Dict[str, Any]]
        if prefer_high_score:
            ordered = sorted(
                recommendations,
                key=lambda rec: rec.get("ranking_score", rec.get("confidence", 0.0)),
                reverse=True,
            )
        else:
            ordered = list(recommendations)

        hard_cap = self.staged_hard_cap if self.staged_hard_cap > 0 else None
        if hard_cap is None:
            staged_total_target = len(ordered)
            ordered_effective = ordered
        else:
            staged_total_target = min(len(ordered), hard_cap)
            ordered_effective = ordered[:staged_total_target]
            if len(ordered) > hard_cap:
                logger.info(
                    "[%s] Staged publish hard-cap reached | cap=%d total=%d",
                    csv_upload_id,
                    hard_cap,
                    len(ordered),
                )

        if not ordered_effective:
            staged_state["waves"] = []
            staged_state["totals"] = {"published": 0, "dropped": 0}
            staged_state["cursor"] = {"stage_idx": 0, "published": 0, "last_bundle_id": None}
            progress_payload = self._build_staged_progress_payload(
                run_identifier, staged_state, next_eta_seconds=None
            )
            await update_generation_progress(
                csv_upload_id,
                step="staged_publish",
                progress=100,
                status="completed",
                message="No bundles available after staging cap.",
                bundle_count=0,
                metadata=progress_payload,
            )
            metrics["total_recommendations"] = 0
            metrics["staged_publish"]["state"] = staged_state
            return []

        thresholds = [threshold for threshold in self.staged_thresholds if threshold > 0]
        if not thresholds:
            thresholds = [staged_total_target]
        thresholds = sorted(set(thresholds))

        dynamic_targets = [min(threshold, staged_total_target) for threshold in thresholds]
        if dynamic_targets[-1] < staged_total_target:
            dynamic_targets.append(staged_total_target)
        else:
            dynamic_targets[-1] = staged_total_target

        max_wave = max(1, self.staged_wave_batch_size)
        expanded_targets: List[int] = []
        cursor = max(0, published)
        for target in dynamic_targets:
            target = min(staged_total_target, target)
            while target - cursor > max_wave:
                cursor += max_wave
                expanded_targets.append(cursor)
            if target > cursor or not expanded_targets:
                cursor = target
                expanded_targets.append(target)
        # Deduplicate while preserving order
        seen_targets = set()
        ordered_expanded: List[int] = []
        for value in expanded_targets:
            if value not in seen_targets:
                ordered_expanded.append(value)
                seen_targets.add(value)
        dynamic_targets = ordered_expanded or dynamic_targets

        total_stages = len(dynamic_targets)
        total_candidates = staged_total_target

        completed_lookup = {wave.get("index"): wave for wave in staged_state.get("waves", [])}
        published = staged_state.get("totals", {}).get("published", 0)
        drops_total = staged_state.get("totals", {}).get("dropped", 0)
        next_stage_index = max(
            staged_state.get("cursor", {}).get("stage_idx", 0), len(completed_lookup)
        )
        staged_results: List[Dict[str, Any]] = []

        accumulated_copy_ms = 0
        accumulated_storage_ms = 0
        accumulated_pricing_ms = 0
        accumulated_inventory_ms = 0
        accumulated_compliance_ms = 0

        last_bundle_id = staged_state.get("cursor", {}).get("last_bundle_id")
        staged_state.setdefault("resume", {})

        logger.info(
            "[%s] Starting staged publish | run=%s stages=%d target=%d thresholds=%s remaining=%.2fs",
            csv_upload_id,
            run_identifier,
            total_stages,
            staged_total_target,
            dynamic_targets,
            self._deadline_remaining() or -1.0,
        )

        for index, stage_target in enumerate(dynamic_targets):
            if published >= staged_total_target:
                break

            if index < next_stage_index:
                logger.info(
                    "[%s] Skipping already-finalized stage %d | cursor=%s",
                    csv_upload_id,
                    index,
                    staged_state.get("cursor"),
                )
                continue

            if stage_target <= published:
                logger.debug(
                    "[%s] Stage %d target=%d already satisfied by published=%d; skipping.",
                    csv_upload_id,
                    index,
                    stage_target,
                    published,
                )
                continue

            async def defer_and_return(reason: str, time_left: Optional[float]) -> List[Dict[str, Any]]:
                staged_state["cursor"] = {
                    "stage_idx": index,
                    "published": published,
                    "last_bundle_id": last_bundle_id,
                }
                staged_state["resume"] = {
                    "stage_idx": index,
                    "published": published,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await self._persist_staged_wave_state(
                    csv_upload_id, staged_state, metrics_state
                )
                effective_remaining = (
                    time_left if time_left is not None else self._deadline_remaining()
                )
                logger.info(
                    "[finalize.defer] run=%s reason=%s time_left_s=%s resume_cursor=%s",
                    run_identifier,
                    reason,
                    None if effective_remaining is None else f"{max(effective_remaining, 0.0):.2f}",
                    staged_state["cursor"],
                )
                progress_payload = self._build_staged_progress_payload(
                    run_identifier,
                    staged_state,
                    next_eta_seconds=None,
                )
                await update_generation_progress(
                    csv_upload_id,
                    step="staged_publish",
                    progress=88,
                    status="in_progress",
                    message="Watchdog approaching; deferring remaining waves.",
                    bundle_count=published or None,
                    metadata=progress_payload,
                )
                return staged_results

            if self._time_budget_exceeded(end_time):
                return await defer_and_return("time_budget_exceeded", None)

            time_remaining = None
            if end_time:
                time_remaining = (end_time - datetime.now()).total_seconds()
            if (
                time_remaining is not None
                and time_remaining <= self.staged_soft_guard_seconds
            ):
                return await defer_and_return("watchdog_imminent", time_remaining)

            stage_slice = ordered_effective[published:min(stage_target, published + self.staged_wave_batch_size)]
            if not stage_slice:
                logger.debug(
                    "[%s] Stage %d yielded empty slice after filtering; aborting staged publish loop.",
                    csv_upload_id,
                    index,
                )
                break

            stage_start = time.time()
            logger.info(
                "[WAVE_START] upload=%s stage=%d target=%d published=%d remaining=%.2fs",
                csv_upload_id,
                index + 1,
                stage_target,
                published,
                self._deadline_remaining() or -1.0,
            )
            await self._emit_heartbeat(
                csv_upload_id,
                step="staged_publish",
                progress=85,
                message=f"Preparing stage {index + 1}/{total_stages}",
                metadata={
                    "stage_target": stage_target,
                    "processed": published,
                    "total_target": staged_total_target,
                },
            )

            await self.store_partial_recommendations(
                stage_slice,
                csv_upload_id,
                stage=f"stage_{index + 1}",
            )

            finalize_tx = uuid.uuid4().hex
            for rec in stage_slice:
                rec["idempotency_key"] = f"{run_identifier}:{rec.get('id')}:{index}"

            with self._start_span(
                "bundle_generator.stage_posts",
                {
                    "csv_upload_id": csv_upload_id,
                    "stage": index + 1,
                    "stage_size": len(stage_slice),
                    "total_target": staged_total_target,
                },
            ):
                processed_slice, track_summary = await self._run_post_filter_tracks(
                    stage_slice,
                    csv_upload_id,
                    metrics,
                    end_time,
                    stage_index=index,
                    total_stages=total_stages,
                    staged_mode=True,
                )

            kept_count = len(processed_slice)
            normalized_drops = track_summary.get("drop_reasons", {})
            dropped_count = sum(normalized_drops.values())
            published += kept_count
            drops_total += dropped_count
            staged_results.extend(processed_slice)

            copy_ms = track_summary.get("copy_ms", 0)
            pricing_ms = track_summary.get("pricing_ms", 0)
            inventory_ms = track_summary.get("inventory_ms", 0)
            compliance_ms = track_summary.get("compliance_ms", 0)
            storage_ms = 0
            publish_result = {"ids": [], "duration_ms": 0}

            if processed_slice:
                publish_result = await self._publish_wave(
                    processed_slice,
                    csv_upload_id,
                    stage_index=index,
                )
                storage_ms = publish_result["duration_ms"]
                published_ids = publish_result["ids"]
                if published_ids:
                    last_bundle_id = published_ids[-1]

            stage_duration = int((time.time() - stage_start) * 1000)
            accumulated_copy_ms += copy_ms
            accumulated_pricing_ms += pricing_ms
            accumulated_inventory_ms += inventory_ms
            accumulated_compliance_ms += compliance_ms
            accumulated_storage_ms += storage_ms

            if self._time_budget_exceeded(end_time):
                return await defer_and_return("time_budget_exceeded_post_stage", None)

            wave_entry = {
                "index": index,
                "target": stage_target,
                "published": kept_count,
                "drops": normalized_drops,
                "duration_ms": stage_duration,
                "persist_ms": storage_ms,
                "copy_ms": copy_ms,
                "pricing_ms": pricing_ms,
                "inventory_ms": inventory_ms,
                "compliance_ms": compliance_ms,
                "dropped": dropped_count,
                "published_ids": [rec.get("id") for rec in processed_slice],
                "finalize_tx": finalize_tx,
                "stage_version": len(staged_state.get("waves", [])) + 1,
            }
            completed_lookup[index] = wave_entry
            staged_state["waves"] = [
                completed_lookup[idx] for idx in sorted(completed_lookup.keys())
            ]
            staged_state["totals"] = {"published": published, "dropped": drops_total}
            staged_state["cursor"] = {
                "stage_idx": index + 1,
                "published": published,
                "last_bundle_id": last_bundle_id,
            }
            staged_state["resume"] = {
                "stage_idx": index + 1,
                "published": published,
                "timestamp": datetime.utcnow().isoformat(),
            }
            staged_state["last_finalize_tx"] = finalize_tx

            drop_summary_str = (
                "{" + ",".join(f"{k}:{v}" for k, v in normalized_drops.items()) + "}"
                if normalized_drops
                else "{}"
            )
            logger.info(
                "[WAVE_DONE] upload=%s stage=%d target=%d kept=%d drops=%d duration_ms=%d total_published=%d remaining_time=%.2fs %s",
                csv_upload_id,
                index + 1,
                stage_target,
                kept_count,
                dropped_count,
                stage_duration,
                published,
                self._deadline_remaining() or -1.0,
                drop_summary_str,
            )

            observed_total = kept_count + dropped_count
            drop_rate = (dropped_count / observed_total) if observed_total else 0.0
            next_eta_seconds: Optional[int] = int(self.staged_wave_cooldown_seconds) or None

            if drop_rate >= self.staged_auto_shrink_threshold and index + 1 < len(dynamic_targets):
                base_next = dynamic_targets[index + 1]
                shrink_target = max(
                    stage_target,
                    min(
                        staged_total_target,
                        int(math.ceil(base_next * self.staged_auto_shrink_factor)),
                    ),
                )
                dynamic_targets[index + 1] = max(shrink_target, stage_target)
                logger.info(
                    "[%s] Drop rate %.2f exceeded threshold %.2f; shrinking next target from %d to %d.",
                    csv_upload_id,
                    drop_rate,
                    self.staged_auto_shrink_threshold,
                    base_next,
                    dynamic_targets[index + 1],
                )
                staged_state["backpressure"] = {
                    "active": True,
                    "reason": "drop_rate_high",
                    "last_event": {
                        "stage_idx": index,
                        "drop_rate": round(drop_rate, 3),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }
            else:
                staged_state["backpressure"] = {
                    "active": False,
                    "reason": None,
                    "last_event": staged_state.get("backpressure", {}).get("last_event"),
                }
                logger.debug(
                    "[%s] Drop rate %.2f within threshold; next target remains %d.",
                    csv_upload_id,
                    drop_rate,
                    dynamic_targets[index + 1]
                    if index + 1 < len(dynamic_targets)
                    else staged_total_target,
                )

            try:
                metrics_collector.record_drop_summary(
                    normalized_drops, namespace=f"staged_stage_{index + 1}"
                )
            except Exception as exc:
                logger.debug("[%s] Failed to record staged metrics: %s", csv_upload_id, exc)

            await self._persist_staged_wave_state(
                csv_upload_id,
                staged_state,
                metrics_state,
            )

            progress_ratio = published / staged_total_target if staged_total_target else 1.0
            progress = 85 + int(progress_ratio * 14)
            progress_message = (
                f"Published {published} bundles (stage {index + 1}/{total_stages}); "
                f"filtered {dropped_count} this stage."
            )
            progress_payload = self._build_staged_progress_payload(
                run_identifier,
                staged_state,
                next_eta_seconds=next_eta_seconds,
            )

            await update_generation_progress(
                csv_upload_id,
                step="staged_publish",
                progress=min(progress, 99),
                status="in_progress",
                message=progress_message,
                bundle_count=published or None,
                metadata=progress_payload,
            )

            if kept_count:
                await notify_partial_ready(
                    csv_upload_id,
                    published,
                    details={
                        "stage_index": index,
                        "stage_target": stage_target,
                        "kept_this_stage": kept_count,
                        "dropped_this_stage": dropped_count,
                        "drops_after_stage": drops_total,
                        "total_target": staged_total_target,
                        "drop_reasons": normalized_drops,
                    },
                )

            if (
                index + 1 < len(dynamic_targets)
                and self.staged_wave_cooldown_seconds > 0
            ):
                try:
                    await asyncio.sleep(self.staged_wave_cooldown_seconds)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug(
                        "[%s] Interruption while cooling down between stages: %s",
                        csv_upload_id,
                        exc,
                    )

            logger.debug(
                "[%s] Stage cursor update | stage_idx=%d published=%d remaining=%d",
                csv_upload_id,
                staged_state["cursor"]["stage_idx"],
                staged_state["cursor"]["published"],
                staged_total_target - published,
            )

        if accumulated_copy_ms:
            metrics["phase_timings"]["phase_8_ai_copy"] = (
                metrics["phase_timings"].get("phase_8_ai_copy", 0) + accumulated_copy_ms
            )
        if accumulated_storage_ms:
            metrics["phase_timings"]["phase_9_storage"] = (
                metrics["phase_timings"].get("phase_9_storage", 0) + accumulated_storage_ms
            )

        metrics["staged_publish"]["state"] = staged_state
        metrics["staged_publish"]["tracks"] = {
            "copy_ms": accumulated_copy_ms,
            "pricing_ms": accumulated_pricing_ms,
            "inventory_ms": accumulated_inventory_ms,
            "compliance_ms": accumulated_compliance_ms,
            "storage_ms": accumulated_storage_ms,
        }
        metrics["staged_publish"]["published"] = staged_state["totals"]["published"]
        metrics["staged_publish"]["dropped"] = staged_state["totals"]["dropped"]
        metrics["total_recommendations"] = staged_state["totals"]["published"]

        try:
            metrics_collector.record_staged_publish(staged_state)
        except Exception as exc:
            logger.debug("[%s] Failed to persist staged summary metrics: %s", csv_upload_id, exc)

        await storage.delete_partial_bundle_recommendations(csv_upload_id)

        final_payload = self._build_staged_progress_payload(
            run_identifier,
            staged_state,
            next_eta_seconds=None,
        )
        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="completed",
            message="Bundle generation complete.",
            bundle_count=staged_state["totals"]["published"] or None,
            metadata=final_payload,
        )

        logger.info(
            "[%s] Staged publish complete | run=%s published=%d dropped=%d waves=%d",
            csv_upload_id,
            run_identifier,
            staged_state["totals"]["published"],
            staged_state["totals"]["dropped"],
            len(staged_state.get("waves", [])),
        )

        await notify_bundle_ready(
            csv_upload_id,
            staged_state["totals"]["published"],
            resume_run=False,
            details={"staged_publish": staged_state},
        )
        return staged_results
    async def _run_post_filter_tracks(
        self,
        stage_recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        metrics: Dict[str, Any],
        end_time: Optional[datetime],
        *,
        stage_index: int,
        total_stages: int,
        staged_mode: bool,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not stage_recommendations:
            return [], {
                "copy_ms": 0,
                "pricing_ms": 0,
                "inventory_ms": 0,
                "compliance_ms": 0,
                "duration_ms": 0,
                "dropped": [],
                "drop_reasons": {},
                "pricing_stats": {},
                "inventory_stats": {},
                "compliance_stats": {},
            }

        logger.info(
            "[%s] Post-filter tracks start | staged=%s stage_index=%d batch_size=%d",
            csv_upload_id,
            staged_mode,
            stage_index + 1,
            len(stage_recommendations),
        )

        copy_task = asyncio.create_task(
            self._generate_ai_copy_for_stage(
                stage_recommendations,
                csv_upload_id,
                metrics,
                end_time,
                stage_index=stage_index,
                total_stages=total_stages,
                staged_mode=staged_mode,
            )
        )
        pricing_task = asyncio.create_task(
            self._run_pricing_track(
                stage_recommendations,
                csv_upload_id,
                end_time,
                stage_index=stage_index,
            )
        )
        inventory_task = asyncio.create_task(
            self._run_inventory_track(
                stage_recommendations,
                csv_upload_id,
                end_time,
                stage_index=stage_index,
            )
        )

        try:
            copy_ms = await copy_task
        except asyncio.CancelledError:
            pricing_task.cancel()
            inventory_task.cancel()
            raise
        except Exception as exc:
            logger.warning("[%s] Copy track failed for stage %d: %s", csv_upload_id, stage_index + 1, exc)
            copy_ms = 0

        pricing_result_obj, inventory_result_obj = await asyncio.gather(
            pricing_task, inventory_task, return_exceptions=True
        )

        if isinstance(pricing_result_obj, Exception):
            logger.warning("[%s] Pricing track raised error: %s", csv_upload_id, pricing_result_obj)
            pricing_result = {
                "duration_ms": 0,
                "updated": 0,
                "skipped": len(stage_recommendations),
                "fallbacks": 0,
                "errors": 1,
            }
        else:
            pricing_result = pricing_result_obj

        if isinstance(inventory_result_obj, Exception):
            logger.warning("[%s] Inventory track raised error: %s", csv_upload_id, inventory_result_obj)
            inventory_result = {
                "duration_ms": 0,
                "dropped": [],
                "reason_counts": {"error": len(stage_recommendations)},
            }
        else:
            inventory_result = inventory_result_obj

        try:
            compliance_result = await self._run_compliance_track(stage_recommendations, stage_index=stage_index)
        except Exception as exc:
            logger.warning("[%s] Compliance track raised error: %s", csv_upload_id, exc)
            compliance_result = {
                "duration_ms": 0,
                "dropped": [],
                "reason_counts": {"error": len(stage_recommendations)},
                "trimmed_fields": 0,
            }

        drop_map: Dict[str, set[str]] = defaultdict(set)
        drop_totals: Dict[str, int] = defaultdict(int)

        def _register_drop(rec_id: Optional[str], reasons: Optional[List[str]], origin: str) -> None:
            if not rec_id:
                return
            normalized = {
                self._normalize_drop_reason(reason, origin) for reason in (reasons or [origin])
            }
            for reason in normalized:
                drop_totals[reason] += 1
            drop_map[rec_id].update(normalized)

        for entry in inventory_result.get("dropped", []):
            _register_drop(entry.get("id"), entry.get("reasons"), "inventory")

        for entry in compliance_result.get("dropped", []):
            _register_drop(entry.get("id"), entry.get("reasons"), "compliance")

        combined_drops = [
            {"id": rec_id, "reasons": sorted(reasons)}
            for rec_id, reasons in drop_map.items()
        ]

        filtered_recommendations = [
            rec for rec in stage_recommendations if rec.get("id") not in drop_map
        ]

        duration_ms = copy_ms + pricing_result.get("duration_ms", 0) + inventory_result.get("duration_ms", 0)
        duration_ms += compliance_result.get("duration_ms", 0)

        track_stats_entry = {
            "stage_index": stage_index + 1,
            "copy_ms": copy_ms,
            "pricing_ms": pricing_result.get("duration_ms", 0),
            "inventory_ms": inventory_result.get("duration_ms", 0),
            "compliance_ms": compliance_result.get("duration_ms", 0),
            "drops": dict(drop_totals),
        }
        if staged_mode:
            metrics.setdefault("staged_publish", {}).setdefault("track_stats", []).append(
                track_stats_entry
            )
        else:
            metrics.setdefault("finalization_tracks", []).append(track_stats_entry)

        summary = {
            "copy_ms": copy_ms,
            "pricing_ms": pricing_result.get("duration_ms", 0),
            "inventory_ms": inventory_result.get("duration_ms", 0),
            "compliance_ms": compliance_result.get("duration_ms", 0),
            "duration_ms": duration_ms,
            "dropped": combined_drops,
            "drop_reasons": dict(drop_totals),
            "pricing_stats": pricing_result,
            "inventory_stats": inventory_result,
            "compliance_stats": compliance_result,
            "drop_rate": (
                len(drop_map) / len(stage_recommendations) if stage_recommendations else 0.0
            ),
        }

        logger.info(
            "[%s] Post-filter tracks done | staged=%s stage=%d kept=%d dropped=%d copy_ms=%d pricing_ms=%d inventory_ms=%d compliance_ms=%d drops=%s",
            csv_upload_id,
            staged_mode,
            stage_index + 1,
            len(filtered_recommendations),
            len(combined_drops),
            copy_ms,
            pricing_result.get("duration_ms", 0),
            inventory_result.get("duration_ms", 0),
            compliance_result.get("duration_ms", 0),
            dict(drop_totals),
        )

        return filtered_recommendations, summary

    async def _run_pricing_track(
        self,
        stage_recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        end_time: Optional[datetime],
        *,
        stage_index: int,
    ) -> Dict[str, Any]:
        start = time.time()
        updated = 0
        skipped = 0
        fallbacks = 0
        errors = 0

        for recommendation in stage_recommendations:
            if self._time_budget_exceeded(end_time):
                logger.warning(
                    "[%s] Pricing track stopping early due to time budget (stage %d).",
                    csv_upload_id,
                    stage_index + 1,
                )
                break

            pricing_payload = recommendation.get("pricing") or {}
            if pricing_payload and pricing_payload.get("bundle_price"):
                skipped += 1
                continue

            skus = self._extract_sku_list(recommendation)
            if len(skus) < 2:
                skipped += 1
                continue

            try:
                result = await self.pricing_engine.compute_bundle_pricing(
                    skus,
                    recommendation.get("objective", "increase_aov"),
                    csv_upload_id,
                    recommendation.get("bundle_type", "FBT"),
                )
                if result.get("success"):
                    recommendation["pricing"] = result.get("pricing", {})
                    updated += 1
                else:
                    recommendation.setdefault("pricing", result.get("pricing", {}))
                    fallbacks += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors += 1
                logger.warning(
                    "[%s] Pricing track error for %s: %s",
                    csv_upload_id,
                    recommendation.get("id"),
                    exc,
                )

        duration_ms = int((time.time() - start) * 1000)
        summary = {
            "duration_ms": duration_ms,
            "updated": updated,
            "skipped": skipped,
            "fallbacks": fallbacks,
            "errors": errors,
        }
        logger.info(
            "[%s] Pricing track summary | stage=%d updated=%d skipped=%d fallbacks=%d errors=%d duration_ms=%d",
            csv_upload_id,
            stage_index + 1,
            updated,
            skipped,
            fallbacks,
            errors,
            duration_ms,
        )
        return summary

    async def _run_inventory_track(
        self,
        stage_recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        end_time: Optional[datetime],
        *,
        stage_index: int,
    ) -> Dict[str, Any]:
        start = time.time()
        dropped: List[Dict[str, Any]] = []
        reason_counts: Dict[str, int] = defaultdict(int)

        try:
            canonical_upload_id = await self._canonical_upload_id(csv_upload_id)
            run_id = await storage.get_run_id_for_upload(canonical_upload_id)

            catalog_upload_id = canonical_upload_id
            if run_id:
                try:
                    latest_catalog = await storage.get_latest_upload_for_run(run_id, "catalog_joined")
                    if latest_catalog and getattr(latest_catalog, "id", None):
                        catalog_upload_id = latest_catalog.id
                        if catalog_upload_id != canonical_upload_id:
                            logger.info(
                                "[%s] Inventory track using catalog upload %s for run %s (orders upload=%s)",
                                csv_upload_id,
                                catalog_upload_id,
                                run_id,
                                canonical_upload_id,
                            )
                except Exception as exc:
                    logger.warning(
                        "[%s] Inventory track failed to resolve catalog upload for run %s: %s",
                        csv_upload_id,
                        run_id,
                        exc,
                    )

            catalog_map = await storage.get_catalog_snapshots_map_by_variant(catalog_upload_id)

            if not catalog_map and run_id:
                logger.warning(
                    "[%s] Inventory track catalog map empty for upload %s; retrying by run_id=%s",
                    csv_upload_id,
                    catalog_upload_id,
                    run_id,
                )
                catalog_map = await storage.get_catalog_snapshots_map_by_variant_and_run(run_id)
        except Exception as exc:
            logger.warning("[%s] Inventory track failed to load catalog map: %s", csv_upload_id, exc)
            duration_ms = int((time.time() - start) * 1000)
            return {
                "duration_ms": duration_ms,
                "dropped": [],
                "reason_counts": {"catalog_error": len(stage_recommendations)},
            }

        if not catalog_map:
            duration_ms = int((time.time() - start) * 1000)
            return {
                "duration_ms": duration_ms,
                "dropped": [],
                "reason_counts": {"catalog_missing": len(stage_recommendations)},
            }

        for recommendation in stage_recommendations:
            if self._time_budget_exceeded(end_time):
                logger.warning(
                    "[%s] Inventory track stopping early due to time budget (stage %d).",
                    csv_upload_id,
                    stage_index + 1,
                )
                break

            rec_id = recommendation.get("id")
            skus = self._extract_sku_list(recommendation)

            missing: List[str] = []
            out_of_stock: List[str] = []
            inactive: List[str] = []

            for sku in skus:
                snapshot = catalog_map.get(sku)
                if not snapshot:
                    missing.append(sku)
                    continue

                available = getattr(snapshot, "available_total", None)
                if available is not None and available <= 0:
                    out_of_stock.append(sku)
                    continue

                status = (getattr(snapshot, "product_status", "") or "").lower()
                if status and status not in {"active", "available"}:
                    inactive.append(sku)

            # Check environment variable to control inventory validation strictness
            # INVENTORY_VALIDATION_MODE: "strict" (drop bundles), "warn" (log only), "off" (skip)
            validation_mode = os.getenv("INVENTORY_VALIDATION_MODE", "warn").lower()

            reasons: List[str] = []
            if missing:
                reasons.append("missing_catalog")
            if out_of_stock:
                reasons.append("out_of_stock")
            if inactive:
                reasons.append("inactive_product")

            if reasons and rec_id:
                # Log inventory issues as warnings instead of dropping bundles
                if validation_mode == "strict":
                    # Original behavior: drop bundles with inventory issues
                    dropped.append(
                        {
                            "id": rec_id,
                            "reasons": reasons,
                            "missing": missing,
                            "out_of_stock": out_of_stock,
                            "inactive": inactive,
                        }
                    )
                    for reason in reasons:
                        reason_counts[reason] += 1
                else:
                    # Warn mode: log issues but don't drop bundles
                    logger.warning(
                        "[%s] Inventory issues for bundle %s (NOT dropping, mode=%s): %s | missing=%s out_of_stock=%s inactive=%s",
                        csv_upload_id,
                        rec_id,
                        validation_mode,
                        reasons,
                        missing,
                        out_of_stock,
                        inactive,
                    )
                    reason_counts["ok_with_warnings"] += 1
            else:
                reason_counts["ok"] += 1

        duration_ms = int((time.time() - start) * 1000)
        summary = {
            "duration_ms": duration_ms,
            "dropped": dropped,
            "reason_counts": dict(reason_counts),
        }
        logger.info(
            "[%s] Inventory track summary | stage=%d dropped=%d reasons=%s duration_ms=%d",
            csv_upload_id,
            stage_index + 1,
            len(dropped),
            dict(reason_counts),
            duration_ms,
        )
        return summary

    async def _run_compliance_track(
        self,
        stage_recommendations: List[Dict[str, Any]],
        *,
        stage_index: int,
    ) -> Dict[str, Any]:
        start = time.time()
        banned_terms = {
            "100% guarantee",
            "risk-free",
            "free money",
            "instant weight loss",
            "miracle cure",
        }
        field_limits = {
            "title": 60,
            "description": 220,
            "valueProposition": 160,
        }

        dropped: List[Dict[str, Any]] = []
        reason_counts: Dict[str, int] = defaultdict(int)
        trimmed_fields = 0

        for recommendation in stage_recommendations:
            ai_copy = recommendation.get("ai_copy")
            if not ai_copy:
                recommendation["ai_copy"] = self.ai_generator.generate_fallback_copy(
                    [],
                    recommendation.get("bundle_type", "FBT"),
                )
                ai_copy = recommendation["ai_copy"]

            for field, limit in field_limits.items():
                value = ai_copy.get(field)
                if isinstance(value, str) and len(value) > limit:
                    ai_copy[field] = value[:limit].strip()
                    trimmed_fields += 1

            text_fragments: List[str] = []
            for value in ai_copy.values():
                if isinstance(value, str):
                    text_fragments.append(value.lower())
                elif isinstance(value, list):
                    text_fragments.extend(str(item).lower() for item in value)

            text_blob = " ".join(text_fragments)
            flagged_terms = [term for term in banned_terms if term in text_blob]
            if flagged_terms:
                dropped.append(
                    {
                        "id": recommendation.get("id"),
                        "reasons": ["policy_violation"],
                        "flagged_terms": flagged_terms,
                    }
                )
                reason_counts["policy_violation"] += 1
            else:
                reason_counts["ok"] += 1

        duration_ms = int((time.time() - start) * 1000)
        summary = {
            "duration_ms": duration_ms,
            "dropped": dropped,
            "reason_counts": dict(reason_counts),
            "trimmed_fields": trimmed_fields,
        }
        logger.info(
            "[%s] Compliance track summary | stage=%d dropped=%d trimmed=%d reasons=%s duration_ms=%d",
            csv_upload_id,
            stage_index + 1,
            len(dropped),
            trimmed_fields,
            dict(reason_counts),
            duration_ms,
        )
        return summary

    async def _generate_ai_copy_for_stage(
        self,
        stage_recommendations: List[Dict[str, Any]],
        csv_upload_id: str,
        metrics: Dict[str, Any],
        end_time: Optional[datetime],
        *,
        stage_index: int,
        total_stages: int,
        staged_mode: bool,
    ) -> int:
        """Generate AI copy for a micro-batch and update recommendations in place."""
        if not stage_recommendations:
            return 0

        start = time.time()
        tasks_executed = 0
        for offset, recommendation in enumerate(stage_recommendations):
            if self._time_budget_exceeded(end_time):
                logger.warning(
                    "[%s] Time budget exceeded while generating copy for stage %d bundle %d.",
                    csv_upload_id,
                    stage_index + 1,
                    offset + 1,
                )
                break

            products_raw = recommendation.get("products") or []
            normalized_products: List[Dict[str, Any]] = []
            for entry in products_raw:
                if isinstance(entry, dict):
                    normalized_products.append(entry)
                else:
                    normalized_products.append({"name": str(entry), "sku": str(entry)})

            bundle_type = recommendation.get("bundle_type", "FBT")
            context = f"Objective: {recommendation.get('objective', 'increase_aov')}"

            try:
                ai_copy = await self.ai_generator.generate_bundle_copy(
                    normalized_products,
                    bundle_type,
                    context=context,
                )
                recommendation["ai_copy"] = ai_copy
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to generate AI copy for stage %d bundle %d: %s",
                    csv_upload_id,
                    stage_index + 1,
                    offset + 1,
                    exc,
                )
                try:
                    recommendation["ai_copy"] = self.ai_generator.generate_fallback_copy(
                        normalized_products,
                        bundle_type,
                    )
                except Exception:
                    recommendation.setdefault(
                        "ai_copy",
                        {
                            "title": "Bundle Deal",
                            "description": "Recommended bundle offer.",
                            "valueProposition": "Great savings when purchasing together.",
                            "explanation": "Generated as part of bundle staging pipeline.",
                        },
                    )
            tasks_executed += 1

        duration_ms = int((time.time() - start) * 1000)
        copy_entry = {
            "stage_index": stage_index + 1,
            "processed": tasks_executed,
            "attempted": len(stage_recommendations),
            "duration_ms": duration_ms,
        }
        if staged_mode:
            metrics.setdefault("staged_publish", {}).setdefault("copy_batches", []).append(
                copy_entry
            )
        else:
            metrics.setdefault("finalization_copy_batches", []).append(copy_entry)
        return duration_ms
    
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

                        # Enrich with type-specific structure for frontend compatibility
                        recommendation = self._enrich_bundle_with_type_structure(recommendation)

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


# Quick-Start Bundle Helpers (FBT + BOGO + Volume)
# ============================================================================

def _build_quick_start_fbt_bundles(
    csv_upload_id: str,
    filtered_lines: List[Any],
    catalog: Dict[str, Any],
    product_scores: Dict[str, float],
    max_fbt_bundles: int,
    covis_vectors: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Build modern FBT bundles using co-visitation similarity (MODERN ML).

    Uses:
    - Product keys (SKU or variant_id as fallback) for product identification
    - Co-visitation graph for semantic similarity (pseudo-Item2Vec)
    - Bandit pricing for dynamic discount selection
    - Blended scoring: co-occurrence + similarity + product quality

    This gives "AI-powered" recommendations without ML training overhead.
    Handles Shopify stores without SKUs by using variant_id as fallback.
    """
    from collections import defaultdict
    from services.ml.pseudo_item2vec import cosine_similarity

    logger.info(f"[{csv_upload_id}] 🔗 FBT BUNDLE GENERATION - STARTED")
    logger.info(f"[{csv_upload_id}]   Input: {len(filtered_lines)} order lines")
    logger.info(f"[{csv_upload_id}]   Target: {max_fbt_bundles} FBT bundles")

    # Build variant_id -> SKU mapping (covis_vectors is keyed by SKU, not variant_id)
    variant_to_sku = {}
    for key, snap in catalog.items():
        variant_id = getattr(snap, 'variant_id', None)
        sku = getattr(snap, 'sku', None)
        if variant_id and sku:
            variant_to_sku[variant_id] = sku

    order_groups: Dict[str, List[str]] = defaultdict(list)

    for line in filtered_lines:
        order_id = getattr(line, 'order_id', None)
        variant_id = getattr(line, 'variant_id', None)

        # ARCHITECTURE: Use variant_id as primary key (always exists, immutable, unique)
        # SKU is retrieved from catalog when building bundle display data
        if order_id and variant_id:
            order_groups[order_id].append(variant_id)

    logger.info(f"[{csv_upload_id}]   Orders grouped: {len(order_groups)}")

    variant_pairs: Dict[tuple, int] = defaultdict(int)  # RENAMED: Now explicitly variant_id pairs

    for _, variant_ids in order_groups.items():
        unique_variants_in_order = list(set(variant_ids))
        for i, variant_id_1 in enumerate(unique_variants_in_order):
            for variant_id_2 in unique_variants_in_order[i + 1:]:
                pair = tuple(sorted((variant_id_1, variant_id_2)))
                variant_pairs[pair] += 1

    logger.info(f"[{csv_upload_id}] 📊 PRODUCT PAIRS FOUND (by variant_id):")
    logger.info(f"[{csv_upload_id}]   Total unique pairs: {len(variant_pairs)}")
    for pair, count in sorted(variant_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"[{csv_upload_id}]     {pair[0]} + {pair[1]}: {count} times")

    if not variant_pairs:
        logger.warning(f"[{csv_upload_id}] ⚠️  No product pairs found! Returning 0 bundles.")
        return []

    # MODERN: Score pairs using co-visitation similarity + co-occurrence
    scored_pairs = []
    filtered_out_count = 0
    for (variant_id_1, variant_id_2), count in variant_pairs.items():
        # Get co-visitation similarity (0-1 range)
        # NOTE: covis_vectors may be keyed by SKU or variant_id (for products without SKUs)
        covis_sim = 0.0
        sku1 = variant_to_sku.get(variant_id_1)
        sku2 = variant_to_sku.get(variant_id_2)
        # Try SKU first, then fall back to variant_id (for products without SKUs)
        key1 = sku1 if sku1 and sku1 in (covis_vectors or {}) else variant_id_1
        key2 = sku2 if sku2 and sku2 in (covis_vectors or {}) else variant_id_2
        if covis_vectors and key1 in covis_vectors and key2 in covis_vectors:
            v1 = covis_vectors[key1]
            v2 = covis_vectors[key2]
            covis_sim = cosine_similarity(v1, v2)

        # Blended score: 60% similarity + 40% co-occurrence frequency
        # High similarity = products naturally go together
        # High co-occurrence = proven purchase pattern
        co_occurrence_score = min(1.0, count / 10.0)  # Normalize to [0, 1]
        blended_score = 0.6 * covis_sim + 0.4 * co_occurrence_score

        # Require minimum quality: either good similarity OR multiple co-purchases
        if covis_sim < 0.1 and count < 2:
            filtered_out_count += 1
            logger.debug(f"[{csv_upload_id}]   Filtered: {variant_id_1}+{variant_id_2} (covis={covis_sim:.2f}, count={count})")
            continue  # Skip weak pairs

        scored_pairs.append(((variant_id_1, variant_id_2), count, covis_sim, blended_score))

    logger.info(f"[{csv_upload_id}] ✅ Scoring complete:")
    logger.info(f"[{csv_upload_id}]   Pairs after filtering: {len(scored_pairs)}/{len(variant_pairs)}")
    logger.info(f"[{csv_upload_id}]   Filtered out (weak): {filtered_out_count}")

    # Sort by blended score (descending)
    scored_pairs.sort(key=lambda x: x[3], reverse=True)

    recommendations: List[Dict[str, Any]] = []
    catalog_miss_count = 0
    price_fail_count = 0

    for (variant_id_1, variant_id_2), count, covis_sim, blended_score in scored_pairs:
        if len(recommendations) >= max_fbt_bundles * 3:  # Generate more candidates for filtering
            break

        p1 = catalog.get(variant_id_1)
        p2 = catalog.get(variant_id_2)
        if not p1 or not p2:
            catalog_miss_count += 1
            logger.warning(f"[{csv_upload_id}]   ❌ Catalog miss: variant_id_1='{variant_id_1}' variant_id_2='{variant_id_2}' | p1_exists={p1 is not None}, p2_exists={p2 is not None}")
            if not p1:
                logger.warning(f"[{csv_upload_id}]      variant_id '{variant_id_1}' not found in catalog")
            if not p2:
                logger.warning(f"[{csv_upload_id}]      variant_id '{variant_id_2}' not found in catalog")
            continue

        price1 = float(getattr(p1, 'price', 0) or 0)
        price2 = float(getattr(p2, 'price', 0) or 0)
        if price1 <= 0 or price2 <= 0:
            price_fail_count += 1
            logger.warning(f"[{csv_upload_id}]   ❌ Price invalid: variant_id_1='{variant_id_1}' variant_id_2='{variant_id_2}' | price1=${price1}, price2=${price2}")
            continue

        total_price = price1 + price2
        avg_price = total_price / 2.0

        # MODERN: Use bandit pricing instead of fixed 10%
        # This provides "dynamic AI pricing" without training
        from services.pricing import BayesianPricingEngine
        pricing_engine = BayesianPricingEngine()

        bandit_result = pricing_engine.multi_armed_bandit_pricing(
            bundle_products=[variant_id_1, variant_id_2],
            product_prices={variant_id_1: Decimal(str(price1)), variant_id_2: Decimal(str(price2))},
            features={
                "covis_similarity": covis_sim,
                "avg_price": avg_price,
                "bundle_type": "FBT",
                "objective": "increase_aov"
            },
            objective="increase_aov"
        )

        bundle_price = float(bandit_result["bundle_price"])
        discount_pct = bandit_result["discount_pct"]

        # Confidence based on similarity + co-occurrence
        conf = min(0.95, 0.3 + (0.4 * covis_sim) + (0.3 * min(1.0, count / 50.0)))

        # Ranking score: blend product quality + similarity
        product_quality_score = product_scores.get(variant_id_1, 0.5) + product_scores.get(variant_id_2, 0.5)
        ranking_score = 0.7 * product_quality_score + 0.3 * covis_sim

        # Build product data objects
        product1_data = {
            "sku": getattr(p1, 'sku', ''),
            "name": getattr(p1, 'product_title', 'Product'),
            "title": getattr(p1, 'product_title', 'Product'),
            "price": price1,
            "variant_id": variant_id_1,
            "product_id": getattr(p1, 'product_id', ''),
            "product_gid": f"gid://shopify/Product/{getattr(p1, 'product_id', '')}",
            "variant_gid": f"gid://shopify/ProductVariant/{variant_id_1}",
            "image_url": getattr(p1, 'image_url', None),
        }
        product2_data = {
            "sku": getattr(p2, 'sku', ''),
            "name": getattr(p2, 'product_title', 'Product'),
            "title": getattr(p2, 'product_title', 'Product'),
            "price": price2,
            "variant_id": variant_id_2,
            "product_id": getattr(p2, 'product_id', ''),
            "product_gid": f"gid://shopify/Product/{getattr(p2, 'product_id', '')}",
            "variant_gid": f"gid://shopify/ProductVariant/{variant_id_2}",
            "image_url": getattr(p2, 'image_url', None),
        }

        # Generate AI copy
        p1_name = getattr(p1, 'product_title', 'Product')
        p2_name = getattr(p2, 'product_title', 'Product')
        bundle_name = f"{p1_name} + {p2_name}"
        bundle_description = f"Get both {p1_name} and {p2_name} together and save {discount_pct}%!"

        recommendations.append({
            "id": str(uuid.uuid4()),
            "csv_upload_id": csv_upload_id,
            "bundle_type": "FBT",
            "objective": "increase_aov",
            # ===== PRODUCTS: Enhanced structure with trigger/addon =====
            "products": {
                # Legacy flat array for backward compatibility
                "items": [product1_data, product2_data],
                # Enhanced FBT structure
                "trigger_product": product1_data,
                "addon_products": [product2_data],
            },
            # ===== PRICING: Discount configuration =====
            "pricing": {
                "original_total": total_price,
                "bundle_price": bundle_price,
                "discount_amount": total_price - bundle_price,
                "discount_pct": f"{discount_pct}%",
                "discount_percentage": float(discount_pct),
                "discount_type": "percentage",
            },
            # ===== AI_COPY: Structured content + bundle settings =====
            "ai_copy": {
                "title": bundle_name,
                "description": bundle_description,
                "tagline": f"Save {discount_pct}% when bought together",
                "cta_text": "Add Bundle to Cart",
                "savings_message": f"Save ${total_price - bundle_price:.2f}!",
                # Bundle settings (stored in ai_copy since no metadata column)
                "is_active": True,
                "show_on": ["product", "cart"],
                # AI features for explainability
                "features": {
                    "covis_similarity": covis_sim,
                    "co_occurrence_count": count,
                    "blended_score": blended_score,
                },
            },
            "confidence": Decimal(str(conf)),
            "predicted_lift": Decimal(str(1.0 + (covis_sim * 0.5))),
            "ranking_score": Decimal(str(ranking_score)),
            "support": Decimal(str(min(1.0, count / 50.0))),
            "lift": Decimal(str(1.0 + covis_sim)),
            "discount_reference": f"__quick_start_{csv_upload_id}__",
            "is_approved": False,  # Merchant must approve bundles
            "created_at": datetime.utcnow(),
        })

        # Stop once we have enough high-quality bundles
        if len(recommendations) >= max_fbt_bundles:
            break

    # Fallback: if we didn't hit desired volume, try best-available pairs that exist in catalog.
    desired_min = max(5, max_fbt_bundles)
    if len(recommendations) < desired_min:
        logger.warning(
            "[%s] FBT shortfall: %d < %d. Filling with best available pairs present in catalog.",
            csv_upload_id,
            len(recommendations),
            desired_min,
        )
        # Handle both old format (products is a list) and new format (products is a dict with "items" key)
        existing_pairs = set()
        for p in recommendations:
            products = p.get("products")
            if not products:
                continue
            # New format: products is a dict with "items" list
            if isinstance(products, dict):
                items = products.get("items", [])
            else:
                # Old format: products is a list directly
                items = products
            if len(items) >= 2:
                existing_pairs.add(
                    tuple(sorted([items[0]["variant_id"], items[1]["variant_id"]]))
                )
        fallback_pairs = sorted(variant_pairs.items(), key=lambda x: x[1], reverse=True)
        for (variant_id_1, variant_id_2), count in fallback_pairs:
            if len(recommendations) >= desired_min:
                break
            pair_key = tuple(sorted((variant_id_1, variant_id_2)))
            if pair_key in existing_pairs:
                continue
            p1 = catalog.get(variant_id_1)
            p2 = catalog.get(variant_id_2)
            if not p1 or not p2:
                continue
            price1 = float(getattr(p1, "price", 0) or 0)
            price2 = float(getattr(p2, "price", 0) or 0)
            if price1 <= 0 or price2 <= 0:
                continue
            total_price = price1 + price2
            avg_price = total_price / 2.0
            conf = min(0.9, 0.25 + 0.2 * min(1.0, count / 10.0))
            ranking_score = 0.5 * product_scores.get(variant_id_1, 0.5) + 0.5 * product_scores.get(
                variant_id_2, 0.5
            )
            recommendations.append(
                {
                    "id": str(uuid.uuid4()),
                    "csv_upload_id": csv_upload_id,
                    "bundle_type": "FBT",
                    "objective": "increase_aov",
                    "products": [
                        {
                            "sku": getattr(p1, "sku", ""),
                            "name": getattr(p1, "product_title", "Product"),
                            "price": price1,
                            "variant_id": variant_id_1,
                            "product_id": getattr(p1, "product_id", ""),
                        },
                        {
                            "sku": getattr(p2, "sku", ""),
                            "name": getattr(p2, "product_title", "Product"),
                            "price": price2,
                            "variant_id": variant_id_2,
                            "product_id": getattr(p2, "product_id", ""),
                        },
                    ],
                    "pricing": {
                        "original_total": total_price,
                        "bundle_price": total_price * 0.9,
                        "discount_amount": total_price * 0.1,
                        "discount_pct": "10%",
                    },
                    "confidence": Decimal(str(conf)),
                    "predicted_lift": Decimal(str(1.0 + 0.1 * conf)),
                    "ranking_score": Decimal(str(ranking_score)),
                    "discount_reference": f"__quick_start_{csv_upload_id}__",
                    "is_approved": False,  # Merchant must approve bundles
                    "created_at": datetime.utcnow(),
                    "features": {
                        "fallback": True,
                        "co_occurrence_count": count,
                    },
                }
            )
            existing_pairs.add(pair_key)

    logger.info(f"[{csv_upload_id}] 🎉 FBT BUNDLE GENERATION - COMPLETED")
    logger.info(f"[{csv_upload_id}]   Bundles created: {len(recommendations)}")
    logger.info(f"[{csv_upload_id}]   Catalog misses: {catalog_miss_count}")
    logger.info(f"[{csv_upload_id}]   Price failures: {price_fail_count}")
    if len(recommendations) == 0:
        logger.error(f"[{csv_upload_id}] ❌ ZERO FBT BUNDLES CREATED!")
        logger.error(f"[{csv_upload_id}]   Possible reasons:")
        logger.error(f"[{csv_upload_id}]     - SKU pairs filtered out: {filtered_out_count}")
        logger.error(f"[{csv_upload_id}]     - Catalog lookup failures: {catalog_miss_count}")
        logger.error(f"[{csv_upload_id}]     - Invalid prices: {price_fail_count}")

    return recommendations


def _build_quick_start_bogo_bundles(
    csv_upload_id: str,
    catalog: Dict[str, Any],
    product_scores: Dict[str, float],
    max_bogo_bundles: int,
) -> List[Dict[str, Any]]:
    """
    Build simple BOGO bundles from slow-mover products.
    Heuristics:
    - is_slow_mover == True
    - available_total > 5
    - Buy 2, get 1 effectively 50% off (3 units for price of 2)
    """
    logger.info(f"[{csv_upload_id}] 🎁 BOGO BUNDLE GENERATION - STARTED")
    logger.info(f"[{csv_upload_id}]   Target: {max_bogo_bundles} BOGO bundles")

    candidates = []
    for key, snap in catalog.items():
        try:
            available = int(getattr(snap, 'available_total', 0) or 0)
        except (TypeError, ValueError):
            available = 0

        if getattr(snap, "is_slow_mover", False) and available > 5:
            # Use SKU for display (catalog has dual keys: variant_id and SKU)
            sku = getattr(snap, 'sku', '')
            candidates.append((sku, available, snap))

    logger.info(f"[{csv_upload_id}]   Slow-mover candidates found: {len(candidates)}/{len(catalog)}")

    if not candidates:
        logger.warning(f"[{csv_upload_id}] ⚠️  No slow-mover products found for BOGO!")
        return []

    # Sort by excess inventory (desc)
    candidates.sort(key=lambda t: t[1], reverse=True)

    bundles: List[Dict[str, Any]] = []
    for sku, available, snap in candidates:
        if len(bundles) >= max_bogo_bundles:
            break

        price = float(getattr(snap, 'price', 0) or 0)
        if price <= 0:
            continue

        # Extract variant_id for product_scores lookup (product_scores is keyed by variant_id)
        variant_id = getattr(snap, 'variant_id', '')

        # Buy 2, get 1 free ~= 50% effective discount on 3 units
        original_total = price * 3
        effective_bundle_price = price * 2  # Pay for 2, get 3
        effective_discount_pct = (1 - effective_bundle_price / original_total) * 100.0

        # Build product data object
        product_name = getattr(snap, 'product_title', 'Product')
        product_id = getattr(snap, 'product_id', '')
        product_data = {
            "sku": sku,
            "name": product_name,
            "title": product_name,
            "price": price,
            "variant_id": variant_id,
            "product_id": product_id,
            "product_gid": f"gid://shopify/Product/{product_id}",
            "variant_gid": f"gid://shopify/ProductVariant/{variant_id}",
            "image_url": getattr(snap, 'image_url', None),
        }

        # Generate AI copy
        bundle_name = f"Buy 2 Get 1 Free - {product_name}"
        bundle_description = f"Buy 2 {product_name} and get 1 FREE! Limited time offer."

        bundles.append({
            "id": str(uuid.uuid4()),
            "csv_upload_id": csv_upload_id,
            "bundle_type": "BOGO",
            "objective": "clear_slow_movers",
            # ===== PRODUCTS: Enhanced structure with qualifiers/rewards =====
            "products": {
                # Legacy flat array for backward compatibility
                "items": [product_data],
                # Enhanced BOGO structure
                "qualifiers": [
                    {
                        **product_data,
                        "quantity": 2,
                    }
                ],
                "rewards": [
                    {
                        **product_data,
                        "quantity": 1,
                        "discount_type": "free",
                        "discount_percent": 100,
                    }
                ],
            },
            # ===== PRICING: BOGO discount configuration =====
            "pricing": {
                "original_total": original_total,
                "bundle_price": effective_bundle_price,
                "discount_amount": original_total - effective_bundle_price,
                "discount_pct": f"{effective_discount_pct:.1f}%",
                "discount_percentage": round(effective_discount_pct, 1),
                "discount_type": "bogo",
                "bogo_config": {
                    "buy_qty": 2,
                    "get_qty": 1,
                    "discount_type": "free",
                    "discount_percent": 100,
                    "same_product": True,
                    "mode": "free_same_variant",
                }
            },
            # ===== AI_COPY: Structured content + bundle settings =====
            "ai_copy": {
                "title": bundle_name,
                "description": bundle_description,
                "tagline": "Buy 2, Get 1 FREE!",
                "cta_text": "Claim Deal",
                "savings_message": f"Save ${original_total - effective_bundle_price:.2f}!",
                # Bundle settings
                "is_active": True,
                "show_on": ["product", "cart"],
                # AI features
                "features": {
                    "is_slow_mover": True,
                    "available_inventory": available,
                },
            },
            "confidence": Decimal("0.6"),
            "predicted_lift": Decimal("1.0"),
            "ranking_score": Decimal(str(product_scores.get(variant_id, 0.5))),
            "support": Decimal("0.5"),
            "lift": Decimal("1.2"),
            "discount_reference": f"__quick_start_{csv_upload_id}__",
            "is_approved": False,  # Merchant must approve bundles
            "created_at": datetime.utcnow(),
        })

    logger.info(f"[{csv_upload_id}] 🎉 BOGO BUNDLE GENERATION - COMPLETED")
    logger.info(f"[{csv_upload_id}]   Bundles created: {len(bundles)}")

    return bundles


def _build_quick_start_volume_bundles(
    csv_upload_id: str,
    variant_sales: Counter,  # RENAMED: Now explicitly variant_id-based
    catalog: Dict[str, Any],
    product_scores: Dict[str, float],
    max_volume_bundles: int,
) -> List[Dict[str, Any]]:
    """
    Build simple volume break bundles:
    - Single anchor product (by variant_id)
    - Tiers: 2+, 3+, 5+ with fixed discounts
    Only for:
    - Popular products (based on variant_sales)
    - Sufficient stock (available_total > 20)
    """
    logger.info(f"[{csv_upload_id}] 📦 VOLUME BUNDLE GENERATION - STARTED")
    logger.info(f"[{csv_upload_id}]   Target: {max_volume_bundles} VOLUME bundles")

    candidates = []
    for variant_id, units_sold in variant_sales.items():
        snap = catalog.get(variant_id)
        if not snap:
            continue

        try:
            available = int(getattr(snap, 'available_total', 0) or 0)
        except (TypeError, ValueError):
            available = 0

        # Heuristic: high enough stock and at least some sales
        # Lowered thresholds for quick-start: stock > 5, sales >= 2
        if available > 5 and units_sold >= 2:
            candidates.append((variant_id, units_sold, available, snap))

    logger.info(f"[{csv_upload_id}]   High-stock candidates found: {len(candidates)}/{len(variant_sales)}")

    if not candidates:
        logger.warning(f"[{csv_upload_id}] ⚠️  No high-stock products found for VOLUME bundles!")
        # Fallback: pick top-N by sales (any stock, just need price > 0) to ensure we emit volume bundles.
        fallback_pool = []
        catalog_misses = 0
        zero_prices = 0

        logger.info(f"[{csv_upload_id}] 🔍 VOLUME FALLBACK DIAGNOSTICS:")
        logger.info(f"[{csv_upload_id}]   variant_sales has {len(variant_sales)} products")
        logger.info(f"[{csv_upload_id}]   Checking each for catalog presence and valid price...")

        for variant_id, units_sold in variant_sales.items():
            snap = catalog.get(variant_id)
            if not snap:
                catalog_misses += 1
                logger.debug(f"[{csv_upload_id}]     ❌ variant_id '{variant_id}' not in catalog")
                continue
            try:
                available = int(getattr(snap, 'available_total', 0) or 0)
            except (TypeError, ValueError):
                available = 0
            price = float(getattr(snap, 'price', 0) or 0)
            if price <= 0:
                zero_prices += 1
                logger.debug(f"[{csv_upload_id}]     ⚠️  variant_id '{variant_id}' has price={price}")
                continue
            # Accept ANY product with valid price, regardless of stock level
            fallback_pool.append((variant_id, units_sold, available, snap))

        logger.info(f"[{csv_upload_id}]   Results: {len(fallback_pool)} valid, {catalog_misses} not in catalog, {zero_prices} with price<=0")

        if not fallback_pool:
            logger.error(f"[{csv_upload_id}] ❌ No products with valid prices for VOLUME bundles!")
            logger.error(f"[{csv_upload_id}]   Checked {len(variant_sales)} products from sales data")
            logger.error(f"[{csv_upload_id}]   {catalog_misses} missing from catalog, {zero_prices} have price<=0")
            return []

        # Sort by sales desc (prioritize popular products), then availability
        fallback_pool.sort(key=lambda t: (t[1], t[2]), reverse=True)
        fallback_pool = fallback_pool[: max_volume_bundles or 2]

        logger.info(
            "[%s] Fallback volume candidates selected (top sellers): %s",
            csv_upload_id,
            [(vid, f"sales={sales}, stock={avail}") for vid, sales, avail, _ in fallback_pool],
        )
        candidates = fallback_pool

    # Sort by units sold desc, then stock desc
    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)

    bundles: List[Dict[str, Any]] = []
    for variant_id, units_sold, available, snap in candidates:
        if len(bundles) >= max_volume_bundles:
            break

        price = float(getattr(snap, 'price', 0) or 0)
        if price <= 0:
            continue

        # ===== ENHANCED: Volume tiers with labels =====
        volume_tiers = [
            {"min_qty": 1, "discount_type": "NONE",       "discount_value": 0, "label": None, "type": "percentage", "value": 0},
            {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5, "label": "Starter Pack", "type": "percentage", "value": 5},
            {"min_qty": 3, "discount_type": "PERCENTAGE", "discount_value": 10, "label": "Popular", "type": "percentage", "value": 10},
            {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15, "label": "Best Value", "type": "percentage", "value": 15},
        ]

        # Build product data object
        product_name = getattr(snap, 'product_title', 'Product')
        product_id_str = getattr(snap, 'product_id', '')
        product_data = {
            "sku": getattr(snap, 'sku', ''),
            "name": product_name,
            "title": product_name,
            "price": price,
            "variant_id": variant_id,
            "product_id": product_id_str,
            "product_gid": f"gid://shopify/Product/{product_id_str}",
            "variant_gid": f"gid://shopify/ProductVariant/{variant_id}",
            "image_url": getattr(snap, 'image_url', None),
        }

        # Generate AI copy
        bundle_name = f"Buy More, Save More - {product_name}"
        bundle_description = f"The more {product_name} you buy, the more you save! Up to 15% off."

        bundles.append({
            "id": str(uuid.uuid4()),
            "csv_upload_id": csv_upload_id,
            "bundle_type": "VOLUME",
            "objective": "increase_aov",
            # ===== PRODUCTS: Enhanced structure with volume info =====
            "products": {
                # Legacy flat array for backward compatibility
                "items": [product_data],
                # Volume tiers also in products for easy access
                "volume_tiers": volume_tiers,
            },
            # ===== PRICING: Volume tier configuration =====
            "pricing": {
                "original_total": price,
                "bundle_price": price,
                "discount_amount": 0,
                "discount_pct": "0%",
                "discount_type": "tiered",
                "volume_tiers": volume_tiers,
            },
            # ===== AI_COPY: Structured content + bundle settings =====
            "ai_copy": {
                "title": bundle_name,
                "description": bundle_description,
                "tagline": "Buy More, Save More!",
                "cta_text": "Select Quantity",
                "savings_message": "Save up to 15%!",
                # Bundle settings
                "is_active": True,
                "show_on": ["product"],
                # AI features
                "features": {
                    "units_sold": units_sold,
                    "available_inventory": available,
                },
            },
            "confidence": Decimal("0.6"),
            "predicted_lift": Decimal("1.0"),
            "ranking_score": Decimal(str(product_scores.get(variant_id, 0.5))),
            "support": Decimal("0.5"),
            "lift": Decimal("1.15"),
            "discount_reference": f"__quick_start_{csv_upload_id}__",
            "is_approved": False,  # Merchant must approve bundles
            "created_at": datetime.utcnow(),
        })

    logger.info(f"[{csv_upload_id}] 🎉 VOLUME BUNDLE GENERATION - COMPLETED")
    logger.info(f"[{csv_upload_id}]   Bundles created: {len(bundles)}")

    return bundles
