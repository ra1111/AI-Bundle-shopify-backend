"""
Bundle Recommendations Router
Handles bundle recommendation generation and management
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging
import os
import time
from datetime import datetime

from database import get_db, BundleRecommendation, AsyncSessionLocal
from services.bundle_generator import BundleGenerator
from services.progress_tracker import update_generation_progress
from routers.uploads import _resolve_orders_upload, _ensure_data_ready, _resolve_all_uploads_from_run
from settings import resolve_shop_id
from services.storage import storage
from services.pipeline_scheduler import pipeline_scheduler
from services.notifications import notify_partial_ready

logger = logging.getLogger(__name__)
router = APIRouter()


# Timeouts / watchdog — profiling-friendly defaults with env overrides
def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

# Default: no hard cutoff; use soft watchdog to defer long runs
BUNDLE_GENERATION_HARD_TIMEOUT_ENABLED = _env_bool("BUNDLE_GENERATION_HARD_TIMEOUT_ENABLED", False)
BUNDLE_GENERATION_TIMEOUT_SECONDS = int(os.getenv("BUNDLE_GENERATION_TIMEOUT_SECONDS", "360"))
BUNDLE_GENERATION_SOFT_WATCHDOG_SECONDS = int(os.getenv("BUNDLE_GENERATION_SOFT_WATCHDOG_SECONDS", "1200"))

# Quick-start mode for first-time installations
QUICK_START_ENABLED = _env_bool("QUICK_START_ENABLED", True)  # Enable by default
QUICK_START_TIMEOUT_SECONDS = int(os.getenv("QUICK_START_TIMEOUT_SECONDS", "120"))  # 2 minutes
QUICK_START_MAX_PRODUCTS = int(os.getenv("QUICK_START_MAX_PRODUCTS", "50"))  # Top 50 products
QUICK_START_MAX_BUNDLES = int(os.getenv("QUICK_START_MAX_BUNDLES", "10"))  # Limit to 10 bundles for speed

class GenerateBundlesRequest(BaseModel):
    csvUploadId: Optional[str] = None

class ApproveRecommendationRequest(BaseModel):
    isApproved: bool = True

@router.post("/generate-bundles")
async def generate_bundles(
    request: GenerateBundlesRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Generate bundle recommendations"""
    try:
        csv_upload_id = request.csvUploadId
        if not csv_upload_id:
            raise HTTPException(status_code=400, detail="csvUploadId is required")

        orders_upload_id, run_id, source_upload = await _resolve_orders_upload(csv_upload_id, db)
        await _ensure_data_ready(orders_upload_id, run_id, db)

        target_upload_id = orders_upload_id if source_upload.csv_type != 'orders' else csv_upload_id

        # Start background task to generate bundles
        background_tasks.add_task(generate_bundles_background, target_upload_id)
        
        scope = f"for CSV upload {target_upload_id}" if target_upload_id else "overall"
        return {"success": True, "message": f"Bundle recommendations generation started {scope}"}
        
    except Exception as e:
        logger.error(f"Bundle generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate bundle recommendations")

@router.get("/bundle-recommendations")
async def get_bundle_recommendations(
    shopId: str,
    uploadId: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get bundle recommendations with optional CSV filtering"""
    try:
        from settings import sanitize_shop_id
        shop_id = sanitize_shop_id(shopId)
        if not shop_id:
            raise HTTPException(status_code=400, detail="shop_id is required")
        from sqlalchemy import select
        
        query = select(BundleRecommendation).where(BundleRecommendation.shop_id == shop_id)
        if uploadId:
            query = query.where(BundleRecommendation.csv_upload_id == uploadId)
        
        # Conditional ordering based on scope
        if uploadId:
            # Per-upload scoped query: Use persistent rank_position with NULLS LAST
            query = query.order_by(
                BundleRecommendation.rank_position.asc().nulls_last(),  # Persistent ranking for this upload
                BundleRecommendation.ranking_score.desc(),  # Fallback for any records without rank_position
                BundleRecommendation.confidence.desc(),
                BundleRecommendation.created_at.desc()
            )
        else:
            # Global query: Use ranking_score for cross-upload comparison
            query = query.order_by(
                BundleRecommendation.ranking_score.desc(),  # Global ranking by score
                BundleRecommendation.confidence.desc(),
                BundleRecommendation.created_at.desc()
            )
        
        query = query.limit(50)
        result = await db.execute(query)
        recommendations = result.scalars().all()
        
        return [
            {
                "id": rec.id,
                "csvUploadId": rec.csv_upload_id,
                "shopId": rec.shop_id,
                "bundleType": rec.bundle_type,
                "objective": rec.objective,
                "products": rec.products,
                "pricing": rec.pricing,
                "aiCopy": rec.ai_copy,
                "confidence": str(rec.confidence),
                "predictedLift": str(rec.predicted_lift),
                "support": str(rec.support) if rec.support is not None else None,
                "lift": str(rec.lift) if rec.lift is not None else None,
                "rankingScore": float(rec.ranking_score) if hasattr(rec, 'ranking_score') and rec.ranking_score is not None else None,
                "discountReference": rec.discount_reference,
                "isApproved": rec.is_approved,
                "isUsed": rec.is_used,
                "rankPosition": rec.rank_position,
                "createdAt": rec.created_at.isoformat() if rec.created_at is not None else None
            }
            for rec in recommendations
        ]
        
    except Exception as e:
        logger.error(f"Get bundle recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get bundle recommendations")


@router.get("/bundle-recommendations/{upload_id}/partial")
async def get_partial_bundle_recommendations_route(upload_id: str):
    """Return partial bundle recommendations persisted during async deferral."""
    try:
        partials = await storage.get_partial_bundle_recommendations(upload_id)
        logger.info(
            "Partial bundle preview requested | upload_id=%s count=%d",
            upload_id,
            len(partials),
        )
        return [
            {
                "id": rec.id,
                "csvUploadId": rec.csv_upload_id,
                "bundleType": rec.bundle_type,
                "objective": rec.objective,
                "products": rec.products,
                "pricing": rec.pricing,
                "aiCopy": rec.ai_copy,
                "confidence": str(rec.confidence),
                "rankingScore": float(rec.ranking_score) if rec.ranking_score is not None else None,
                "discountReference": rec.discount_reference,
                "isPartial": True,
                "createdAt": rec.created_at.isoformat() if rec.created_at else None,
            }
            for rec in partials
        ]
    except Exception as exc:
        logger.error(f"Partial preview retrieval failed for {upload_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to fetch partial bundles")


@router.post("/generate-bundles/{upload_id}/resume")
async def resume_bundle_generation(upload_id: str, background_tasks: BackgroundTasks):
    """Trigger a resume of bundle generation from the latest checkpoint."""
    try:
        background_tasks.add_task(generate_bundles_background, upload_id, True)
        logger.info("Bundle resume queued | upload_id=%s", upload_id)
        return {"success": True, "message": f"Bundle generation resume queued for {upload_id}"}
    except Exception as exc:
        logger.error(f"Failed to queue resume for {upload_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to queue resume")

@router.patch("/bundle-recommendations/{recommendation_id}/approve")
async def approve_recommendation(
    recommendation_id: str,
    request: ApproveRecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Approve or reject a bundle recommendation"""
    try:
        recommendation = await db.get(BundleRecommendation, recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Update recommendation approval status
        from sqlalchemy import update
        await db.execute(
            update(BundleRecommendation)
            .where(BundleRecommendation.id == recommendation_id)
            .values(is_approved=request.isApproved)
        )
        await db.commit()
        
        return {
            "success": True,
            "message": f"Recommendation {'approved' if request.isApproved else 'rejected'}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update recommendation")

async def generate_bundles_background(csv_upload_id: Optional[str], resume_only: bool = False, from_auto_trigger: bool = False):
    """Background task to generate bundle recommendations with concurrency control and async deferrals.

    Args:
        csv_upload_id: The CSV upload ID to generate bundles for
        resume_only: If True, resume an interrupted generation
        from_auto_trigger: If True, skip initial progress updates (auto-trigger handles them)
    """
    import traceback
    from services.concurrency_control import concurrency_controller

    start_time = time.time()
    scope = f"for CSV upload {csv_upload_id}" if csv_upload_id else "overall"
    logger.info(f"{'Resuming' if resume_only else 'Starting'} bundle generation {scope}")
    logger.info(f"[{csv_upload_id}] ========== BUNDLE GENERATION BACKGROUND TASK STARTED ==========")

    if not csv_upload_id:
        logger.error("Bundle generation requires a valid CSV upload ID")
        return

    # Add queueing step at the very start (skip if called from auto-trigger which handles its own progress)
    if not resume_only and not from_auto_trigger:
        await update_generation_progress(
            csv_upload_id,
            step="queueing",
            progress=0,
            status="in_progress",
            message="Bundle generation queued and starting...",
        )

    try:
        logger.info(f"Bundle generation {scope}: attempting to acquire shop lock")
        # Use the new concurrency control system that locks by shop_id
        async with concurrency_controller.acquire_shop_lock_for_csv_upload(
            csv_upload_id, "bundle_generation"
        ) as lock_context:
            conn = lock_context["conn"]
            shop_id = lock_context["shop_id"]
            
            logger.info(f"Acquired bundle generation lock for shop {shop_id} (CSV upload {csv_upload_id})")

            # Atomically update status with precondition check
            logger.info(f"Bundle generation {scope}: updating CSV upload status to generating_bundles (resume={resume_only})")
            expected_status = "bundle_generation_async" if resume_only else None
            status_update = await concurrency_controller.atomic_status_update_with_precondition(
                csv_upload_id,
                "generating_bundles",
                expected_current_status=expected_status
            )

            if not status_update["success"]:
                logger.error(f"Failed to update CSV upload {csv_upload_id} status: {status_update['error']}")
                return

            logger.info(f"Status updated: {status_update['previous_status']} -> {status_update['new_status']}")

            # Resolve all 4 upload IDs (orders, catalog, variants, inventory) from the run
            # This fixes the Quickstart bug where each data type has a separate upload ID
            catalog_upload_id = None
            variants_upload_id = None
            inventory_upload_id = None
            try:
                async with AsyncSessionLocal() as db:
                    orders_id, catalog_id, variants_id, inventory_id, run_id = await _resolve_all_uploads_from_run(csv_upload_id, db)
                    catalog_upload_id = catalog_id
                    variants_upload_id = variants_id
                    inventory_upload_id = inventory_id
                    logger.info(
                        f"[{csv_upload_id}] Resolved upload IDs: "
                        f"orders={orders_id}, catalog={catalog_id}, variants={variants_id}, inventory={inventory_id}"
                    )
            except Exception as resolve_exc:
                logger.warning(
                    f"[{csv_upload_id}] Could not resolve all upload IDs (likely single upload, not Quickstart): {resolve_exc}. "
                    f"Continuing with csv_upload_id={csv_upload_id} for all data types."
                )

            try:
                # Check if shop has existing bundles to determine Quick Mode vs Full Pipeline
                from services.bundle_generator import BundleGenerator
                generator = BundleGenerator()

                should_use_quick_mode = False
                existing_bundle_count = 0

                if not resume_only:
                    preflight_start = time.time()
                    logger.info(f"[{csv_upload_id}] 🔍 Step: Checking existing bundles for shop_id={shop_id}...")
                    try:
                        # Check if shop has ANY existing bundles (shop-level, not upload-specific)
                        existing_bundle_count = await storage.get_bundle_count_for_shop(shop_id)
                        # Quick Mode runs when shop has NO existing bundles
                        should_use_quick_mode = existing_bundle_count == 0
                        preflight_duration = (time.time() - preflight_start) * 1000

                        logger.info(
                            f"[{csv_upload_id}] ✅ Pre-flight check COMPLETE in {preflight_duration:.0f}ms:\n"
                            f"  Shop: {shop_id}\n"
                            f"  Existing bundles for shop: {existing_bundle_count}\n"
                            f"  Should use Quick Mode: {should_use_quick_mode}"
                        )

                    except Exception as e:
                        preflight_duration = (time.time() - preflight_start) * 1000
                        logger.error(
                            f"[{csv_upload_id}] ❌ Pre-flight check FAILED after {preflight_duration:.0f}ms!\n"
                            f"  Error type: {type(e).__name__}\n"
                            f"  Error message: {str(e)}\n"
                            f"  Traceback:\n{traceback.format_exc()}"
                        )
                        # Default to Quick Mode on error (safer for new shops)
                        should_use_quick_mode = True
                        logger.info(f"[{csv_upload_id}] Defaulting to Quick Mode due to preflight error")

                # FAST PATH: Quick-start mode when shop has NO existing bundles
                if should_use_quick_mode:
                    quick_start_overall_start = time.time()
                    logger.info(
                        f"[{csv_upload_id}] 🚀 QUICK-START MODE ACTIVATED (no existing bundles)\n"
                        f"  Shop: {shop_id}\n"
                        f"  Existing bundles: {existing_bundle_count}\n"
                        f"  Max products: {QUICK_START_MAX_PRODUCTS}\n"
                        f"  Max bundles: {QUICK_START_MAX_BUNDLES}\n"
                        f"  Timeout: {QUICK_START_TIMEOUT_SECONDS}s\n"
                        f"  Full generation will be queued for background processing"
                    )

                    try:
                        # Run quick-start generation (fast preview)
                        logger.info(f"[{csv_upload_id}] 🔧 Starting generate_quick_start_bundles()...")
                        generation_result = await asyncio.wait_for(
                            generator.generate_quick_start_bundles(
                                csv_upload_id,
                                max_products=QUICK_START_MAX_PRODUCTS,
                                max_bundles=QUICK_START_MAX_BUNDLES,
                                timeout_seconds=QUICK_START_TIMEOUT_SECONDS,
                                catalog_upload_id=catalog_upload_id,
                                variants_upload_id=variants_upload_id,
                                inventory_upload_id=inventory_upload_id,
                            ),
                            timeout=float(QUICK_START_TIMEOUT_SECONDS + 30)  # Extra 30s buffer
                        )
                        quick_start_overall_duration = (time.time() - quick_start_overall_start) * 1000
                        bundle_count = generation_result.get('metrics', {}).get('total_recommendations', 0)

                        logger.info(
                            f"[{csv_upload_id}] ✅ Quick-start completed in {quick_start_overall_duration:.0f}ms\n"
                            f"  Bundles: {bundle_count}\n"
                            f"  Duration: {generation_result.get('metrics', {}).get('processing_time_ms', 0) / 1000:.2f}s"
                        )

                        # Always return early after quick-start (whether 0 or 10 bundles)
                        # Status is already set correctly by bundle_generator:
                        #   - bundleCount > 0 → status="completed"
                        #   - bundleCount = 0 → status="failed"
                        # If 0 bundles: User can click sync again → quick mode retries
                        # If >0 bundles: Quick mode succeeded → stop here

                        if bundle_count > 0:
                            # Update upload status to completed for successful quick-start
                            quick_complete = await storage.safe_mark_upload_completed(csv_upload_id)
                            if quick_complete:
                                logger.info(f"[{csv_upload_id}] Marked quick-start upload as completed")

                            # Notify user that preview bundles are ready
                            try:
                                await notify_partial_ready(csv_upload_id, generation_result.get('metrics', {}))
                            except Exception as notify_exc:
                                logger.warning(f"Failed to send quick-start notification: {notify_exc}")

                            logger.info(
                                f"[{csv_upload_id}] ✅ Quick-start SUCCESS - generated {bundle_count} bundles\n"
                                f"  Quick bundles are ready for immediate use\n"
                                f"  NOT running full pipeline (quick mode succeeded)"
                            )
                        else:
                            logger.warning(
                                f"[{csv_upload_id}] ⚠️ Quick-start generated 0 bundles\n"
                                f"  Status is marked as 'failed' by bundle_generator\n"
                                f"  User can click 'Sync Data' again to retry quick mode\n"
                                f"  NOT falling back to full pipeline automatically"
                            )

                        # Return early - quick-start is complete (success or failure)
                        return

                    except asyncio.TimeoutError:
                        quick_start_overall_duration = (time.time() - quick_start_overall_start) * 1000
                        logger.error(
                            f"[{csv_upload_id}] ⏱️ TIMEOUT: Quick-start exceeded {QUICK_START_TIMEOUT_SECONDS}s!\n"
                            f"  Actual duration: {quick_start_overall_duration:.0f}ms\n"
                            f"  Falling back to full generation pipeline\n"
                            f"  Traceback:\n{traceback.format_exc()}"
                        )
                        # Fall through to normal generation
                    except Exception as quick_exc:
                        quick_start_overall_duration = (time.time() - quick_start_overall_start) * 1000
                        logger.error(
                            f"[{csv_upload_id}] ❌ Quick-start FAILED after {quick_start_overall_duration:.0f}ms!\n"
                            f"  Error type: {type(quick_exc).__name__}\n"
                            f"  Error message: {str(quick_exc)}\n"
                            f"  Falling back to full generation pipeline\n"
                            f"  Traceback:\n{traceback.format_exc()}"
                        )
                        # Fall through to normal generation

                # NORMAL PATH: Full V2 pipeline (shop has existing bundles or quick-start failed)
                logger.info(
                    f"Bundle generation {scope}: using FULL V2 PIPELINE "
                    f"(existing_bundles={existing_bundle_count}, quick_mode_skipped={not should_use_quick_mode})"
                )

                # Soft watchdog task setup (only used when hard timeout is disabled)
                async def _soft_watchdog(csv_upload_id: str, started_at: float):
                    check_interval = 10
                    while True:
                        await asyncio.sleep(check_interval)
                        elapsed = time.time() - started_at
                        if elapsed > BUNDLE_GENERATION_SOFT_WATCHDOG_SECONDS:
                            # Flip to async deferral but DO NOT fail the run
                            logger.warning(
                                "Soft watchdog deferring bundle generation after %ds | upload_id=%s",
                                int(elapsed),
                                csv_upload_id,
                            )
                            try:
                                await update_generation_progress(
                                    csv_upload_id,
                                    step="optimization",
                                    progress=78,
                                    status="in_progress",
                                    message=f"Continuing asynchronously after {int(elapsed)} seconds.",
                                    metadata={"soft_watchdog_seconds": BUNDLE_GENERATION_SOFT_WATCHDOG_SECONDS},
                                )
                                await concurrency_controller.atomic_status_update_with_precondition(
                                    csv_upload_id,
                                    "bundle_generation_async",
                                    expected_current_status="generating_bundles",
                                    additional_fields={
                                        "bundle_generation_metrics": {
                                            "async_deferred": True,
                                            "deferred_at": datetime.utcnow().isoformat(),
                                            "soft_watchdog_triggered": True,
                                            "elapsed_seconds": int(elapsed),
                                        }
                                    },
                                )
                                # Schedule resume and exit watchdog
                                pipeline_scheduler.schedule(_resume_bundle_generation(csv_upload_id))
                            except Exception as watchdog_exc:  # pragma: no cover - defensive logging
                                logger.warning("Soft watchdog encountered an error: %s", watchdog_exc)
                            return

                logger.info(
                    "Bundle generation %s: invoking generator with %s",
                    scope,
                    f"hard timeout {BUNDLE_GENERATION_TIMEOUT_SECONDS}s" if BUNDLE_GENERATION_HARD_TIMEOUT_ENABLED else "soft watchdog mode",
                )

                watchdog_task = None
                try:
                    if BUNDLE_GENERATION_HARD_TIMEOUT_ENABLED:
                        generation_result = await asyncio.wait_for(
                            generator.generate_bundle_recommendations(
                                csv_upload_id,
                                catalog_upload_id=catalog_upload_id,
                                variants_upload_id=variants_upload_id,
                                inventory_upload_id=inventory_upload_id,
                            ),
                            timeout=float(BUNDLE_GENERATION_TIMEOUT_SECONDS),
                        )
                        logger.info("Bundle generation %s: generator completed without hard-timeout", scope)
                    else:
                        # Launch soft watchdog and run without hard timeout to measure true runtime
                        watchdog_task = asyncio.create_task(_soft_watchdog(csv_upload_id, start_time))
                        generation_result = await generator.generate_bundle_recommendations(
                            csv_upload_id,
                            catalog_upload_id=catalog_upload_id,
                            variants_upload_id=variants_upload_id,
                            inventory_upload_id=inventory_upload_id,
                        )
                        logger.info("Bundle generation %s: generator completed under soft-watchdog mode", scope)
                except asyncio.TimeoutError:
                    logger.error(
                        "TIMEOUT: Bundle generation exceeded %s seconds for %s, force terminating",
                        BUNDLE_GENERATION_TIMEOUT_SECONDS,
                        csv_upload_id,
                    )
                    timeout_message = f"Bundle generation timed out after {BUNDLE_GENERATION_TIMEOUT_SECONDS} seconds."
                    metrics_payload = {
                        "timeout_error": True,
                        "total_recommendations": 0,
                        "processing_time_ms": BUNDLE_GENERATION_TIMEOUT_SECONDS * 1000,
                        "timeout_reason": f"{BUNDLE_GENERATION_TIMEOUT_SECONDS}_second_hard_limit_exceeded",
                    }
                    generation_result = {
                        "recommendations": [],
                        "metrics": metrics_payload,
                        "v2_pipeline": True,
                        "csv_upload_id": csv_upload_id,
                    }

                    await update_generation_progress(
                        csv_upload_id,
                        step="finalization",
                        progress=100,
                        status="failed",
                        message=timeout_message,
                    )

                    try:
                        partial_recommendations = await storage.get_partial_bundle_recommendations(csv_upload_id)
                        if partial_recommendations:
                            logger.info("Found %d partial results after timeout", len(partial_recommendations))
                            metrics_payload["partial_recommendations_found"] = len(partial_recommendations)
                    except Exception as exc:
                        logger.warning("Could not retrieve partial results after timeout: %s", exc)

                    await concurrency_controller.atomic_status_update_with_precondition(
                        csv_upload_id,
                        "bundle_generation_failed",
                        expected_current_status="generating_bundles",
                        additional_fields={
                            "error_message": timeout_message,
                            "bundle_generation_metrics": metrics_payload,
                        },
                    )
                    return
                finally:
                    if watchdog_task:
                        try:
                            watchdog_task.cancel()
                        except asyncio.CancelledError:
                            # Task was already cancelled, this is fine
                            pass
                        except Exception as e:
                            logger.debug(f"Error cancelling watchdog task: {e}")
                
                if isinstance(generation_result, dict) and generation_result.get("async_deferred"):
                    metrics = generation_result.get("metrics", {})
                    dataset_profile = generation_result.get("dataset_profile") or metrics.get("dataset_profile") or {}
                    logger.info(
                        "Async deferral acknowledged %s | dataset_profile=%s",
                        scope,
                        dataset_profile,
                    )
                    status_payload = {
                        "bundle_generation_metrics": {
                            **metrics,
                            "dataset_profile": dataset_profile,
                            "async_deferred": True,
                            "deferred_at": datetime.utcnow().isoformat(),
                        }
                    }
                    async_status = await concurrency_controller.atomic_status_update_with_precondition(
                        csv_upload_id,
                        "bundle_generation_async",
                        expected_current_status="generating_bundles",
                        additional_fields=status_payload,
                    )
                    if not async_status["success"]:
                        logger.warning(
                            "Failed to transition upload %s to async state: %s",
                            csv_upload_id,
                            async_status["error"],
                        )
                    await update_generation_progress(
                        csv_upload_id,
                        step="optimization",
                        progress=78,
                        status="in_progress",
                        message="Continuing remaining phases asynchronously.",
                        metadata={"dataset_profile": dataset_profile},
                    )
                    pipeline_scheduler.schedule(_resume_bundle_generation(csv_upload_id))
                    return

                # Calculate generation metrics
                generation_time = time.time() - start_time
                
                # Log and store comprehensive generation metrics
                if isinstance(generation_result, dict) and "metrics" in generation_result:
                    metrics = generation_result["metrics"]
                    logger.info(f"Bundle generation completed {scope} in {generation_time:.2f}s")
                    logger.info(f"Generation metrics: {metrics}")
                    
                    # Log per-type bundle counts
                    bundle_counts = metrics.get("bundle_counts", {})
                    total_bundles = sum(bundle_counts.values())
                    logger.info(f"Total bundles generated: {total_bundles}")
                    for bundle_type, count in bundle_counts.items():
                        logger.info(f"  {bundle_type}: {count} bundles")
                    
                    # Log drop reasons summary
                    drop_reasons = metrics.get("drop_reasons", {})
                    if drop_reasons:
                        total_dropped = sum(drop_reasons.values())
                        logger.info(f"Total bundles dropped: {total_dropped}")
                        for reason, count in drop_reasons.items():
                            logger.info(f"  {reason}: {count} bundles")
                    
                    # Store metrics in database
                    enhanced_metrics = {
                        **metrics,
                        "generation_time_seconds": round(generation_time, 2),
                        "generation_timestamp": datetime.now().isoformat(),
                        "total_bundles_generated": total_bundles,
                        "total_bundles_dropped": sum(drop_reasons.values()) if drop_reasons else 0,
                        "shop_id": shop_id  # Include shop_id in metrics
                    }

                    # Merge previously stored checkpoint metadata if present
                    try:
                        existing_upload = await storage.get_csv_upload(csv_upload_id)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning(f"Unable to load existing metrics for merge: {exc}")
                        existing_upload = None

                    if existing_upload and getattr(existing_upload, "bundle_generation_metrics", None):
                        existing_metrics_state = dict(existing_upload.bundle_generation_metrics)
                        if existing_metrics_state.get("checkpoints"):
                            enhanced_metrics.setdefault(
                                "checkpoints", existing_metrics_state.get("checkpoints")
                            )
                        if existing_metrics_state.get("last_checkpoint"):
                            enhanced_metrics.setdefault(
                                "last_checkpoint", existing_metrics_state.get("last_checkpoint")
                            )
                        if existing_metrics_state.get("latest_metrics_snapshot"):
                            enhanced_metrics.setdefault(
                                "latest_metrics_snapshot",
                                existing_metrics_state.get("latest_metrics_snapshot")
                            )
                    
                    # Atomically store metrics with compare-and-set
                    logger.info(f"Bundle generation {scope}: persisting bundle_generation_metrics field")
                    metrics_update = await concurrency_controller.atomic_status_update_with_precondition(
                        csv_upload_id,
                        "generating_bundles",  # Keep current status
                        expected_current_status="generating_bundles",
                        additional_fields={"bundle_generation_metrics": enhanced_metrics}
                    )
                    
                    if metrics_update["success"]:
                        logger.info(f"Stored generation metrics for CSV upload {csv_upload_id}")
                    else:
                        logger.warning(f"Failed to store generation metrics: {metrics_update['error']}")
                else:
                    logger.info(f"Bundle generation completed {scope} in {generation_time:.2f}s")
                
                current_status = None
                try:
                    upload_row = await storage.get_csv_upload(csv_upload_id)
                    current_status = getattr(upload_row, "status", None)
                except Exception as exc:
                    logger.warning(
                        "Bundle generation %s: unable to fetch upload state prior to completion update: %s",
                        scope,
                        exc,
                    )

                terminal_statuses = {
                    "bundle_generation_failed",
                    "bundle_generation_timed_out",
                    "bundle_generation_cancelled",
                }

                if current_status in terminal_statuses:
                    logger.info(
                        "Bundle generation %s: skipping completion overwrite because status is already %s",
                        scope,
                        current_status,
                    )
                    # Still emit finalization progress so frontend knows generation is complete
                    try:
                        actual_bundle_count = await storage.get_bundle_count_for_upload(csv_upload_id)
                        generation_time = time.time() - start_time
                        await update_generation_progress(
                            csv_upload_id,
                            step="finalization",
                            progress=100,
                            status="failed" if "failed" in current_status else "completed",
                            message=f"Generation ended with status: {current_status}",
                            bundle_count=actual_bundle_count,
                            metadata={
                                "terminal_status": current_status,
                                "generation_time_seconds": round(generation_time, 2),
                            },
                        )
                    except Exception as progress_exc:
                        logger.warning(f"Failed to emit terminal status progress: {progress_exc}")
                else:
                    logger.info(
                        f"Bundle generation {scope}: attempting status transition to bundle_generation_completed"
                    )
                    completion_update = await concurrency_controller.atomic_status_update_with_precondition(
                        csv_upload_id,
                        "bundle_generation_completed",
                        expected_current_status="generating_bundles",
                    )

                    if completion_update["success"]:
                        logger.info(f"Bundle generation completed successfully for shop {shop_id}")
                    else:
                        safe_transition = await storage.safe_mark_upload_completed(csv_upload_id)
                        if safe_transition:
                            logger.info(
                                "Bundle generation %s: CAS fallback succeeded via safe_mark_upload_completed",
                                scope,
                            )
                        else:
                            logger.warning(
                                "Bundle generation %s: failed to mark as completed (%s) and safe fallback rejected",
                                scope,
                                completion_update["error"],
                            )

                    # CRITICAL: Always emit finalization progress AFTER DB commit
                    # This ensures frontend receives the completion signal with accurate bundle count
                    try:
                        # Get actual bundle count from DB for accurate reporting
                        actual_bundle_count = await storage.get_bundle_count_for_upload(csv_upload_id)
                        generation_time = time.time() - start_time

                        # Zero bundles is a valid outcome (not a failure) - always completed
                        final_status = "completed"
                        final_message = (
                            f"Bundle generation complete: {actual_bundle_count} bundles in {generation_time:.1f}s"
                            if actual_bundle_count > 0
                            else "Analysis complete - no bundle patterns found in data"
                        )

                        await update_generation_progress(
                            csv_upload_id,
                            step="finalization",
                            progress=100,
                            status=final_status,
                            message=final_message,
                            bundle_count=actual_bundle_count,
                            metadata={
                                "generation_time_seconds": round(generation_time, 2),
                                "bundles_in_db": actual_bundle_count,
                                "pipeline": "v2_full",
                            },
                        )
                        logger.info(
                            f"Bundle generation {scope}: emitted finalization progress | "
                            f"status={final_status} bundle_count={actual_bundle_count}"
                        )
                    except Exception as progress_exc:
                        # Don't fail the generation if progress update fails
                        logger.warning(
                            f"Bundle generation {scope}: failed to emit finalization progress: {progress_exc}"
                        )
                
            except Exception as e:
                # Bundle generation failed - update status atomically
                generation_time = time.time() - start_time
                logger.error(
                    f"[{csv_upload_id}] ❌ BUNDLE GENERATION FAILED (inner try block)\n"
                    f"  Shop: {shop_id}\n"
                    f"  Duration: {generation_time:.2f}s\n"
                    f"  Error type: {type(e).__name__}\n"
                    f"  Error message: {str(e)}\n"
                    f"  Full traceback:\n{traceback.format_exc()}"
                )

                # Log additional details if this was a timeout or infinite loop issue
                if "timeout" in str(e).lower() or generation_time > 300:
                    logger.error(f"CRITICAL: Bundle generation appears to have hit infinite loop or timeout issue")
                    logger.error(f"Generation time: {generation_time:.2f}s")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error details: {str(e)}")
                
                failure_update = await concurrency_controller.atomic_status_update_with_precondition(
                    csv_upload_id, 
                    "bundle_generation_failed",
                    expected_current_status="generating_bundles",
                    additional_fields={"error_message": f"Bundle generation failed: {str(e)}"}
                )
                
                if not failure_update["success"]:
                    logger.error(f"Failed to update failure status: {failure_update['error']}")

                await update_generation_progress(
                    csv_upload_id,
                    step="finalization",
                    progress=100,
                    status="failed",
                    message=f"Bundle generation failed: {e}",
                )
                
                raise
            
            # Lock is automatically released when exiting the context manager
            
    except asyncio.TimeoutError as e:
        generation_time = time.time() - start_time
        logger.error(
            f"[{csv_upload_id}] ⏱️ TIMEOUT: Bundle generation process timeout!\n"
            f"  Duration: {generation_time:.2f}s\n"
            f"  Timeout limit: {BUNDLE_GENERATION_TIMEOUT_SECONDS}s\n"
            f"  Error: {str(e)}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        # Update status without lock since this was a timeout
        await concurrency_controller.atomic_status_update_with_precondition(
            csv_upload_id, 
            "bundle_generation_failed",
            additional_fields={"error_message": f"Process timeout after {BUNDLE_GENERATION_TIMEOUT_SECONDS} seconds: {str(e)}"}
        )
        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=f"Process timeout after {BUNDLE_GENERATION_TIMEOUT_SECONDS} seconds: {e}",
        )
        return
        
    except TimeoutError as e:
        generation_time = time.time() - start_time
        logger.error(
            f"[{csv_upload_id}] 🔒 LOCK TIMEOUT: Could not acquire bundle generation lock!\n"
            f"  Duration: {generation_time:.2f}s\n"
            f"  Error: {str(e)}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        # Update status without lock since we couldn't acquire it
        await concurrency_controller.atomic_status_update_with_precondition(
            csv_upload_id, 
            "bundle_generation_failed",
            additional_fields={"error_message": f"Lock timeout: {str(e)}"}
        )
        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=f"Lock timeout: {e}",
        )
        return
        
    except ValueError as e:
        generation_time = time.time() - start_time
        logger.error(
            f"[{csv_upload_id}] ❌ INVALID: CSV upload or shop identification error!\n"
            f"  Duration: {generation_time:.2f}s\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error: {str(e)}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        await concurrency_controller.atomic_status_update_with_precondition(
            csv_upload_id, 
            "bundle_generation_failed",
            additional_fields={"error_message": f"Invalid CSV upload: {str(e)}"}
        )
        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=f"Invalid CSV upload: {e}",
        )
        return
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(
            f"[{csv_upload_id}] 💥 UNEXPECTED ERROR in bundle generation!\n"
            f"  Duration: {generation_time:.2f}s\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error message: {str(e)}\n"
            f"  Full traceback:\n{traceback.format_exc()}"
        )
        
        # Update status without lock in case of unexpected errors
        await concurrency_controller.atomic_status_update_with_precondition(
            csv_upload_id, 
            "bundle_generation_failed",
            additional_fields={"error_message": f"Unexpected error: {str(e)}"}
        )
        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=f"Unexpected error: {e}",
        )
        raise


def _resume_bundle_generation(csv_upload_id: str):
    async def _runner():
        await asyncio.sleep(0.1)
        await generate_bundles_background(csv_upload_id, resume_only=True)

    return _runner


def _run_full_generation_after_quickstart(csv_upload_id: str, shop_id: str):
    """Schedule full bundle generation after quick-start preview is complete.

    This runs the comprehensive v2 pipeline in the background while the merchant
    sees the quick-start preview bundles immediately. When complete, the full
    bundles replace the preview bundles and the shop sync status is marked complete.
    """
    async def _runner():
        # Wait a bit before starting full generation to avoid overload
        await asyncio.sleep(5)

        logger.info(
            f"[{csv_upload_id}] 🔄 Starting full bundle generation after quick-start\n"
            f"  Shop: {shop_id}\n"
            f"  Mode: Comprehensive v2 pipeline\n"
            f"  Timeout: {BUNDLE_GENERATION_SOFT_WATCHDOG_SECONDS}s soft watchdog"
        )

        try:
            # Clear quick-start bundles before full generation
            quick_start_bundles = []
            try:
                from sqlalchemy import select, and_
                async with storage.get_session() as session:
                    query = select(BundleRecommendation).where(
                        and_(
                            BundleRecommendation.csv_upload_id == csv_upload_id,
                            BundleRecommendation.discount_reference.like("__quick_start_%")
                        )
                    )
                    result = await session.execute(query)
                    quick_start_bundles = list(result.scalars().all())

                if quick_start_bundles:
                    logger.info(
                        f"[{csv_upload_id}] Found {len(quick_start_bundles)} quick-start bundles to replace"
                    )
            except Exception as e:
                logger.warning(f"[{csv_upload_id}] Could not query quick-start bundles: {e}")

            # Run full generation (this will replace quick-start bundles)
            from services.bundle_generator import BundleGenerator
            generator = BundleGenerator()

            # Resolve all upload IDs for full generation
            catalog_upload_id = None
            variants_upload_id = None
            inventory_upload_id = None
            try:
                async with AsyncSessionLocal() as db:
                    _, catalog_id, variants_id, inventory_id, _ = await _resolve_all_uploads_from_run(csv_upload_id, db)
                    catalog_upload_id = catalog_id
                    variants_upload_id = variants_id
                    inventory_upload_id = inventory_id
            except Exception as resolve_exc:
                logger.warning(
                    f"[{csv_upload_id}] Could not resolve upload IDs for full generation: {resolve_exc}"
                )

            # Use the normal generate_bundle_recommendations method
            generation_result = await generator.generate_bundle_recommendations(
                csv_upload_id,
                catalog_upload_id=catalog_upload_id,
                variants_upload_id=variants_upload_id,
                inventory_upload_id=inventory_upload_id,
            )

            total_recs = generation_result.get('metrics', {}).get('total_recommendations', 0)

            logger.info(
                f"[{csv_upload_id}] ✅ Full generation complete after quick-start\n"
                f"  Total bundles: {total_recs}\n"
                f"  Processing time: {generation_result.get('metrics', {}).get('processing_time_ms', 0) / 1000:.2f}s"
            )

            # Mark shop sync as completed now that full generation is done
            try:
                await storage.mark_shop_sync_completed(shop_id)
                logger.info(f"[{csv_upload_id}] Marked shop {shop_id} initial sync as completed")
            except Exception as e:
                logger.warning(f"[{csv_upload_id}] Failed to mark sync completed: {e}")

            # Notify user that full bundles are ready
            try:
                await notify_bundle_ready(csv_upload_id, generation_result.get('metrics', {}))
            except Exception as notify_exc:
                logger.warning(f"[{csv_upload_id}] Failed to send full-generation notification: {notify_exc}")

        except Exception as e:
            logger.error(
                f"[{csv_upload_id}] ❌ Full generation after quick-start failed: {e}",
                exc_info=True
            )
            # Don't fail hard - merchant already has preview bundles

    return _runner


@router.post("/retry-bundle-generation/{upload_id}")
async def retry_bundle_generation(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retry bundle generation for a stuck upload.

    This endpoint resets an upload that's stuck in 'generating_bundles' status
    (e.g., due to Cloud Run killing the async task) and re-triggers bundle generation.
    """
    from services.bundle_auto_trigger import maybe_trigger_bundle_generation

    # First, resolve to the orders upload if needed
    try:
        orders_upload_id, run_id, source_upload = await _resolve_orders_upload(upload_id, db)
    except Exception as e:
        logger.error(f"Failed to resolve upload {upload_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")

    # Get current status
    current_upload = await storage.get_csv_upload(orders_upload_id)
    if not current_upload:
        raise HTTPException(status_code=404, detail=f"Upload not found: {orders_upload_id}")

    current_status = getattr(current_upload, "status", None)
    logger.info(f"Retry requested for upload {orders_upload_id}, current status: {current_status}")

    # Only allow retry for stuck states
    retriable_statuses = {"generating_bundles", "bundle_generation_failed", "bundle_generation_queued"}
    if current_status not in retriable_statuses:
        return {
            "success": False,
            "message": f"Upload status '{current_status}' is not retriable. Allowed: {retriable_statuses}",
            "upload_id": orders_upload_id,
            "current_status": current_status,
        }

    # Reset status to 'completed' so auto-trigger can fire
    try:
        await storage.update_csv_upload(orders_upload_id, {"status": "completed"})
        logger.info(f"Reset upload {orders_upload_id} status from '{current_status}' to 'completed'")
    except Exception as e:
        logger.error(f"Failed to reset upload status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset upload status: {e}")

    # Clear any existing progress record to start fresh
    try:
        await update_generation_progress(
            orders_upload_id,
            step="retry_requested",
            progress=0,
            status="in_progress",
            message="Retrying bundle generation...",
        )
    except Exception as e:
        logger.warning(f"Failed to reset progress record: {e}")

    # Trigger bundle generation (this will run synchronously on Cloud Run)
    try:
        await maybe_trigger_bundle_generation(orders_upload_id)
        logger.info(f"Re-triggered bundle generation for upload {orders_upload_id}")
    except Exception as e:
        logger.error(f"Failed to trigger bundle generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger bundle generation: {e}")

    return {
        "success": True,
        "message": "Bundle generation retry triggered successfully",
        "upload_id": orders_upload_id,
        "run_id": run_id,
        "previous_status": current_status,
    }
