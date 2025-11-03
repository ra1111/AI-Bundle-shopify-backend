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

from database import get_db, BundleRecommendation
from services.bundle_generator import BundleGenerator
from services.progress_tracker import update_generation_progress
from routers.uploads import _resolve_orders_upload, _ensure_data_ready
from settings import resolve_shop_id
from services.storage import storage
from services.pipeline_scheduler import pipeline_scheduler

logger = logging.getLogger(__name__)
router = APIRouter()

BUNDLE_GENERATION_TIMEOUT_SECONDS = int(os.getenv("BUNDLE_GENERATION_TIMEOUT_SECONDS", "320"))

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
    shopId: Optional[str] = None,
    uploadId: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get bundle recommendations with optional CSV filtering"""
    try:
        shop_id = resolve_shop_id(shopId)
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

async def generate_bundles_background(csv_upload_id: Optional[str], resume_only: bool = False):
    """Background task to generate bundle recommendations with concurrency control and async deferrals."""
    from services.concurrency_control import concurrency_controller
    
    start_time = time.time()
    scope = f"for CSV upload {csv_upload_id}" if csv_upload_id else "overall"
    logger.info(f"{'Resuming' if resume_only else 'Starting'} bundle generation {scope}")
    
    if not csv_upload_id:
        logger.error("Bundle generation requires a valid CSV upload ID")
        return
    
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
            
            try:
                # Generate bundle recommendations with AGGRESSIVE timeout protection
                from services.bundle_generator import BundleGenerator
                generator = BundleGenerator()
                
                # HARD TIMEOUT: Force termination after configured ceiling to protect request lifecycle
                try:
                    logger.info(
                        f"Bundle generation {scope}: invoking generator with "
                        f"{BUNDLE_GENERATION_TIMEOUT_SECONDS}s timeout"
                    )
                    generation_result = await asyncio.wait_for(
                        generator.generate_bundle_recommendations(csv_upload_id),
                        timeout=float(BUNDLE_GENERATION_TIMEOUT_SECONDS),
                    )
                    logger.info(f"Bundle generation {scope}: generator completed without timeout")
                except asyncio.TimeoutError:
                    logger.error(
                        f"TIMEOUT: Bundle generation exceeded {BUNDLE_GENERATION_TIMEOUT_SECONDS} seconds for "
                        f"{csv_upload_id}, force terminating"
                    )
                    # Return a timeout result instead of letting it hang
                    generation_result = {
                        "recommendations": [],
                        "metrics": {
                            "timeout_error": True,
                            "total_recommendations": 0,
                            "processing_time_ms": BUNDLE_GENERATION_TIMEOUT_SECONDS * 1000,
                            "timeout_reason": f"{BUNDLE_GENERATION_TIMEOUT_SECONDS}_second_hard_limit_exceeded"
                        },
                        "v2_pipeline": True,
                        "csv_upload_id": csv_upload_id
                    }
                    
                    await update_generation_progress(
                        csv_upload_id,
                        step="finalization",
                        progress=100,
                        status="failed",
                        message="Bundle generation timed out after 6 minutes.",
                    )
                    
                    # Try to get any partial results that might have been persisted
                    try:
                        from database import get_db
                        from sqlalchemy import select
                        from database import BundleRecommendation
                        async with get_db() as db:
                            query = select(BundleRecommendation).where(
                                BundleRecommendation.csv_upload_id == csv_upload_id,
                                BundleRecommendation.shop_id == shop_id
                            )
                            result = await db.execute(query)
                            partial_recommendations = result.scalars().all()
                            if partial_recommendations:
                                logger.info(f"Found {len(partial_recommendations)} partial results after timeout")
                                generation_result["metrics"]["partial_recommendations_found"] = len(partial_recommendations)
                    except Exception as e:
                        logger.warning(f"Could not retrieve partial results after timeout: {e}")
                
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
                
                # Atomically update status to completed with precondition
                logger.info(f"Bundle generation {scope}: attempting status transition to bundle_generation_completed")
                completion_update = await concurrency_controller.atomic_status_update_with_precondition(
                    csv_upload_id, 
                    "bundle_generation_completed",
                    expected_current_status="generating_bundles"
                )
                
                if completion_update["success"]:
                    logger.info(f"Bundle generation completed successfully for shop {shop_id}")
                else:
                    logger.warning(f"Failed to mark generation as completed: {completion_update['error']}")
                
            except Exception as e:
                # Bundle generation failed - update status atomically
                generation_time = time.time() - start_time
                logger.exception(f"Bundle generation failed for shop {shop_id} after {generation_time:.2f}s: {e}")

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
        logger.error(f"TIMEOUT: Bundle generation process timeout for CSV upload {csv_upload_id}: {e}")
        # Update status without lock since this was a timeout
        await concurrency_controller.atomic_status_update_with_precondition(
            csv_upload_id, 
            "bundle_generation_failed",
            additional_fields={"error_message": f"Process timeout after 6 minutes: {str(e)}"}
        )
        await update_generation_progress(
            csv_upload_id,
            step="finalization",
            progress=100,
            status="failed",
            message=f"Process timeout after 6 minutes: {e}",
        )
        return
        
    except TimeoutError as e:
        logger.error(f"Could not acquire bundle generation lock for CSV upload {csv_upload_id}: {e}")
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
        logger.error(f"Invalid CSV upload or shop identification: {e}")
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
        logger.exception(f"Unexpected error in bundle generation after {generation_time:.2f}s: {e}")
        
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
