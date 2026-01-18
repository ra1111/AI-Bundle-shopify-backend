"""
Shopify data upload endpoints
Accepts CSV payloads from the Remix embedded app and feeds the existing pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import asyncio
import json as json_lib

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal, BundleRecommendation, CsvUpload, GenerationProgress, get_db
from routers.bundle_recommendations import generate_bundles_background
from routers.uploads import (
    _ensure_data_ready,
    _resolve_orders_upload,
    process_csv_background,
)
from services.storage import storage
from settings import resolve_shop_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shopify", tags=["Shopify Integration"])


class ShopifyUploadRequest(BaseModel):
    """Payload posted by the Remix frontend."""

    shop_id: str = Field(..., alias="shopId", min_length=1, max_length=255)
    csv_type: str = Field(..., alias="csvType", min_length=1, max_length=50)
    csv_data: str = Field(..., alias="csvData", min_length=1)
    run_id: Optional[str] = Field(None, alias="runId", max_length=255)
    trigger_pipeline: bool = Field(False, alias="triggerPipeline")

    model_config = ConfigDict(populate_by_name=True)

    @property
    def csv_size_mb(self) -> float:
        """Calculate CSV data size in megabytes."""
        return len(self.csv_data.encode('utf-8')) / (1024 * 1024)


class BatchUploadRequest(BaseModel):
    """Batch upload of all 4 CSV types in a single request."""

    shop_id: str = Field(..., alias="shopId", min_length=1, max_length=255)
    run_id: Optional[str] = Field(None, alias="runId", max_length=255)
    catalog_csv: str = Field(..., alias="catalogCsv", min_length=1)
    variants_csv: str = Field(..., alias="variantsCsv", min_length=1)
    inventory_csv: str = Field(..., alias="inventoryCsv", min_length=1)
    orders_csv: str = Field(..., alias="ordersCsv", min_length=1)

    model_config = ConfigDict(populate_by_name=True)

    @property
    def total_size_mb(self) -> float:
        """Calculate total CSV data size in megabytes."""
        total_bytes = (
            len(self.catalog_csv.encode('utf-8')) +
            len(self.variants_csv.encode('utf-8')) +
            len(self.inventory_csv.encode('utf-8')) +
            len(self.orders_csv.encode('utf-8'))
        )
        return total_bytes / (1024 * 1024)


# Maximum CSV size in MB (configurable via environment)
import os
MAX_CSV_SIZE_MB = float(os.getenv("MAX_CSV_SIZE_MB", "100"))


class UploadStatusResponse(BaseModel):
    upload_id: str
    shop_id: Optional[str]
    status: str
    total_rows: int
    processed_rows: int
    error_message: Optional[str]
    bundle_count: Optional[int]


class RecommendationResponse(BaseModel):
    id: str
    bundle_type: str
    objective: str
    products: Any
    pricing: Any
    ai_copy: Any
    confidence: Optional[float]
    predicted_lift: Optional[float]
    ranking_score: Optional[float]
    is_approved: bool
    created_at: Optional[str]


class RecommendationsEnvelope(BaseModel):
    shop_id: str
    count: int
    recommendations: list[RecommendationResponse]


@router.post("/upload")
async def shopify_upload(
    request: ShopifyUploadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Frontend sends CSV strings harvested from Shopify GraphQL.
    We write a CsvUpload record and reuse the existing CSV processor in the background.
    """

    # Validate CSV size
    csv_size = request.csv_size_mb
    logger.info(
        "[shopify_upload] Received upload shop_id=%s type=%s trigger_pipeline=%s size=%.2fMB",
        request.shop_id,
        request.csv_type,
        request.trigger_pipeline,
        csv_size,
    )

    if csv_size > MAX_CSV_SIZE_MB:
        logger.warning(
            "[shopify_upload] CSV too large: %.2fMB > %.2fMB limit",
            csv_size,
            MAX_CSV_SIZE_MB,
        )
        raise HTTPException(
            status_code=413,
            detail=f"CSV data too large: {csv_size:.2f}MB exceeds {MAX_CSV_SIZE_MB}MB limit",
        )

    alias_map = {
        "products": "catalog_joined",
        "catalog": "catalog_joined",
        "catalog_joined": "catalog_joined",
        "orders": "orders",
        "variants": "variants",
        "inventory": "inventory_levels",
        "inventory_levels": "inventory_levels",
    }
    normalized_type = alias_map.get(request.csv_type, request.csv_type)

    valid_types = {"orders", "catalog_joined", "variants", "inventory_levels"}
    if normalized_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid csvType {request.csv_type!r}. Must be one of: {', '.join(sorted(valid_types))}",
        )

    effective_run_id = (request.run_id or str(uuid4())).strip()

    # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
    from settings import sanitize_shop_id
    resolved_shop_id = sanitize_shop_id(request.shop_id)
    if not resolved_shop_id:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid shop_id: '{request.shop_id}'. shop_id is required and cannot be empty."
        )

    upload_id = str(uuid4())
    upload = CsvUpload(
        id=upload_id,
        filename=f"shopify_sync_{normalized_type}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
        csv_type=normalized_type,
        run_id=effective_run_id,
        shop_id=resolved_shop_id,
        total_rows=0,
        processed_rows=0,
        status="processing",
        error_message=None,
        code_version="1.0.0",
        schema_version="1.0",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(upload)
    await db.commit()
    await db.refresh(upload)

    logger.info(
        "[shopify_upload] Created CsvUpload id=%s run_id=%s shop_id=%s csv_type=%s",
        upload_id,
        effective_run_id,
        resolved_shop_id,
        normalized_type,
    )

    background_tasks.add_task(
        process_shopify_upload_background,
        request.csv_data,
        upload_id,
        normalized_type,
        request.trigger_pipeline,
    )

    return {
        "uploadId": upload_id,
        "runId": effective_run_id,
        "status": "processing",
        "shopId": resolved_shop_id,
        "csvType": normalized_type,
        "triggerPipeline": request.trigger_pipeline,
        "message": "Upload accepted. Poll /api/shopify/status/{uploadId} for progress.",
    }


@router.post("/upload-batch")
async def shopify_upload_batch(
    request: BatchUploadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Batch upload all 4 CSV types in a single request.
    More efficient than 4 separate uploads - single roundtrip, auto-triggers pipeline.
    """
    total_size = request.total_size_mb
    logger.info(
        "[shopify_upload_batch] Received batch upload shop_id=%s size=%.2fMB",
        request.shop_id,
        total_size,
    )

    # Validate total size (allow 4x single limit since we have 4 CSVs)
    max_batch_size = MAX_CSV_SIZE_MB * 4
    if total_size > max_batch_size:
        logger.warning(
            "[shopify_upload_batch] Batch too large: %.2fMB > %.2fMB limit",
            total_size,
            max_batch_size,
        )
        raise HTTPException(
            status_code=413,
            detail=f"Batch data too large: {total_size:.2f}MB exceeds {max_batch_size}MB limit",
        )

    effective_run_id = (request.run_id or str(uuid4())).strip()

    # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
    from settings import sanitize_shop_id
    resolved_shop_id = sanitize_shop_id(request.shop_id)
    if not resolved_shop_id:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid shop_id: '{request.shop_id}'. shop_id is required and cannot be empty."
        )

    # Create a single "primary" upload record for orders (used for status tracking)
    # All 4 CSVs share the same run_id
    orders_upload_id = str(uuid4())
    catalog_upload_id = str(uuid4())
    variants_upload_id = str(uuid4())
    inventory_upload_id = str(uuid4())

    timestamp = datetime.utcnow()
    timestamp_str = timestamp.strftime('%Y%m%dT%H%M%SZ')

    # Create all 4 upload records
    uploads = [
        CsvUpload(
            id=orders_upload_id,
            filename=f"shopify_sync_orders_{timestamp_str}.csv",
            csv_type="orders",
            run_id=effective_run_id,
            shop_id=resolved_shop_id,
            total_rows=0,
            processed_rows=0,
            status="processing",
            error_message=None,
            code_version="1.0.0",
            schema_version="1.0",
            created_at=timestamp,
            updated_at=timestamp,
        ),
        CsvUpload(
            id=catalog_upload_id,
            filename=f"shopify_sync_catalog_joined_{timestamp_str}.csv",
            csv_type="catalog_joined",
            run_id=effective_run_id,
            shop_id=resolved_shop_id,
            total_rows=0,
            processed_rows=0,
            status="processing",
            error_message=None,
            code_version="1.0.0",
            schema_version="1.0",
            created_at=timestamp,
            updated_at=timestamp,
        ),
        CsvUpload(
            id=variants_upload_id,
            filename=f"shopify_sync_variants_{timestamp_str}.csv",
            csv_type="variants",
            run_id=effective_run_id,
            shop_id=resolved_shop_id,
            total_rows=0,
            processed_rows=0,
            status="processing",
            error_message=None,
            code_version="1.0.0",
            schema_version="1.0",
            created_at=timestamp,
            updated_at=timestamp,
        ),
        CsvUpload(
            id=inventory_upload_id,
            filename=f"shopify_sync_inventory_levels_{timestamp_str}.csv",
            csv_type="inventory_levels",
            run_id=effective_run_id,
            shop_id=resolved_shop_id,
            total_rows=0,
            processed_rows=0,
            status="processing",
            error_message=None,
            code_version="1.0.0",
            schema_version="1.0",
            created_at=timestamp,
            updated_at=timestamp,
        ),
    ]

    for upload in uploads:
        db.add(upload)
    await db.commit()

    logger.info(
        "[shopify_upload_batch] Created 4 CsvUpload records run_id=%s shop_id=%s orders_id=%s",
        effective_run_id,
        resolved_shop_id,
        orders_upload_id,
    )

    # Queue single background task to process all CSVs and trigger pipeline
    background_tasks.add_task(
        process_batch_upload_background,
        request.catalog_csv,
        request.variants_csv,
        request.inventory_csv,
        request.orders_csv,
        catalog_upload_id,
        variants_upload_id,
        inventory_upload_id,
        orders_upload_id,
        effective_run_id,
    )

    return {
        "uploadId": orders_upload_id,  # Primary ID for status tracking
        "runId": effective_run_id,
        "status": "processing",
        "shopId": resolved_shop_id,
        "catalogUploadId": catalog_upload_id,
        "variantsUploadId": variants_upload_id,
        "inventoryUploadId": inventory_upload_id,
        "ordersUploadId": orders_upload_id,
        "message": "Batch upload accepted. Poll /api/shopify/status/{uploadId} for progress.",
    }


@router.get("/status/{upload_id}", response_model=UploadStatusResponse)
async def get_upload_status(upload_id: str, db: AsyncSession = Depends(get_db)):
    """Check processing status for a CsvUpload created via the Shopify endpoint.

    Accepts either an upload_id (primary key) or a run_id (shared across all 4 CSVs).
    When run_id is provided, returns status for the orders upload in that run.
    """

    logger.info(f"🔍 STATUS CHECK: upload_id={upload_id}")

    upload = await db.get(CsvUpload, upload_id)

    # If not found by upload_id, try looking up by run_id
    if not upload:
        run_id_stmt = (
            select(CsvUpload)
            .where(CsvUpload.run_id == upload_id)
            .where(CsvUpload.csv_type == "orders")
            .order_by(CsvUpload.created_at.desc())
            .limit(1)
        )
        result = await db.execute(run_id_stmt)
        upload = result.scalar_one_or_none()
        if upload:
            logger.info(f"✅ Resolved run_id {upload_id} -> orders upload {upload.id}")

    if not upload:
        logger.warning(f"❌ STATUS CHECK: upload {upload_id} NOT FOUND")
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

    logger.info(
        f"✅ STATUS CHECK: upload_id={upload_id} status={upload.status} "
        f"rows={upload.processed_rows}/{upload.total_rows}"
    )

    # Check for bundles regardless of status (they might exist from previous runs)
    bundle_count: Optional[int] = None
    completed_statuses = {
        "completed",
        "bundle_generation_completed",
        "bundle_generation_in_progress",
        "bundle_generation_queued",
        "bundle_generation_async",
    }

    if upload.status in completed_statuses or upload.status == "processing":
        # Always check for existing bundles
        # Use upload.id (resolved orders upload ID) not upload_id (might be run_id)
        stmt = select(func.count(BundleRecommendation.id)).where(
            BundleRecommendation.csv_upload_id == upload.id
        )
        result = await db.execute(stmt)
        count = result.scalar() or 0
        bundle_count = count if count > 0 else None

    # Normalize status values for frontend simplicity
    # Map internal detailed statuses to simple states
    frontend_status = upload.status
    if upload.status in {"bundle_generation_completed"}:
        frontend_status = "completed"
    elif upload.status in {"bundle_generation_in_progress", "bundle_generation_queued", "bundle_generation_async", "generating_bundles"}:
        frontend_status = "generating_bundles"
    elif upload.status in {"bundle_generation_failed", "bundle_generation_timed_out", "bundle_generation_cancelled"}:
        frontend_status = "failed"
    elif upload.status == "completed" and bundle_count is None:
        # CSV processing completed but no bundles yet - check if bundle generation is pending
        # This handles the race condition where CSV finishes before auto-trigger kicks in
        try:
            progress_stmt = select(GenerationProgress).where(
                GenerationProgress.upload_id == upload.id
            )
            progress_result = await db.execute(progress_stmt)
            progress_record = progress_result.scalar_one_or_none()
            if progress_record and progress_record.status == "in_progress":
                # Bundle generation is actually in progress - don't say "completed"
                frontend_status = "generating_bundles"
                logger.info(
                    f"📊 STATUS OVERRIDE: upload_id={upload_id} csv_status=completed but "
                    f"progress shows in_progress at step={progress_record.step} - returning generating_bundles"
                )
        except Exception as exc:
            logger.warning(f"Failed to check generation progress for {upload_id}: {exc}")

    logger.info(
        f"📊 STATUS RESPONSE: upload_id={upload_id} "
        f"internal_status={upload.status} → frontend_status={frontend_status} "
        f"bundle_count={bundle_count}"
    )

    return UploadStatusResponse(
        upload_id=upload.id,
        shop_id=upload.shop_id,
        status=frontend_status,  # Simplified status for frontend
        total_rows=upload.total_rows,
        processed_rows=upload.processed_rows,
        error_message=upload.error_message,
        bundle_count=bundle_count,
    )


@router.get("/progress/{upload_id}/stream")
async def stream_progress(upload_id: str):
    """
    Server-Sent Events (SSE) endpoint for real-time progress updates.
    More efficient than polling - pushes updates as they happen.

    Returns SSE stream with progress events:
    - event: progress
    - data: JSON with status, progress, step, bundle_count, etc.
    """
    async def event_generator():
        last_status = None
        last_progress = -1
        retry_count = 0
        max_retries = 3

        while True:
            try:
                async with AsyncSessionLocal() as db:
                    # Try to find upload by ID or run_id
                    upload = await db.get(CsvUpload, upload_id)

                    if not upload:
                        run_id_stmt = (
                            select(CsvUpload)
                            .where(CsvUpload.run_id == upload_id)
                            .where(CsvUpload.csv_type == "orders")
                            .order_by(CsvUpload.created_at.desc())
                            .limit(1)
                        )
                        result = await db.execute(run_id_stmt)
                        upload = result.scalar_one_or_none()

                    if not upload:
                        # Upload not found yet - send waiting event
                        event_data = {
                            "status": "waiting",
                            "message": "Waiting for upload to be created...",
                            "upload_id": upload_id,
                        }
                        yield f"event: progress\ndata: {json_lib.dumps(event_data)}\n\n"
                        retry_count += 1
                        if retry_count > max_retries * 10:  # Wait up to 30 seconds
                            event_data = {
                                "status": "error",
                                "message": f"Upload {upload_id} not found",
                            }
                            yield f"event: error\ndata: {json_lib.dumps(event_data)}\n\n"
                            break
                        await asyncio.sleep(1)
                        continue

                    retry_count = 0  # Reset retry count on success

                    # Check for bundles
                    bundle_count = None
                    stmt = select(func.count(BundleRecommendation.id)).where(
                        BundleRecommendation.csv_upload_id == upload.id
                    )
                    result = await db.execute(stmt)
                    count = result.scalar() or 0
                    bundle_count = count if count > 0 else None

                    # Get generation progress for step, message, metadata
                    # This is the source of truth for generation state
                    gen_progress = None
                    try:
                        gen_stmt = select(GenerationProgress).where(
                            GenerationProgress.upload_id == upload.id
                        )
                        gen_result = await db.execute(gen_stmt)
                        gen_progress = gen_result.scalar_one_or_none()
                    except Exception:
                        pass  # Fall back to CsvUpload-based progress

                    # Use generation_progress if available, else derive from CsvUpload
                    if gen_progress:
                        step = gen_progress.step
                        progress = gen_progress.progress
                        message = gen_progress.message or ""
                        metadata = gen_progress.metadata_json if isinstance(gen_progress.metadata_json, dict) else {}
                        frontend_status = gen_progress.status
                        if frontend_status == "in_progress":
                            frontend_status = "processing"
                    else:
                        # Fallback: derive step from CsvUpload status
                        step = "queueing"
                        progress = 0
                        message = "Processing..."
                        metadata = {}
                        if upload.total_rows > 0:
                            progress = int((upload.processed_rows / upload.total_rows) * 100)

                        # Normalize status for frontend
                        frontend_status = upload.status
                        if upload.status in {"bundle_generation_completed"}:
                            frontend_status = "completed"
                            step = "finalization"
                            progress = 100
                        elif upload.status in {"bundle_generation_in_progress", "bundle_generation_queued", "bundle_generation_async"}:
                            frontend_status = "processing"
                            step = "ml_generation"
                        elif upload.status in {"bundle_generation_failed", "bundle_generation_timed_out", "bundle_generation_cancelled"}:
                            frontend_status = "failed"
                            message = upload.error_message or "Generation failed"

                    # Only send event if something changed
                    if frontend_status != last_status or progress != last_progress:
                        last_status = frontend_status
                        last_progress = progress

                        event_data = {
                            "upload_id": upload.id,
                            "step": step,
                            "status": frontend_status,
                            "progress": progress,
                            "message": message,
                            "bundle_count": bundle_count if bundle_count else 0,
                            "metadata": metadata,
                            "error_message": upload.error_message,
                        }
                        yield f"event: progress\ndata: {json_lib.dumps(event_data)}\n\n"

                    # Check for terminal states
                    if frontend_status in {"completed", "failed"}:
                        # Send final event and close
                        event_data = {
                            "upload_id": upload.id,
                            "step": "finalization" if frontend_status == "completed" else step,
                            "status": frontend_status,
                            "progress": 100 if frontend_status == "completed" else progress,
                            "bundle_count": bundle_count if bundle_count else 0,
                            "final": True,
                        }
                        yield f"event: complete\ndata: {json_lib.dumps(event_data)}\n\n"
                        break

                await asyncio.sleep(1)  # Check every 1 second (more responsive than polling)

            except Exception as e:
                logger.exception(f"[SSE] Error streaming progress for {upload_id}")
                event_data = {
                    "status": "error",
                    "message": str(e),
                }
                yield f"event: error\ndata: {json_lib.dumps(event_data)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get(
    "/recommendations/{shop_id}",
    response_model=RecommendationsEnvelope,
)
async def get_shop_recommendations(
    shop_id: str,
    approved: Optional[bool] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Fetch bundle recommendations for a shop (optionally restricted to approval state)."""

    # Only sanitize - NEVER fallback to a different shop
    from settings import sanitize_shop_id
    normalized_shop_id = sanitize_shop_id(shop_id)
    if not normalized_shop_id:
        raise HTTPException(status_code=400, detail="Invalid shop_id")

    stmt = (
        select(BundleRecommendation)
        .where(BundleRecommendation.shop_id == normalized_shop_id)
        .order_by(
            BundleRecommendation.ranking_score.desc(),
            BundleRecommendation.confidence.desc(),
            BundleRecommendation.created_at.desc(),
        )
        .limit(max(limit, 1))
    )

    if approved is not None:
        stmt = stmt.where(BundleRecommendation.is_approved == approved)

    result = await db.execute(stmt)
    recommendations = list(result.scalars().all())

    payload = [
        RecommendationResponse(
            id=rec.id,
            bundle_type=rec.bundle_type,
            objective=rec.objective,
            products=rec.products,
            pricing=rec.pricing,
            ai_copy=rec.ai_copy,
            confidence=float(rec.confidence) if rec.confidence is not None else None,
            predicted_lift=float(rec.predicted_lift)
            if rec.predicted_lift is not None
            else None,
            ranking_score=float(rec.ranking_score)
            if rec.ranking_score is not None
            else None,
            is_approved=bool(rec.is_approved),
            created_at=rec.created_at.isoformat() if rec.created_at else None,
        )
        for rec in recommendations
    ]

    return RecommendationsEnvelope(
        shop_id=normalized_shop_id,
        count=len(payload),
        recommendations=payload,
    )


@router.post("/recommendations/{recommendation_id}/approve")
async def approve_recommendation(
    recommendation_id: str,
    shop_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Mark a recommendation as approved for the given shop."""

    # Only sanitize - NEVER fallback to a different shop
    from settings import sanitize_shop_id
    normalized_shop_id = sanitize_shop_id(shop_id)
    if not normalized_shop_id:
        raise HTTPException(status_code=400, detail="Invalid shop_id")

    recommendation = await db.get(BundleRecommendation, recommendation_id)
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    # Compare exact shop IDs (both sanitized)
    rec_shop_id = sanitize_shop_id(recommendation.shop_id)
    if rec_shop_id != normalized_shop_id:
        raise HTTPException(status_code=403, detail="Not authorized for this shop")

    recommendation.is_approved = True
    await db.commit()

    logger.info(
        "[approve_recommendation] recommendation_id=%s shop_id=%s approved",
        recommendation_id,
        normalized_shop_id,
    )

    return {
        "recommendationId": recommendation_id,
        "status": "approved",
        "shopId": normalized_shop_id,
    }


async def process_shopify_upload_background(
    csv_content: str,
    upload_id: str,
    csv_type: str,
    trigger_pipeline: bool,
) -> None:
    """
    Background orchestration step.
    1. Reuse the existing CSV processor to ingest rows.
    2. Optionally run the bundle generation pipeline once data prerequisites are satisfied.
    """

    logger.info(
        "[process_shopify_upload] Starting ingestion upload_id=%s csv_type=%s trigger_pipeline=%s",
        upload_id,
        csv_type,
        trigger_pipeline,
    )

    try:
        await process_csv_background(csv_content, upload_id, csv_type)
    except Exception:
        logger.exception(
            "[process_shopify_upload] CSV processing failed upload_id=%s", upload_id
        )
        return

    if not trigger_pipeline:
        logger.info(
            "[process_shopify_upload] Ingestion complete upload_id=%s (pipeline skipped)",
            upload_id,
        )
        return

    logger.info(
        "[process_shopify_upload] Ingestion complete upload_id=%s. Preparing to launch pipeline.",
        upload_id,
    )

    async with AsyncSessionLocal() as session:
        try:
            orders_upload_id, run_id, source_upload = await _resolve_orders_upload(
                upload_id, session
            )
            await _ensure_data_ready(orders_upload_id, run_id, session)
        except HTTPException as exc:
            logger.warning(
                "[process_shopify_upload] Pipeline aborted upload_id=%s reason=%s",
                upload_id,
                exc.detail,
            )
            await storage.update_csv_upload(
                orders_upload_id,
                {
                    "error_message": f"Pipeline skipped: {exc.detail}",
                    "updated_at": datetime.utcnow(),
                },
            )
            return
        except Exception:
            logger.exception(
                "[process_shopify_upload] Failed readiness checks upload_id=%s", upload_id
            )
            await storage.update_csv_upload(
                orders_upload_id,
                {
                    "error_message": "Pipeline skipped due to unexpected readiness failure.",
                    "updated_at": datetime.utcnow(),
                },
            )
            return

    try:
        target_upload_id = (
            orders_upload_id if source_upload and source_upload.csv_type != "orders" else upload_id
        )
        await generate_bundles_background(target_upload_id)
        logger.info(
            "[process_shopify_upload] Pipeline completed for upload_id=%s",
            target_upload_id,
        )
    except Exception:
        logger.exception(
            "[process_shopify_upload] Pipeline execution failed upload_id=%s", upload_id
        )


async def process_batch_upload_background(
    catalog_csv: str,
    variants_csv: str,
    inventory_csv: str,
    orders_csv: str,
    catalog_upload_id: str,
    variants_upload_id: str,
    inventory_upload_id: str,
    orders_upload_id: str,
    run_id: str,
) -> None:
    """
    Background task for batch upload processing.
    Processes all 4 CSVs sequentially, then auto-triggers bundle generation pipeline.
    """
    logger.info(
        "[process_batch_upload] Starting batch ingestion run_id=%s orders_id=%s",
        run_id,
        orders_upload_id,
    )

    # Process CSVs in dependency order: catalog/variants first, then orders
    csv_tasks = [
        ("catalog_joined", catalog_csv, catalog_upload_id),
        ("variants", variants_csv, variants_upload_id),
        ("inventory_levels", inventory_csv, inventory_upload_id),
        ("orders", orders_csv, orders_upload_id),
    ]

    for csv_type, csv_content, upload_id in csv_tasks:
        try:
            logger.info(
                "[process_batch_upload] Processing %s upload_id=%s",
                csv_type,
                upload_id,
            )
            await process_csv_background(csv_content, upload_id, csv_type)
            logger.info(
                "[process_batch_upload] Completed %s upload_id=%s",
                csv_type,
                upload_id,
            )
        except Exception:
            logger.exception(
                "[process_batch_upload] Failed to process %s upload_id=%s",
                csv_type,
                upload_id,
            )
            # Update all uploads with error status
            await storage.update_csv_upload(
                orders_upload_id,
                {
                    "status": "failed",
                    "error_message": f"Batch processing failed at {csv_type}",
                    "updated_at": datetime.utcnow(),
                },
            )
            return

    logger.info(
        "[process_batch_upload] All CSVs processed. Checking data readiness for run_id=%s",
        run_id,
    )

    # Auto-trigger bundle generation pipeline
    async with AsyncSessionLocal() as session:
        try:
            await _ensure_data_ready(orders_upload_id, run_id, session)
        except HTTPException as exc:
            logger.warning(
                "[process_batch_upload] Pipeline aborted run_id=%s reason=%s",
                run_id,
                exc.detail,
            )
            await storage.update_csv_upload(
                orders_upload_id,
                {
                    "error_message": f"Pipeline skipped: {exc.detail}",
                    "updated_at": datetime.utcnow(),
                },
            )
            return
        except Exception:
            logger.exception(
                "[process_batch_upload] Failed readiness checks run_id=%s", run_id
            )
            await storage.update_csv_upload(
                orders_upload_id,
                {
                    "error_message": "Pipeline skipped due to unexpected readiness failure.",
                    "updated_at": datetime.utcnow(),
                },
            )
            return

    try:
        logger.info(
            "[process_batch_upload] Launching bundle generation for orders_upload_id=%s",
            orders_upload_id,
        )
        await generate_bundles_background(orders_upload_id)
        logger.info(
            "[process_batch_upload] Pipeline completed for run_id=%s",
            run_id,
        )
    except Exception:
        logger.exception(
            "[process_batch_upload] Pipeline execution failed run_id=%s", run_id
        )
