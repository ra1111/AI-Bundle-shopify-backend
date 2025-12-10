"""
Shopify data upload endpoints
Accepts CSV payloads from the Remix embedded app and feeds the existing pipeline.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal, BundleRecommendation, CsvUpload, get_db
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
    resolved_shop_id = resolve_shop_id(request.shop_id)

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


@router.get("/status/{upload_id}", response_model=UploadStatusResponse)
async def get_upload_status(upload_id: str, db: AsyncSession = Depends(get_db)):
    """Check processing status for a CsvUpload created via the Shopify endpoint."""

    logger.info(f"🔍 STATUS CHECK: upload_id={upload_id}")

    upload = await db.get(CsvUpload, upload_id)
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
        stmt = select(func.count(BundleRecommendation.id)).where(
            BundleRecommendation.csv_upload_id == upload_id
        )
        result = await db.execute(stmt)
        count = result.scalar() or 0
        bundle_count = count if count > 0 else None

    # Normalize status values for frontend simplicity
    # Map internal detailed statuses to simple states
    frontend_status = upload.status
    if upload.status in {"bundle_generation_completed"}:
        frontend_status = "completed"
    elif upload.status in {"bundle_generation_in_progress", "bundle_generation_queued", "bundle_generation_async"}:
        frontend_status = "processing"
    elif upload.status in {"bundle_generation_failed", "bundle_generation_timed_out", "bundle_generation_cancelled"}:
        frontend_status = "failed"

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

    normalized_shop_id = resolve_shop_id(shop_id)

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

    normalized_shop_id = resolve_shop_id(shop_id)
    recommendation = await db.get(BundleRecommendation, recommendation_id)
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    if resolve_shop_id(recommendation.shop_id) != normalized_shop_id:
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
