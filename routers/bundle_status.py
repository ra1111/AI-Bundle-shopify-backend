"""
Bundle status endpoint (alias for upload status with optional generation progress).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import ProgrammingError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from database import BundleRecommendation, CsvUpload, GenerationProgress, get_db

router = APIRouter(prefix="/api", tags=["bundle-status"])
logger = logging.getLogger(__name__)

# Keep terminal state mapping consistent with /api/shopify/status/{uploadId}
COMPLETED_STATES = {
    "completed",
    "bundle_generation_completed",
    "bundle_generation_in_progress",
    "bundle_generation_queued",
    "bundle_generation_async",
}
FAILED_STATES = {
    "bundle_generation_failed",
    "bundle_generation_timed_out",
    "bundle_generation_cancelled",
}


async def _load_upload(
    upload_id: str, db: AsyncSession
) -> Optional[CsvUpload]:
    try:
        return await db.get(CsvUpload, upload_id)
    except ProgrammingError as exc:
        if "csv_uploads" in str(exc):
            logger.error("csv_uploads table missing while checking bundle status %s", upload_id)
            raise HTTPException(
                status_code=503,
                detail="Upload tracking table missing; run migrations.",
            )
        raise
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error loading upload %s", upload_id)
        raise HTTPException(status_code=500, detail="Failed to load bundle status") from exc


async def _count_bundles(upload_id: str, db: AsyncSession) -> Optional[int]:
    stmt = select(func.count(BundleRecommendation.id)).where(
        BundleRecommendation.csv_upload_id == upload_id
    )
    try:
        result = await db.execute(stmt)
        count = result.scalar() or 0
        return count if count > 0 else None
    except ProgrammingError as exc:
        # If the table is missing, fall back to None instead of raising 500
        if "bundle_recommendations" in str(exc):
            logger.error("bundle_recommendations table missing while counting bundles for %s", upload_id)
            return None
        raise
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable")


async def _load_progress(upload_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    stmt = select(GenerationProgress).where(GenerationProgress.upload_id == upload_id)
    try:
        result = await db.execute(stmt)
    except ProgrammingError as exc:
        if "generation_progress" in str(exc):
            # Treat missing table as "no progress yet" to avoid a 500
            logger.error("generation_progress table missing while checking bundle status %s", upload_id)
            return None
        raise
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable")

    record = result.scalar_one_or_none()
    if not record:
        return None

    updated_at = record.updated_at or datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    else:
        updated_at = updated_at.astimezone(timezone.utc)

    metadata_raw = record.metadata_json
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    payload: Dict[str, Any] = {
        "step": record.step,
        "progress": record.progress,
        "status": record.status,
        "message": record.message,
        "metadata": metadata,
        "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
    }

    if metadata.get("run_id"):
        payload["run_id"] = metadata["run_id"]
    if metadata.get("staged_publish"):
        payload["staged_state"] = metadata["staged_publish"]

    return payload


@router.get("/bundle-status/{upload_id}")
async def get_bundle_status(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    upload = await _load_upload(upload_id, db)
    if not upload:
        progress_payload = await _load_progress(upload_id, db)
        if progress_payload:
            # Return a soft 202 with progress when the upload record is missing
            return {
                "upload_id": upload_id,
                "shop_id": None,
                "status": progress_payload.get("status", "processing"),
                "total_rows": 0,
                "processed_rows": 0,
                "error_message": "Upload record not found; reporting progress only.",
                "bundle_count": None,
                "generation_progress": progress_payload,
            }
        raise HTTPException(status_code=404, detail="Upload not found")

    bundle_count = None
    if upload.status in COMPLETED_STATES or upload.status == "processing":
        bundle_count = await _count_bundles(upload_id, db)

    status = upload.status
    if upload.status in {"bundle_generation_completed"}:
        status = "completed"
    elif upload.status in {"bundle_generation_in_progress", "bundle_generation_queued", "bundle_generation_async"}:
        status = "processing"
    elif upload.status in FAILED_STATES:
        status = "failed"

    progress_payload = await _load_progress(upload_id, db)

    response: Dict[str, Any] = {
        "upload_id": upload.id,
        "shop_id": upload.shop_id,
        "status": status,
        "total_rows": upload.total_rows,
        "processed_rows": upload.processed_rows,
        "error_message": upload.error_message,
        "bundle_count": bundle_count,
    }

    if progress_payload:
        response["generation_progress"] = progress_payload

    return response
