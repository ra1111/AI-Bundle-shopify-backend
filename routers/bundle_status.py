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
    pending_payload: Dict[str, Any] = {
        "upload_id": upload_id,
        "shop_id": None,
        "status": "processing",
        "total_rows": 0,
        "processed_rows": 0,
        "error_message": "Upload not found yet; still processing.",
        "bundle_count": None,
    }

    upload = await _load_upload(upload_id, db)

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
            logger.info(f"bundle-status: Resolved run_id {upload_id} -> orders upload {upload.id}")

    if not upload:
        progress_payload = await _load_progress(upload_id, db)
        if progress_payload:
            # Return a soft 202 with progress when the upload record is missing
            pending_payload.update(
                {
                    "status": progress_payload.get("status", "processing"),
                    "generation_progress": progress_payload,
                }
            )
            return pending_payload
        return pending_payload

    # Use the resolved upload.id for bundle count (not the original upload_id which might be run_id)
    bundle_count = None
    if upload.status in COMPLETED_STATES or upload.status == "processing":
        bundle_count = await _count_bundles(upload.id, db)

    status = upload.status
    if upload.status in {"bundle_generation_completed"}:
        status = "completed"
    elif upload.status in {"bundle_generation_in_progress", "bundle_generation_queued", "bundle_generation_async", "generating_bundles"}:
        status = "processing"
    elif upload.status in FAILED_STATES:
        status = "failed"
    elif upload.status == "completed" and bundle_count is None:
        # CSV processing completed but no bundles yet - check if bundle generation is pending
        # This handles the race condition where CSV finishes before auto-trigger kicks in
        progress_payload = await _load_progress(upload.id, db)
        if progress_payload:
            if progress_payload.get("status") == "in_progress":
                # Bundle generation is actually in progress - don't say "completed"
                status = "processing"
                logger.info(
                    f"bundle-status: STATUS OVERRIDE for {upload_id} - "
                    f"csv_status=completed but progress shows in_progress at step={progress_payload.get('step')}"
                )
            elif progress_payload.get("status") == "completed":
                # Bundle generation finished - keep as completed
                status = "completed"
            # If status is "failed", keep as completed (user can retry)
        else:
            # NO progress record yet - check if upload is recent (Quick Start scenario)
            now = datetime.now(timezone.utc)
            created_at = upload.created_at
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            # If upload was created within last 5 minutes, assume bundle generation is pending
            if created_at and (now - created_at).total_seconds() < 300:
                status = "processing"
                logger.info(
                    f"bundle-status: STATUS OVERRIDE for {upload_id} - "
                    f"csv_status=completed, no progress record, but upload is recent "
                    f"({(now - created_at).total_seconds():.0f}s old) - assuming bundle generation pending"
                )

    progress_payload = await _load_progress(upload.id, db)

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
