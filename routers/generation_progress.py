"""
Generation progress polling endpoint.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import JSONResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import ProgrammingError, OperationalError

from database import GenerationProgress, CsvUpload, BundleRecommendation, get_db

router = APIRouter()

PROGRESS_TTL = timedelta(hours=24)
logger = logging.getLogger(__name__)


async def _ensure_progress_table(db: AsyncSession) -> None:
    """Best-effort creation of generation_progress when migrations haven't run."""
    try:
        await db.rollback()
        await db.run_sync(
            lambda sync_conn: GenerationProgress.__table__.create(
                bind=sync_conn, checkfirst=True
            )
        )
        await db.commit()
        logger.warning("generation_progress table created lazily at request time")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Failed to auto-create generation_progress table: %s", exc, exc_info=True
        )
        await db.rollback()


@router.get("/generation-progress/shop/{shop_domain}/active")
async def get_active_sync_for_shop(
    shop_domain: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get the most recent in-progress sync for a shop.
    Returns the upload_id if there's an active sync, or null if none.
    This allows the frontend to resume progress tracking after navigation.
    """
    # Look for in-progress syncs for this shop in the last 24 hours
    cutoff = datetime.now(timezone.utc) - PROGRESS_TTL

    stmt = (
        select(GenerationProgress)
        .where(GenerationProgress.shop_domain == shop_domain)
        .where(GenerationProgress.status == "in_progress")
        .where(GenerationProgress.updated_at > cutoff)
        .order_by(GenerationProgress.updated_at.desc())
        .limit(1)
    )

    try:
        result = await db.execute(stmt)
        record = result.scalar_one_or_none()
    except Exception as exc:
        logger.warning(f"Error looking up active sync for {shop_domain}: {exc}")
        return {"active_sync": None}

    if not record:
        return {"active_sync": None}

    # Return the active sync info
    metadata = record.metadata_json if isinstance(record.metadata_json, dict) else {}

    return {
        "active_sync": {
            "upload_id": record.upload_id,
            "step": record.step,
            "progress": record.progress,
            "status": record.status,
            "message": record.message,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            "run_id": metadata.get("run_id"),
        }
    }


@router.get("/generation-progress/{upload_id}")
async def get_generation_progress(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    stmt = select(GenerationProgress).where(GenerationProgress.upload_id == upload_id)
    pending_payload = {
        "upload_id": upload_id,
        "step": "queueing",
        "progress": 5,  # Show minimal progress so UI doesn't look stuck
        "status": "in_progress",
        "message": "Uploading data and preparing AI analysis...",
        "metadata": {},
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        # Poll quickly during initial queueing to catch when task starts
        "recommended_poll_interval_ms": 2000,
    }
    try:
        result = await db.execute(stmt)
    except ProgrammingError as exc:
        if "generation_progress" in str(exc):
            logger.error("generation_progress table missing while polling %s", upload_id)
            await _ensure_progress_table(db)
            result = await db.execute(stmt)
        else:
            logger.exception("ProgrammingError reading generation progress for %s", upload_id)
            return JSONResponse(status_code=202, content=pending_payload)
    except OperationalError:
        return JSONResponse(
            status_code=202,
            content={
                **pending_payload,
                "message": "Database unavailable; retrying soon.",
            },
        )
    except Exception as exc:
        logger.exception("Unexpected error reading generation progress for %s", upload_id)
        return JSONResponse(status_code=202, content=pending_payload)

    record = result.scalar_one_or_none()

    # If not found by upload_id, try looking up by run_id
    # This allows frontend to poll using the shared run_id across all 4 CSVs
    if not record:
        # Check if provided ID is a run_id, and find the orders upload for it
        run_id_stmt = (
            select(CsvUpload.id)
            .where(CsvUpload.run_id == upload_id)
            .where(CsvUpload.csv_type == "orders")
            .order_by(CsvUpload.created_at.desc())
            .limit(1)
        )
        try:
            run_result = await db.execute(run_id_stmt)
            orders_upload_id = run_result.scalar_one_or_none()
            if orders_upload_id:
                logger.info(f"Resolved run_id {upload_id} -> orders upload {orders_upload_id}")
                # Now lookup progress for the orders upload
                progress_stmt = select(GenerationProgress).where(
                    GenerationProgress.upload_id == orders_upload_id
                )
                progress_result = await db.execute(progress_stmt)
                record = progress_result.scalar_one_or_none()
        except Exception as exc:
            logger.warning(f"Failed to resolve run_id {upload_id}: {exc}")

    if not record:
        # No progress record yet - check if CSV upload exists and is recent
        # If so, bundle generation should start soon, keep showing "in_progress"
        try:
            csv_stmt = select(CsvUpload).where(CsvUpload.id == (orders_upload_id if 'orders_upload_id' in dir() and orders_upload_id else upload_id))
            csv_result = await db.execute(csv_stmt)
            csv_upload = csv_result.scalar_one_or_none()
            if csv_upload:
                # CSV exists but no progress - generation is pending
                logger.info(
                    f"generation-progress: No progress record for {upload_id}, "
                    f"but CSV exists with status={csv_upload.status} - returning in_progress"
                )
        except Exception:
            pass  # Ignore errors, just return pending_payload
        return JSONResponse(status_code=202, content=pending_payload)

    updated_at = record.updated_at
    if updated_at is None:
        updated_at = datetime.now(timezone.utc)
    elif updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    else:
        updated_at = updated_at.astimezone(timezone.utc)

    if datetime.now(timezone.utc) - updated_at > PROGRESS_TTL:
        raise HTTPException(status_code=410, detail="Generation progress expired")

    metadata_raw = record.metadata_json
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
    staged_payload = None
    if isinstance(metadata.get("staged_publish"), dict):
        staged_payload = metadata["staged_publish"]
    elif metadata.get("staged"):
        staged_payload = metadata

    # ADAPTIVE POLLING: Recommend faster polling during active steps, slower when waiting
    # This reduces perceived latency without hammering the server
    step = record.step or ""
    status = record.status or ""
    if status == "completed" or status == "failed":
        # Final state - no need to poll anymore
        recommended_poll_ms = 0
    elif step in ("task_scheduled", "queueing", "initializing"):
        # Task just scheduled - poll quickly to catch when it starts
        recommended_poll_ms = 2000
    elif step in ("ml_generation", "optimization", "enrichment"):
        # Active processing - medium polling
        recommended_poll_ms = 3000
    elif step in ("finalization", "staged_publish"):
        # Almost done - poll quickly
        recommended_poll_ms = 2000
    else:
        # Default - standard polling
        recommended_poll_ms = 5000

    # Extract bundle_count from metadata for top-level response (frontend expects this)
    bundle_count = metadata.get("bundle_count") if isinstance(metadata, dict) else None

    # If status is completed or near completion, also query actual bundle count from DB
    # This ensures accurate count even if metadata wasn't updated in time
    actual_db_count = 0
    if (status == "completed" or record.progress >= 80) and bundle_count is None:
        try:
            # The upload_id in progress table is the orders upload ID
            orders_upload_id = record.upload_id
            stmt = select(func.count(BundleRecommendation.id)).where(
                BundleRecommendation.csv_upload_id == orders_upload_id
            )
            result = await db.execute(stmt)
            actual_db_count = result.scalar() or 0
            if actual_db_count > 0:
                bundle_count = actual_db_count
                logger.info(
                    f"Progress endpoint: queried actual bundle_count={actual_db_count} for upload {orders_upload_id}"
                )
        except Exception as exc:
            logger.warning(f"Failed to query bundle count for {record.upload_id}: {exc}")

    # RACE CONDITION FIX: If status is "completed" but there are 0 bundles,
    # this might be a stale/incorrect state. Check if the record is recent
    # and return "in_progress" to prevent frontend from showing "More Data Needed".
    if status == "completed" and actual_db_count == 0 and bundle_count is None:
        # Check if the upload is recent (within last 5 minutes)
        now = datetime.now(timezone.utc)
        age_seconds = (now - updated_at).total_seconds()

        if age_seconds < 300:  # 5 minutes
            logger.warning(
                f"generation-progress: RACE CONDITION FIX - status=completed but 0 bundles, "
                f"record is recent ({age_seconds:.0f}s old) | "
                f"upload_id={record.upload_id} step={step} progress={record.progress} "
                f"→ Returning in_progress instead of completed"
            )
            # Override status to prevent premature "More Data Needed" message
            status = "in_progress"
        else:
            logger.warning(
                f"generation-progress: SUSPICIOUS STATE - status=completed but 0 bundles | "
                f"upload_id={record.upload_id} step={step} progress={record.progress} "
                f"updated_at={updated_at.isoformat()} age={age_seconds:.0f}s"
            )

    response: Dict[str, Any] = {
        "upload_id": record.upload_id,
        "shop_domain": record.shop_domain,
        "step": record.step,
        "progress": record.progress,
        "status": status,  # Use potentially modified status (race condition fix)
        "message": record.message,
        "metadata": metadata,
        "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
        # Hint for frontend to adjust polling interval dynamically
        "recommended_poll_interval_ms": recommended_poll_ms,
    }

    # Add bundle_count at top level for frontend compatibility
    if bundle_count is not None:
        response["bundle_count"] = bundle_count

    if staged_payload:
        response["staged"] = staged_payload.get("staged", True)
        response["staged_state"] = staged_payload
        if staged_payload.get("run_id"):
            response["run_id"] = staged_payload["run_id"]
    elif isinstance(metadata, dict) and metadata.get("run_id"):
        response["run_id"] = metadata["run_id"]

    return response


@router.post("/force-reset/{upload_id}")
async def force_reset_sync(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Force-reset a stuck sync by marking its progress as failed.
    This allows starting a new sync when one is stuck.
    """
    from sqlalchemy import update
    from services.storage import storage

    logger.info(f"Force-reset requested for upload {upload_id}")

    # Update GenerationProgress to failed
    try:
        stmt = (
            update(GenerationProgress)
            .where(GenerationProgress.upload_id == upload_id)
            .values(
                status="failed",
                message="Force-reset by user to allow new sync",
                updated_at=datetime.now(timezone.utc),
            )
        )
        await db.execute(stmt)
        await db.commit()
        logger.info(f"GenerationProgress reset for {upload_id}")
    except Exception as exc:
        logger.warning(f"Failed to reset GenerationProgress for {upload_id}: {exc}")

    # Also reset the CSV upload status
    try:
        await storage.update_csv_upload(upload_id, {"status": "failed"})
        logger.info(f"CsvUpload status reset for {upload_id}")
    except Exception as exc:
        logger.warning(f"Failed to reset CsvUpload for {upload_id}: {exc}")

    return {
        "success": True,
        "message": f"Sync {upload_id} has been force-reset. You can now start a new sync.",
        "upload_id": upload_id,
    }
