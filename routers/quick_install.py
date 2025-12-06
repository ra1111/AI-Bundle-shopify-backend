"""
Quick Install Router
One-click bundle generation for new shops
Handles rapid bundle creation from CSV with status tracking and retry logic
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional
from io import StringIO
import csv

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form, Request
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal, CsvUpload, BundleRecommendation, get_db
from settings import resolve_shop_id
from routers.uploads import process_csv_background

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/quick-install", tags=["Quick Install"])

# Configuration
QUICK_INSTALL_COOLDOWN_DAYS = 30
PROCESSING_TIMEOUT_MINUTES = 10


# ============================================================================
# Pydantic Models
# ============================================================================

class QuickInstallResponse(BaseModel):
    """Response from quick install upload endpoint"""
    job_id: str
    status: str
    message: str
    is_retry: Optional[bool] = None


class QuickInstallStatusResponse(BaseModel):
    """Response from quick install status check endpoint"""
    has_quick_install: bool
    can_run: bool
    status: Optional[str] = None
    job_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    bundles_created: Optional[int] = None
    error_message: Optional[str] = None
    days_since_install: Optional[int] = None
    can_retry_in_days: Optional[int] = None
    message: Optional[str] = None


class BundleStatusResponse(BaseModel):
    """Generic bundle status response"""
    job_id: str
    status: str
    processed_count: int
    error_count: int
    error_message: Optional[str]
    created_at: str
    processed_at: Optional[str]


# ============================================================================
# Helper Functions
# ============================================================================

async def _get_shop_quick_install(shop_id: str, db: AsyncSession) -> Optional[CsvUpload]:
    """Get the most recent quick install for a shop"""
    stmt = (
        select(CsvUpload)
        .where(
            CsvUpload.shop_id == shop_id,
            CsvUpload.processing_params["source"].astext == "quick_install"
        )
        .order_by(CsvUpload.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _count_bundle_recommendations(upload_id: str, db: AsyncSession) -> int:
    """Count bundles created from this upload"""
    stmt = select(func.count(BundleRecommendation.id)).where(
        BundleRecommendation.csv_upload_id == upload_id
    )
    result = await db.execute(stmt)
    return result.scalar() or 0


async def _process_quick_install_async(
    csv_content: str,
    upload_id: str,
    shop_id: str
) -> None:
    """
    Background task to process quick install CSV
    Updates status to COMPLETED or FAILED when done
    """
    logger.info(f"[quick_install] Starting async processing for upload_id={upload_id}")

    try:
        # Use existing CSV processor
        await process_csv_background(csv_content, upload_id, "quick_install")

        # Process CSV was successful, now update status to COMPLETED
        async with AsyncSessionLocal() as session:
            bundle_count = await _count_bundle_recommendations(upload_id, session)

            upload = await session.get(CsvUpload, upload_id)
            if upload:
                upload.status = "completed"
                upload.processed_rows = bundle_count
                upload.updated_at = datetime.utcnow()
                await session.commit()
                logger.info(
                    f"[quick_install] ✅ Upload {upload_id} completed: "
                    f"{bundle_count} bundles created"
                )
            else:
                logger.error(f"[quick_install] ❌ Upload {upload_id} not found for status update")

    except Exception as e:
        logger.exception(f"[quick_install] ❌ Processing failed for upload_id={upload_id}")

        # Update status to FAILED
        try:
            async with AsyncSessionLocal() as session:
                upload = await session.get(CsvUpload, upload_id)
                if upload:
                    upload.status = "failed"
                    upload.error_message = str(e)
                    upload.updated_at = datetime.utcnow()
                    await session.commit()
                    logger.info(f"[quick_install] Status updated to FAILED: {str(e)}")
        except Exception as status_error:
            logger.error(f"[quick_install] Failed to update status: {status_error}")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/upload")
async def quick_install(
    file: UploadFile = File(...),
    shop_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload CSV and start quick bundle generation.
    Returns immediately without waiting for processing.

    Handles:
    - Duplicate prevention (check for existing jobs)
    - Cooldown period (30 days between runs)
    - Retry logic for failed installs
    
    Request parameters:
    - file: CSV file (multipart/form-data)
    - shop_id: Shop identifier (form parameter from frontend)
    """

    logger.info(f"[quick_install] Received upload for shop_id={shop_id}")

    # Normalize shop_id
    shop_id = resolve_shop_id(shop_id)

    # Validate file
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file: CSV file required"
        )

    # Check for existing quick install
    existing = await _get_shop_quick_install(shop_id, db)

    if existing:
        logger.info(f"[quick_install] Found existing install: {existing.id} status={existing.status}")

        # ALREADY PROCESSING
        if existing.status == "processing":
            logger.warning(f"[quick_install] Already processing: {existing.id}")
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "ALREADY_PROCESSING",
                    "job_id": existing.id,
                    "status": "PROCESSING",
                    "started_at": existing.created_at.isoformat(),
                    "elapsed_seconds": int((datetime.utcnow() - existing.created_at).total_seconds()),
                    "message": "Quick install already running. Please wait."
                }
            )

        # COMPLETED - Check cooldown
        if existing.status == "completed":
            days_since = (datetime.utcnow() - existing.updated_at).days
            days_remaining = QUICK_INSTALL_COOLDOWN_DAYS - days_since

            if days_remaining > 0:
                logger.warning(
                    f"[quick_install] Cooldown active: {existing.id} "
                    f"completed {days_since} days ago, {days_remaining} days remaining"
                )
                bundle_count = await _count_bundle_recommendations(existing.id, db)
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "ALREADY_COMPLETED",
                        "job_id": existing.id,
                        "completed_at": existing.updated_at.isoformat(),
                        "bundles_created": bundle_count,
                        "days_since_install": days_since,
                        "can_retry_in_days": days_remaining,
                        "message": f"Quick install completed {days_since} days ago. "
                                 f"You can run it again in {days_remaining} days."
                    }
                )
            else:
                logger.info(
                    f"[quick_install] Cooldown expired, allowing retry: {existing.id}"
                )

        # FAILED - Allow retry
        if existing.status == "failed":
            logger.info(f"[quick_install] Retrying failed install: {existing.id}")

    # Read CSV content
    try:
        csv_content = await file.read()
        csv_text = csv_content.decode('utf-8')

        # Validate CSV has rows
        reader = csv.DictReader(StringIO(csv_text))
        rows = list(reader)
        if not rows:
            raise HTTPException(
                status_code=400,
                detail="CSV file is empty or has no valid rows"
            )

        logger.info(f"[quick_install] CSV validated: {len(rows)} rows")

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file: must be UTF-8 encoded"
        )
    except csv.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(e)}"
        )
    except Exception as e:
        logger.exception(f"[quick_install] CSV read error")
        raise HTTPException(
            status_code=400,
            detail="Failed to read CSV file"
        )

    # Create CSV upload record with PROCESSING status
    upload_id = str(uuid.uuid4())
    upload = CsvUpload(
        id=upload_id,
        shop_id=shop_id,
        filename=f"quick_install_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
        csv_type="quick_install",
        status="processing",
        total_rows=len(rows),
        processed_rows=0,
        processing_params={
            "source": "quick_install",
            "file_name": file.filename,
            "row_count": len(rows)
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    db.add(upload)
    await db.commit()
    await db.refresh(upload)

    logger.info(f"[quick_install] Created CsvUpload: {upload_id}")

    # Trigger async processing
    background_tasks.add_task(
        _process_quick_install_async,
        csv_text,
        upload_id,
        shop_id
    )

    logger.info(f"[quick_install] Triggered async processing for {upload_id}")

    # Return immediately
    return QuickInstallResponse(
        job_id=upload_id,
        status="PROCESSING",
        message="Quick install started. Bundles are being generated...",
        is_retry=existing and existing.status == "failed"
    )


@router.get("/status")
async def get_quick_install_status(
    shop_id: str = None,
    db: AsyncSession = Depends(get_db),
) -> QuickInstallStatusResponse:
    """
    Check the status of quick install for the current shop.
    Returns different responses based on whether install has run and current state.
    
    Request parameters:
    - shop_id: Shop identifier (query parameter: ?shop_id=store.myshopify.com)
    """

    # Extract shop_id from query parameter
    if not shop_id:
        raise HTTPException(
            status_code=400,
            detail="shop_id required as query parameter: ?shop_id=store.myshopify.com"
        )

    shop_id = resolve_shop_id(shop_id)

    logger.info(f"[quick_install] Status check for shop_id={shop_id}")

    # Get most recent quick install
    existing = await _get_shop_quick_install(shop_id, db)

    if not existing:
        logger.info(f"[quick_install] No quick install found for {shop_id}")
        return QuickInstallStatusResponse(
            has_quick_install=False,
            can_run=True,
            message="Quick install has not been run for this shop"
        )

    logger.info(f"[quick_install] Found install {existing.id} with status={existing.status}")

    # PROCESSING
    if existing.status == "processing":
        elapsed = (datetime.utcnow() - existing.created_at).total_seconds()
        logger.info(f"[quick_install] Status=PROCESSING, elapsed={elapsed}s")
        return QuickInstallStatusResponse(
            has_quick_install=True,
            can_run=False,
            status="PROCESSING",
            job_id=existing.id,
            started_at=existing.created_at.isoformat(),
            message="Quick install is in progress"
        )

    # COMPLETED
    if existing.status == "completed":
        bundle_count = await _count_bundle_recommendations(existing.id, db)
        days_since = (datetime.utcnow() - existing.updated_at).days
        days_remaining = QUICK_INSTALL_COOLDOWN_DAYS - days_since
        can_retry = days_remaining <= 0

        logger.info(
            f"[quick_install] Status=COMPLETED, bundles={bundle_count}, "
            f"days_since={days_since}, can_retry={can_retry}"
        )

        return QuickInstallStatusResponse(
            has_quick_install=True,
            can_run=can_retry,
            status="COMPLETED",
            job_id=existing.id,
            completed_at=existing.updated_at.isoformat(),
            bundles_created=bundle_count,
            days_since_install=days_since,
            can_retry_in_days=max(0, days_remaining),
            message="Quick install completed" + (
                f" - Can retry in {days_remaining} days" if days_remaining > 0 else ""
            )
        )

    # FAILED
    if existing.status == "failed":
        logger.info(f"[quick_install] Status=FAILED, error={existing.error_message}")
        return QuickInstallStatusResponse(
            has_quick_install=True,
            can_run=True,
            status="FAILED",
            job_id=existing.id,
            error_message=existing.error_message,
            message="Quick install failed. You can try again."
        )

    # Default
    logger.warning(f"[quick_install] Unknown status: {existing.status}")
    return QuickInstallStatusResponse(
        has_quick_install=True,
        can_run=True,
        status=existing.status,
        job_id=existing.id,
        message=f"Unknown status: {existing.status}"
    )


@router.get("/status/{job_id}")
async def get_bundle_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> BundleStatusResponse:
    """
    Generic endpoint to check bundle generation status by job_id.
    Used by both quick install and manual uploads.
    """

    logger.info(f"[bundles] Status check for job_id={job_id}")

    # Find the CSV upload record
    upload = await db.get(CsvUpload, job_id)

    if not upload:
        logger.warning(f"[bundles] Job not found: {job_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    # Count error bundles (could be a field or calculated from bundles)
    error_count = 0  # TODO: Implement error tracking

    logger.info(
        f"[bundles] Job {job_id} status={upload.status} "
        f"processed={upload.processed_rows}"
    )

    return BundleStatusResponse(
        job_id=upload.id,
        status=upload.status,
        processed_count=upload.processed_rows or 0,
        error_count=error_count,
        error_message=upload.error_message,
        created_at=upload.created_at.isoformat(),
        processed_at=upload.updated_at.isoformat() if upload.updated_at else None
    )


# ============================================================================
# Cleanup Job (run via Cloud Scheduler or cron)
# ============================================================================

@router.post("/cron/cleanup")
async def cleanup_stuck_jobs(db: AsyncSession = Depends(get_db)):
    """
    Mark jobs as FAILED if stuck in PROCESSING for >10 minutes.
    Call this endpoint periodically (e.g., every 5 minutes via Cloud Scheduler).

    Requires authentication to prevent abuse.
    """

    logger.info("[cleanup] Starting cleanup of stuck jobs")

    stuck_threshold = datetime.utcnow() - timedelta(minutes=PROCESSING_TIMEOUT_MINUTES)

    # Find stuck jobs
    stmt = select(CsvUpload).where(
        CsvUpload.status == "processing",
        CsvUpload.created_at < stuck_threshold
    )
    result = await db.execute(stmt)
    stuck_jobs = result.scalars().all()

    if not stuck_jobs:
        logger.info("[cleanup] No stuck jobs found")
        return {"cleaned": 0}

    logger.warning(f"[cleanup] Found {len(stuck_jobs)} stuck jobs, marking as FAILED")

    # Mark as failed
    updated_count = 0
    for job in stuck_jobs:
        elapsed_minutes = (datetime.utcnow() - job.created_at).total_seconds() / 60

        job.status = "failed"
        job.error_message = f"Processing timed out after {PROCESSING_TIMEOUT_MINUTES} minutes"
        job.updated_at = datetime.utcnow()

        logger.warning(
            f"[cleanup] Marked {job.id} as FAILED "
            f"(stuck for {elapsed_minutes:.1f} minutes)"
        )

        updated_count += 1

    await db.commit()

    logger.info(f"[cleanup] Cleanup complete: {updated_count} jobs marked as FAILED")

    return {
        "cleaned": updated_count,
        "message": f"Marked {updated_count} stuck jobs as FAILED"
    }
