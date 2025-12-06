"""
Generation progress polling endpoint.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import ProgrammingError, OperationalError

from database import GenerationProgress, get_db

router = APIRouter()

PROGRESS_TTL = timedelta(hours=24)
logger = logging.getLogger(__name__)


@router.get("/generation-progress/{upload_id}")
async def get_generation_progress(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    stmt = select(GenerationProgress).where(GenerationProgress.upload_id == upload_id)
    try:
        result = await db.execute(stmt)
    except ProgrammingError as exc:
        if "generation_progress" in str(exc):
            logger.error("generation_progress table missing while polling %s", upload_id)
            raise HTTPException(
                status_code=503,
                detail="Generation progress storage not initialized; run migrations.",
            )
        raise
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        logger.exception("Unexpected error reading generation progress for %s", upload_id)
        raise HTTPException(status_code=500, detail="Failed to read generation progress") from exc

    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Generation progress not found")

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

    response: Dict[str, Any] = {
        "upload_id": record.upload_id,
        "shop_domain": record.shop_domain,
        "step": record.step,
        "progress": record.progress,
        "status": record.status,
        "message": record.message,
        "metadata": metadata,
        "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
    }
    if staged_payload:
        response["staged"] = staged_payload.get("staged", True)
        response["staged_state"] = staged_payload
        if staged_payload.get("run_id"):
            response["run_id"] = staged_payload["run_id"]
    elif isinstance(metadata, dict) and metadata.get("run_id"):
        response["run_id"] = metadata["run_id"]

    return response
