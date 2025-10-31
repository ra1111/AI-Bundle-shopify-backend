"""
Generation progress polling endpoint.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import GenerationProgress, get_db

router = APIRouter()

PROGRESS_TTL = timedelta(hours=24)


@router.get("/generation-progress/{upload_id}")
async def get_generation_progress(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    stmt = select(GenerationProgress).where(GenerationProgress.upload_id == upload_id)
    result = await db.execute(stmt)
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

    metadata = record.metadata_json or {}

    return {
        "upload_id": record.upload_id,
        "shop_domain": record.shop_domain,
        "step": record.step,
        "progress": record.progress,
        "status": record.status,
        "message": record.message,
        "metadata": metadata,
        "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
    }
