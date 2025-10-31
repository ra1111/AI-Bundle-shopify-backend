"""
Generation progress persistence helpers.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Union
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database import AsyncSessionLocal, CsvUpload, GenerationProgress
from settings import DEFAULT_SHOP_ID, resolve_shop_id

logger = logging.getLogger(__name__)

ProgressStatus = Literal["in_progress", "completed", "failed"]


async def _resolve_shop_domain(session, upload_id: str) -> str:
    """Fetch the associated shop domain for an upload."""
    try:
        result = await session.execute(
            select(CsvUpload.shop_id).where(CsvUpload.id == upload_id)
        )
        shop_id = result.scalar()
        if shop_id:
            return resolve_shop_id(shop_id)
    except Exception:
        logger.exception("Failed to resolve shop domain for upload %s", upload_id)
    return resolve_shop_id(DEFAULT_SHOP_ID)


async def update_generation_progress(
    upload_id: Union[UUID, str],
    *,
    step: str,
    progress: int,
    status: ProgressStatus,
    message: Optional[str] = None,
    bundle_count: Optional[int] = None,
    time_remaining: Optional[int] = None,
) -> None:
    """
    Upsert the latest generation progress snapshot for an upload.
    """
    upload_id_str = str(upload_id)
    safe_progress = max(0, min(100, int(progress)))
    metadata_payload = {
        key: value
        for key, value in {
            "bundle_count": bundle_count,
            "time_remaining": time_remaining,
        }.items()
        if value is not None
    }

    async with AsyncSessionLocal() as session:
        shop_domain = await _resolve_shop_domain(session, upload_id_str)

        stmt = pg_insert(GenerationProgress).values(
            upload_id=upload_id_str,
            shop_domain=shop_domain,
            step=step,
            progress=safe_progress,
            status=status,
            message=message,
            metadata=metadata_payload or None,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=[GenerationProgress.upload_id],
            set_={
                "shop_domain": shop_domain,
                "step": step,
                "progress": safe_progress,
                "status": status,
                "message": message,
                "metadata": metadata_payload or None,
                "updated_at": func.now(),
            },
        )

        await session.execute(stmt)
        await session.commit()

        logger.info(
            "Progress updated for upload %s step=%s progress=%s status=%s",
            upload_id_str,
            step,
            safe_progress,
            status,
        )
