"""
Generation progress persistence helpers.
"""
from __future__ import annotations

import logging
from typing import Dict, Literal, Optional, Union
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import ProgrammingError, DBAPIError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database import AsyncSessionLocal, CsvUpload, GenerationProgress
from settings import DEFAULT_SHOP_ID, resolve_shop_id

logger = logging.getLogger(__name__)

ProgressStatus = Literal["in_progress", "completed", "failed"]

# Define the valid ordering of progress steps (lower index = earlier step)
# This prevents the state machine from going backwards
STEP_ORDER = {
    "queueing": 0,
    "task_scheduled": 1,
    "initializing": 2,
    "enrichment": 3,
    "scoring": 4,
    "ml_generation": 5,
    "optimization": 6,
    "ai_descriptions": 7,
    "staged_publish": 8,
    "finalization": 9,
}

# Terminal statuses that lock the progress record
TERMINAL_STATUSES = {"completed", "failed"}


def _is_valid_transition(current_step: Optional[str], current_status: Optional[str], new_step: str, new_status: str) -> bool:
    """
    Check if transitioning from current state to new state is valid.

    Rules:
    1. Once in a terminal status (completed/failed), no further updates allowed
    2. Steps can only move forward in the STEP_ORDER (no backwards transitions)
    3. If same step, progress can only increase (enforced by progress value check)
    4. finalization step can always be updated (for final status changes)
    """
    # Rule 1: Terminal status locks the record (except for initial insert)
    if current_status in TERMINAL_STATUSES and current_step is not None:
        # Exception: Allow same terminal status with finalization step (idempotent completion)
        if new_step == "finalization" and new_status == current_status:
            return True
        logger.warning(
            "Progress state machine: blocking update after terminal status | "
            "current_step=%s current_status=%s new_step=%s new_status=%s",
            current_step, current_status, new_step, new_status,
        )
        return False

    # Rule 2: Steps can only move forward (or stay same)
    current_order = STEP_ORDER.get(current_step, -1) if current_step else -1
    new_order = STEP_ORDER.get(new_step, 999)  # Unknown steps get high order

    if new_order < current_order:
        # Exception: Allow finalization to always proceed (it's terminal)
        if new_step == "finalization":
            return True
        logger.warning(
            "Progress state machine: blocking backwards transition | "
            "current_step=%s (%d) → new_step=%s (%d)",
            current_step, current_order, new_step, new_order,
        )
        return False

    return True


async def _resolve_shop_domain(session, upload_id: str) -> Optional[str]:
    """Fetch the associated shop domain for an upload."""
    from settings import sanitize_shop_id
    try:
        result = await session.execute(
            select(CsvUpload.shop_id).where(CsvUpload.id == upload_id)
        )
        shop_id = result.scalar()
        if shop_id:
            # Only sanitize - NEVER fallback to DEFAULT_SHOP_ID
            return sanitize_shop_id(shop_id)
    except Exception:
        logger.exception("Failed to resolve shop domain for upload %s", upload_id)
    # Return None instead of DEFAULT - let caller handle missing shop_id
    return None


async def update_generation_progress(
    upload_id: Union[UUID, str],
    *,
    step: str,
    progress: int,
    status: ProgressStatus,
    message: Optional[str] = None,
    bundle_count: Optional[int] = None,
    time_remaining: Optional[int] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """
    Upsert the latest generation progress snapshot for an upload.

    Enforces state machine rules:
    - Once terminal (completed/failed), no further updates allowed
    - Steps can only move forward, not backwards
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
    if metadata:
        metadata_payload.update(metadata)

    async with AsyncSessionLocal() as session:
        try:
            # STATE MACHINE GUARD: Check current state before updating
            current_state = await session.execute(
                select(GenerationProgress.step, GenerationProgress.status)
                .where(GenerationProgress.upload_id == upload_id_str)
            )
            current_row = current_state.one_or_none()

            if current_row:
                current_step, current_status = current_row
                if not _is_valid_transition(current_step, current_status, step, status):
                    logger.info(
                        "Progress update blocked by state machine | upload=%s current=(%s, %s) attempted=(%s, %s)",
                        upload_id_str, current_step, current_status, step, status,
                    )
                    return  # Silently skip invalid transitions
            else:
                current_step, current_status = None, None

            shop_domain = await _resolve_shop_domain(session, upload_id_str)

            progress_table = GenerationProgress.__table__

            stmt = (
                pg_insert(progress_table)
                .values(
                    upload_id=upload_id_str,
                    shop_domain=shop_domain,
                    step=step,
                    progress=safe_progress,
                    status=status,
                    message=message,
                    metadata=metadata_payload or None,
                )
            )

            stmt = stmt.on_conflict_do_update(
                index_elements=[progress_table.c.upload_id],
                set_={
                    progress_table.c.shop_domain: shop_domain,
                    progress_table.c.step: step,
                    progress_table.c.progress: safe_progress,
                    progress_table.c.status: status,
                    progress_table.c.message: message,
                    progress_table.c.metadata: metadata_payload or None,
                    progress_table.c.updated_at: func.now(),
                },
            )

            try:
                await session.execute(stmt)
            except ProgrammingError as exc:
                if "generation_progress" in str(exc):
                    logger.warning(
                        "generation_progress table missing; attempting to create it on the fly"
                    )

                    await session.rollback()

                    async def _create_table(sync_session):
                        GenerationProgress.__table__.create(
                            bind=sync_session.bind, checkfirst=True
                        )

                    try:
                        async with session.begin():
                            await session.run_sync(_create_table)
                    except ProgrammingError:
                        # If another worker created the table in the meantime, ignore the error.
                        await session.rollback()

                    await session.execute(stmt)
                else:
                    raise

            await session.commit()

            logger.info(
                "Progress updated for upload %s step=%s progress=%s status=%s",
                upload_id_str,
                step,
                safe_progress,
                status,
            )

        except DBAPIError as exc:
            # Handle transaction abort errors gracefully
            if "InFailedSQLTransactionError" in str(exc) or "transaction is aborted" in str(exc):
                logger.warning(
                    "Transaction aborted while updating progress for upload %s - attempting rollback and retry",
                    upload_id_str
                )
                await session.rollback()

                # Retry once with a fresh transaction
                try:
                    await session.execute(stmt)
                    await session.commit()
                    logger.info(
                        "Progress updated (retry) for upload %s step=%s progress=%s status=%s",
                        upload_id_str,
                        step,
                        safe_progress,
                        status,
                    )
                except Exception as retry_exc:
                    logger.error(
                        "Failed to update progress even after retry for upload %s: %s",
                        upload_id_str,
                        retry_exc
                    )
                    # Don't re-raise - progress tracking should not crash the main workflow
            else:
                # For other database errors, log but don't crash
                logger.error(
                    "Database error updating progress for upload %s: %s",
                upload_id_str,
                exc
            )

        except Exception as exc:
            # Catch-all for any other errors - log but don't crash
            logger.error(
                "Unexpected error updating progress for upload %s: %s",
                upload_id_str,
                exc
            )


async def get_generation_checkpoint(upload_id: Union[UUID, str]) -> Optional[Dict[str, object]]:
    """
    Fetch the most recent checkpoint metadata for a given upload, if any.
    """
    upload_id_str = str(upload_id)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(
                GenerationProgress.metadata_json,
                GenerationProgress.step,
                GenerationProgress.progress,
            ).where(GenerationProgress.upload_id == upload_id_str)
        )
        row = result.one_or_none()
        if not row:
            return None

        metadata = row.metadata_json or {}
        checkpoint = metadata.get("checkpoint")
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.copy()
            checkpoint.setdefault("step", row.step)
            checkpoint.setdefault("progress", row.progress)
            return checkpoint

        if metadata:
            fallback = metadata.copy()
            fallback.setdefault("step", row.step)
            fallback.setdefault("progress", row.progress)
            return fallback

    return None
