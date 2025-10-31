"""
Automatic bundle generation trigger.

This helper watches for CSV ingestion completion events and, when all
required datasets for a run are marked completed, schedules bundle
generation in the background using the same pipeline exposed by the
bundle recommendations router.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

from services.storage import storage
from services.progress_tracker import update_generation_progress

logger = logging.getLogger(__name__)

# Required CSV types that must be completed before bundles can be generated.
REQUIRED_UPLOAD_TYPES: Set[str] = {"orders", "variants", "catalog_joined"}

# Status values that indicate an upload is ready (or already finished) for the purposes
# of kicking off bundle generation. Any other status means we should wait.
READY_STATUSES: Set[str] = {"completed", "bundle_generation_completed"}

# Track uploads we've already scheduled in this process to avoid duplicate tasks
# while the async job spins up.
_pending_runs: Set[str] = set()


async def maybe_trigger_bundle_generation(completed_upload_id: str) -> None:
    """
    Inspect the run containing the given upload and, if all required CSV types
    are completed, schedule bundle generation automatically.
    """
    if not completed_upload_id:
        return

    run_id = await storage.get_run_id_for_upload(completed_upload_id)
    if not run_id:
        # CSV not associated with a run; nothing to do.
        return

    uploads = await storage.get_run_uploads(run_id)
    if not uploads:
        return

    # Reduce to the most recent upload per required type.
    latest_by_type: Dict[str, Any] = {}
    for upload in sorted(
        uploads,
        key=lambda u: getattr(u, "created_at", None) or datetime.min,
        reverse=True,
    ):
        csv_type = getattr(upload, "csv_type", None)
        if csv_type in REQUIRED_UPLOAD_TYPES and csv_type not in latest_by_type:
            latest_by_type[csv_type] = upload

    missing_types = REQUIRED_UPLOAD_TYPES - latest_by_type.keys()
    if missing_types:
        logger.debug(
            "Auto-bundle: run %s missing required CSV types: %s",
            run_id,
            ", ".join(sorted(missing_types)),
        )
        return

    # Ensure all required uploads reached a ready status.
    for csv_type, upload in latest_by_type.items():
        if upload is None or getattr(upload, "status", None) not in READY_STATUSES:
            logger.debug(
                "Auto-bundle: run %s waiting for %s upload (status=%s)",
                run_id,
                csv_type,
                getattr(upload, "status", None),
            )
            return

    orders_upload = latest_by_type["orders"]
    orders_upload_id = getattr(orders_upload, "id", None)
    orders_status = getattr(orders_upload, "status", "")

    if not orders_upload_id:
        return

    if orders_upload_id in _pending_runs:
        logger.debug(
            "Auto-bundle: bundle generation already pending for orders upload %s",
            orders_upload_id,
        )
        return

    if orders_status in {"generating_bundles", "bundle_generation_completed", "bundle_generation_failed"}:
        logger.info(
            "Auto-bundle: skipping orders upload %s (status=%s)",
            orders_upload_id,
            orders_status,
        )
        return

    # Update status to reflect the queued bundle generation (best-effort).
    try:
        await storage.update_csv_upload(orders_upload_id, {"status": "bundle_generation_queued"})
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Auto-bundle: failed to mark orders upload %s as queued: %s",
            orders_upload_id,
            exc,
        )

    try:
        await update_generation_progress(
            orders_upload_id,
            step="queueing",
            progress=0,
            status="in_progress",
            message="Bundle generation queued; waiting for background worker.",
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Auto-bundle: failed to record queue progress for upload %s: %s",
            orders_upload_id,
            exc,
        )

    logger.info(
        "Auto-bundle: scheduling bundle generation for orders upload %s (run %s)",
        orders_upload_id,
        run_id,
    )
    _pending_runs.add(orders_upload_id)

    async def _run_generation(upload_id: str) -> None:
        try:
            from routers.bundle_recommendations import generate_bundles_background

            await generate_bundles_background(upload_id)
        except Exception:
            logger.exception(
                "Auto-bundle: bundle generation task failed for upload %s",
                upload_id,
            )
        finally:
            _pending_runs.discard(upload_id)

    asyncio.create_task(_run_generation(orders_upload_id))


__all__ = ["maybe_trigger_bundle_generation"]
