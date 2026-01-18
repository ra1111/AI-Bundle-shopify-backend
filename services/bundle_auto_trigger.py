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

from services.storage import storage, update_csv_upload_status
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
        logger.info(
            "Auto-bundle: upload %s has no run_id; skipping trigger",
            completed_upload_id,
        )
        return

    logger.info(
        "Auto-bundle: evaluating upload %s (run %s) for auto trigger",
        completed_upload_id,
        run_id,
    )

    try:
        uploads = await storage.get_run_uploads(run_id)
    except Exception as exc:
        logger.exception(
            "Auto-bundle: failed to fetch uploads for run %s from upload %s: %s",
            run_id,
            completed_upload_id,
            exc,
        )
        return

    if not uploads:
        logger.info(
            "Auto-bundle: no uploads found for run %s (from %s)",
            run_id,
            completed_upload_id,
        )
        return

    logger.info(
        "Auto-bundle: run %s has %d total uploads",
        run_id,
        len(uploads),
    )

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

    logger.info(
        "Auto-bundle: latest uploads snapshot for run %s -> %s",
        run_id,
        {
            csv_type: {
                "id": getattr(upload, "id", None),
                "status": getattr(upload, "status", None),
                "created_at": getattr(upload, "created_at", None),
            }
            for csv_type, upload in latest_by_type.items()
        },
    )

    missing_types = REQUIRED_UPLOAD_TYPES - latest_by_type.keys()
    if missing_types:
        logger.info(
            "Auto-bundle: run %s waiting for missing CSV types: %s",
            run_id,
            ", ".join(sorted(missing_types)),
        )
        return

    # Ensure all required uploads reached a ready status.
    for csv_type, upload in latest_by_type.items():
        if upload is None or getattr(upload, "status", None) not in READY_STATUSES:
            logger.info(
            "Auto-bundle: run %s waiting for %s upload status to be ready (current=%s)",
            run_id,
            csv_type,
            getattr(upload, "status", None),
        )
            if upload is not None:
                logger.info(
                    "Auto-bundle: upload %s details -> processed_rows=%s total_rows=%s error=%s",
                    getattr(upload, "id", None),
                    getattr(upload, "processed_rows", None),
                    getattr(upload, "total_rows", None),
                    getattr(upload, "error_message", None),
                )
            return

    # All required CSVs are ready!
    logger.info(
        "Auto-bundle: ✓ ALL REQUIRED CSV TYPES READY for run %s - proceeding to bundle generation",
        run_id,
    )

    orders_upload = latest_by_type["orders"]
    orders_upload_id = getattr(orders_upload, "id", None)
    orders_status = getattr(orders_upload, "status", "")

    if not orders_upload_id:
        return

    if orders_upload_id in _pending_runs:
        logger.info(
            "Auto-bundle: generation already pending for orders upload %s",
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

    # Ensure ingestion for variants/catalog is complete before queueing Phase 1.
    try:
        coverage = await storage.summarize_upload_coverage(orders_upload_id, run_id)
        order_lines = coverage.get("order_lines", 0)
        variant_count = coverage.get("variants", 0)
        catalog_count = coverage.get("catalog", 0)

        # Invariants: all required datasets must exist
        assert order_lines > 0, f"Invariant failed: order_lines missing for upload {orders_upload_id}"
        assert variant_count > 0, f"Invariant failed: variants missing for upload {orders_upload_id}"
        assert catalog_count > 0, f"Invariant failed: catalog missing for upload {orders_upload_id}"

        if variant_count == 0 or catalog_count == 0:
            logger.warning(
                "Auto-bundle: NOT scheduling bundle generation for %s (variants=%d catalog=%d)",
                orders_upload_id,
                variant_count,
                catalog_count,
            )
            await update_csv_upload_status(
                csv_upload_id=orders_upload_id,
                status="ingestion_incomplete",
                error_message="Variants/catalog not yet ingested; bundle generation not scheduled.",
                extra_metrics=coverage,
            )
            return

        if coverage.get("catalog_coverage_ratio", 1.0) < 1.0:
            logger.warning(
                "Auto-bundle: partial catalog coverage for upload %s (ratio=%.3f missing=%d)",
                orders_upload_id,
                coverage.get("catalog_coverage_ratio", 0.0),
                coverage.get("missing_catalog_variants", 0),
            )
    except AssertionError as exc:
        logger.warning("Auto-bundle: coverage invariant failed for upload %s: %s", orders_upload_id, exc)
        await update_csv_upload_status(
            csv_upload_id=orders_upload_id,
            status="ingestion_incomplete",
            error_message=str(exc),
            extra_metrics=coverage if "coverage" in locals() else None,
        )
        return
    except Exception as exc:
        logger.warning(
            "Auto-bundle: coverage check failed for upload %s (run %s): %s",
            orders_upload_id,
            run_id,
            exc,
        )
        return

    # Update status to reflect the queued bundle generation (best-effort).
    try:
        await storage.update_csv_upload(orders_upload_id, {"status": "bundle_generation_queued"})
        logger.info(
            "Auto-bundle: orders upload %s status -> bundle_generation_queued",
            orders_upload_id,
        )
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
        "Auto-bundle: scheduling bundle generation for orders upload %s (run %s) -> coroutine queued",
        orders_upload_id,
        run_id,
    )
    _pending_runs.add(orders_upload_id)

    async def _run_generation(upload_id: str) -> None:
        try:
            from routers.bundle_recommendations import generate_bundles_background

            logger.info(
                "Auto-bundle: background coroutine starting for upload %s",
                upload_id,
            )
            await generate_bundles_background(upload_id)
        except Exception:
            logger.exception(
                "Auto-bundle: bundle generation task failed for upload %s",
                upload_id,
            )
        finally:
            _pending_runs.discard(upload_id)

    asyncio.create_task(_run_generation(orders_upload_id))

    # IMMEDIATE PROGRESS UPDATE: Give user instant feedback that task is scheduled
    # This reduces perceived latency even if Cloud Run cold start delays actual execution
    try:
        await update_generation_progress(
            orders_upload_id,
            step="task_scheduled",
            progress=10,
            status="in_progress",
            message="Bundle generation scheduled, starting AI analysis...",
        )
        logger.info(
            "Auto-bundle: ✓ Immediate progress update sent for %s",
            orders_upload_id,
        )
    except Exception as progress_exc:
        # Don't fail the trigger if progress update fails
        logger.warning(
            "Auto-bundle: Failed to send immediate progress update for %s: %s",
            orders_upload_id,
            progress_exc,
        )

    logger.info(
        "Auto-bundle: ✓ BACKGROUND TASK CREATED for orders upload %s (run %s) - bundle generation will start shortly",
        orders_upload_id,
        run_id,
    )


__all__ = ["maybe_trigger_bundle_generation"]
