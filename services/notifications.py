"""
Notification stubs for merchant-facing updates.
Currently logs events; replace with real integrations (email/Slack/Webhooks) when available.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


async def notify_partial_ready(
    csv_upload_id: str,
    bundle_count_or_metrics: Optional[Any] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Notify merchant that partial/quick-start bundles are ready.

    Args:
        csv_upload_id: The upload ID
        bundle_count_or_metrics: Either bundle count (int) or metrics dict
        details: Optional additional details about the notification
    """
    # Handle both old (metrics dict) and new (bundle count int) call patterns
    if isinstance(bundle_count_or_metrics, dict):
        metrics = bundle_count_or_metrics
        bundle_count = metrics.get('total_recommendations', 0)
    elif isinstance(bundle_count_or_metrics, int):
        metrics = None
        bundle_count = bundle_count_or_metrics
    else:
        metrics = None
        bundle_count = 0

    payload = {
        "upload_id": csv_upload_id,
        "bundle_count": bundle_count,
        "quick_start_mode": metrics.get('quick_start_mode', False) if metrics else False,
    }
    if metrics:
        payload["metrics"] = metrics
    if details:
        payload["details"] = details
    logger.info("[NOTIFY] Partial bundles ready | payload=%s", payload)


async def notify_bundle_ready(
    csv_upload_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    resume_run: bool = False,
) -> None:
    """Notify merchant that full bundle generation is complete.

    Args:
        csv_upload_id: The upload ID
        metrics: Generation metrics containing bundle_count and other details
        resume_run: Whether this was a resumed generation
    """
    bundle_count = metrics.get('total_recommendations', 0) if metrics else 0
    payload = {
        "upload_id": csv_upload_id,
        "bundle_count": bundle_count,
        "resume_run": resume_run,
    }
    if metrics:
        payload["metrics"] = metrics
    logger.info("[NOTIFY] Bundle generation complete | payload=%s", payload)
