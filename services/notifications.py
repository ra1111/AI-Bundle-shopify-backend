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
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Notify merchant that partial/quick-start bundles are ready.

    Args:
        csv_upload_id: The upload ID
        metrics: Generation metrics containing bundle_count and other details
    """
    bundle_count = metrics.get('total_recommendations', 0) if metrics else 0
    payload = {
        "upload_id": csv_upload_id,
        "bundle_count": bundle_count,
        "quick_start_mode": metrics.get('quick_start_mode', False) if metrics else False,
    }
    if metrics:
        payload["metrics"] = metrics
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
