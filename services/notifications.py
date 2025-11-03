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
    bundle_count: int,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "upload_id": csv_upload_id,
        "bundle_count": bundle_count,
    }
    if details:
        payload["details"] = details
    logger.info("[NOTIFY] Partial bundles ready | payload=%s", payload)


async def notify_bundle_ready(
    csv_upload_id: str,
    bundle_count: int,
    resume_run: bool = False,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "upload_id": csv_upload_id,
        "bundle_count": bundle_count,
        "resume_run": resume_run,
    }
    if details:
        payload["details"] = details
    logger.info("[NOTIFY] Bundle generation complete | payload=%s", payload)
