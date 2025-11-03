"""
Notification stubs for merchant-facing updates.
Currently logs events; replace with real integrations (email/Slack/Webhooks) when available.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def notify_partial_ready(csv_upload_id: str, bundle_count: int) -> None:
    logger.info(
        "[NOTIFY] Partial bundles ready for upload %s | count=%d",
        csv_upload_id,
        bundle_count,
    )


async def notify_bundle_ready(csv_upload_id: str, bundle_count: int, resume_run: bool = False) -> None:
    logger.info(
        "[NOTIFY] Bundle generation complete for %s | count=%d resume_run=%s",
        csv_upload_id,
        bundle_count,
        resume_run,
    )
