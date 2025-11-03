"""
Lightweight async job scheduler for deferred bundle pipeline stages.
Provides bounded concurrency for background resume jobs.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional, Set


class PipelineScheduler:
    """Simple semaphore-backed task scheduler for async bundle work."""

    def __init__(self, max_concurrency: Optional[int] = None) -> None:
        self._logger = logging.getLogger(__name__)
        self._concurrency = max_concurrency or int(
            os.getenv("ASYNC_PIPELINE_CONCURRENCY", "2")
        )
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._tasks: Set[asyncio.Task] = set()

    def schedule(self, coro_factory: Callable[[], Awaitable[None]]) -> asyncio.Task:
        """Schedule coroutine factory to run under semaphore."""

        async def _runner() -> None:
            async with self._semaphore:
                try:
                    self._logger.info(
                        "Executing deferred pipeline job | pending=%d",
                        len(self._tasks),
                    )
                    await coro_factory()
                except Exception:  # pragma: no cover - defensive logging
                    self._logger.exception("Deferred pipeline job failed")

        task = asyncio.create_task(_runner())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._logger.info(
            "Deferred pipeline job scheduled | active=%d capacity=%d",
            len(self._tasks),
            self._concurrency,
        )
        return task


pipeline_scheduler = PipelineScheduler()
