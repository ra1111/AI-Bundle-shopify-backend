"""Utility helpers for soft deadlines inside long-running bundle tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Deadline:
    """Monotonic deadline tracker.

    Use ``Deadline(seconds=270)`` to create a guard aligned with Cloud Run's
    300 second hard timeout; check ``deadline.expired`` (or ``deadline.remaining``)
    inside long loops to exit gracefully before the watchdog fires.
    """

    seconds: float

    def __post_init__(self) -> None:
        self._deadline = time.monotonic() + max(0.0, float(self.seconds))

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self._deadline

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())
