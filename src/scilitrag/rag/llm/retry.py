"""Retry only transient model failures within a single monotonic deadline."""

import asyncio
import time
from dataclasses import dataclass

import httpx

from scilitrag.config import Settings

RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
MAX_BACKOFF_S = 2.0


@dataclass(frozen=True)
class RetryBudget:
    """Bound the combined duration of HTTP attempts and their backoff delays."""

    max_retries: int
    backoff_s: float
    deadline: float

    @classmethod
    def start(cls, settings: Settings) -> "RetryBudget":
        """Create one deadline for the complete logical model request."""
        return cls(
            settings.llm_max_retries,
            settings.llm_retry_backoff_s,
            time.monotonic() + settings.request_timeout_s,
        )

    def remaining(self) -> float:
        """Return the remaining request budget, clamped at zero."""
        return max(0.0, self.deadline - time.monotonic())

    def delay(self, attempt: int, error: Exception) -> float | None:
        """Return a permitted backoff for a zero-based attempt, or stop retrying."""
        if attempt >= self.max_retries:
            return None
        delay = min(self.backoff_s * (2**attempt), MAX_BACKOFF_S)
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code not in RETRYABLE_STATUS:
                return None
            value = error.response.headers.get("retry-after", "")
            try:
                delay = min(max(float(value), delay), MAX_BACKOFF_S) if value else delay
            except ValueError:
                # Unsupported Retry-After values keep the calculated backoff.
                pass
        elif not isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
            return None
        # A retry is useful only if time remains after its backoff.
        return delay if delay < self.remaining() else None

    def wait(self, delay: float) -> None:
        time.sleep(delay)

    async def wait_async(self, delay: float) -> None:
        await asyncio.sleep(delay)
