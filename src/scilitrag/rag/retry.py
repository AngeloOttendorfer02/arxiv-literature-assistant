import asyncio
import time
from dataclasses import dataclass

import httpx

RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class RetryBudget:
    max_retries: int
    backoff_s: float
    deadline: float

    @classmethod
    def start(cls, settings):
        return cls(
            settings.llm_max_retries,
            settings.llm_retry_backoff_s,
            time.monotonic() + settings.request_timeout_s,
        )

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def delay(self, attempt: int, error: Exception) -> float | None:
        if attempt >= self.max_retries:
            return None
        delay = min(self.backoff_s * (2**attempt), 2.0)
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code not in RETRYABLE_STATUS:
                return None
            value = error.response.headers.get("retry-after", "")
            try:
                delay = min(max(float(value), delay), 2.0) if value else delay
            except ValueError:
                pass
        elif not isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
            return None
        return delay if delay < self.remaining() else None

    def wait(self, delay: float) -> None:
        time.sleep(delay)

    async def wait_async(self, delay: float) -> None:
        await asyncio.sleep(delay)
