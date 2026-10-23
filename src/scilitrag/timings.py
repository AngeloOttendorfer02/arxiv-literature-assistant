"""Per-request timings that components report without a contract change."""

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    timings_ms: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


# Set by the API for the pipeline task; worker threads see the same object.
current_metrics: ContextVar[RequestMetrics | None] = ContextVar("request_metrics", default=None)


def record_timing(name: str, ms: float) -> None:
    """Add ms to the request log entry, summed if the step runs more than once.

    Does nothing outside a request, e.g. in scripts and tests.
    """
    metrics = current_metrics.get()
    if metrics is not None:
        metrics.timings_ms[name] = metrics.timings_ms.get(name, 0.0) + ms
