import json
import logging
import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from scilitrag.contracts import SENTENCE_REMOVED, DoneEvent, ErrorEvent, Model, StatusEvent

GUARD_STAGES = ("prompt_guard", "topic_gate", "pii")
APP_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"

request_logger = logging.getLogger("scilitrag.requests")
# Set per request by the API middleware; copied into worker threads and tasks automatically.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_app_log() -> None:
    """Application log lines (all scilitrag.* loggers) carry the current request id."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, _AppLogHandler)]:
        root.removeHandler(handler)
    handler = _AppLogHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter(APP_LOG_FORMAT))
    root.addHandler(handler)
    logging.getLogger("scilitrag").setLevel(logging.INFO)


class _AppLogHandler(logging.StreamHandler):
    """Marks the handler added by setup_app_log, so a second call replaces it."""


def setup_request_log(log_dir: str | None) -> None:
    """One JSON line per request on stdout and, if possible, in log_dir/requests.jsonl.

    Replaces earlier handlers, so the latest app configuration wins.
    """
    for handler in list(request_logger.handlers):
        request_logger.removeHandler(handler)
        handler.close()
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(Path(log_dir) / "requests.jsonl", encoding="utf-8"))
        except OSError as exc:
            logging.getLogger(__name__).warning("request log file disabled: %s", exc)
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        request_logger.addHandler(handler)


@dataclass
class RequestLog:
    """Telemetry of one request, built from its events. Never holds the query text."""

    query_chars: int
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    scenario: str | None = None
    error_code: str | None = None
    guardrail_events: list[dict] = field(default_factory=list)
    done: DoneEvent | None = None

    def observe(self, event: Model) -> None:
        match event:
            case StatusEvent(stage="plan", result=result):
                self.scenario = result
            case StatusEvent(stage=stage, result=result) if (
                stage in GUARD_STAGES or result.startswith(SENTENCE_REMOVED)
            ):
                self.guardrail_events.append(event.model_dump())
            case DoneEvent(refused=refused):
                self.done = event
                self.status = "refused" if refused else "ok"
            case ErrorEvent(code=code):
                self.error_code = code
                self.status = "timeout" if code == "timeout" else "error"

    def record(self) -> dict:
        done = self.done
        return {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": self.request_id,
            "status": self.status,
            "error_code": self.error_code,
            "query_chars": self.query_chars,
            "scenario": self.scenario,
            "ttft_engine_ms": done.ttft_engine_ms if done else None,
            "ttft_user_ms": done.ttft_user_ms if done else None,
            "total_ms": done.total_ms if done else None,
            "stage_latency_ms": done.stage_latency_ms if done else {},
            "tokens": done.tokens if done else None,
            "tokens_per_s": done.tokens_per_s if done else None,
            "sentences_checked": done.sentences_checked if done else None,
            "sentences_removed": done.sentences_removed if done else None,
            "guardrail_events": self.guardrail_events,
        }

    def write(self) -> None:
        request_logger.info(json.dumps(self.record()))
