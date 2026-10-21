"""Strict JSON completions and content-free generation diagnostics."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from scilitrag.config import Settings
from scilitrag.rag.errors import LLMClientError

logger = logging.getLogger(__name__)


class JsonCompletionClient(Protocol):
    """The completion capability needed by schema validation and its test doubles."""

    def complete(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class StructuredResult:
    """Validation diagnostics without prompts, source passages or model output."""

    schema_name: str
    elapsed_ms: float
    valid: bool


def complete_json[T: BaseModel](
    client: JsonCompletionClient, messages: list[dict[str, str]], model: type[T]
) -> T:
    """Validate a model response against the caller's strict Pydantic schema."""
    start = time.perf_counter()
    valid = False
    try:
        schema = model.model_json_schema()
        raw = client.complete(messages, schema=schema)
        # Strict JSON validation prevents number/string and boolean coercion.
        result = model.model_validate_json(raw, strict=True)
        valid = True
        return result
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
        raise LLMClientError(
            "The model returned invalid structured output.", kind="structured"
        ) from None
    finally:
        diagnostic = StructuredResult(model.__name__, (time.perf_counter() - start) * 1000, valid)
        logger.info(
            "structured schema=%s valid=%s latency_ms=%.3f",
            diagnostic.schema_name,
            diagnostic.valid,
            diagnostic.elapsed_ms,
        )


def generation_metadata(settings: Settings) -> dict[str, str | int | float | None]:
    """Return public generation settings for reproducible model comparisons."""
    # Reusable comparison metadata: no endpoint, API credentials or prompt text.
    return {
        "model": settings.llm_model,
        "temperature": 0,
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_s": settings.request_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
