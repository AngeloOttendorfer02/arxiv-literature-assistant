import json
import logging
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from scilitrag.rag.errors import LLMClientError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StructuredResult:
    schema_name: str
    elapsed_ms: float
    valid: bool


def complete_json[T: BaseModel](client, messages, model: type[T]) -> T:
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


def generation_metadata(settings) -> dict:
    # Reusable comparison metadata: no endpoint, API credentials or prompt text.
    return {
        "model": settings.llm_model,
        "temperature": 0,
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_s": settings.request_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
