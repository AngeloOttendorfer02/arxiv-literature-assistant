import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv

DEFAULT_EMBED_MODEL = "ibm-granite/granite-embedding-english-r2"
DEFAULT_NLI_MODEL = "dleemiller/ModernCE-large-nli"
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_LOG_DIR = "logs"
REQUIRED_WITHOUT_MOCKS = ("DATABASE_URL", "LLM_BASE_URL", "LLM_MODEL")


class ConfigError(RuntimeError):
    """A missing or invalid runtime setting that should stop component startup."""


@dataclass(frozen=True)
class Settings:
    """One immutable configuration snapshot shared by the selected components."""

    database_url: str | None
    llm_base_url: str | None
    llm_model: str | None
    embed_model: str
    use_mocks: bool
    real_components: frozenset[str]
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    log_dir: str | None = DEFAULT_LOG_DIR
    llm_max_retries: int = 1
    llm_retry_backoff_s: float = 0.2
    llm_max_output_tokens: int = 1024
    # Sent as reasoning_effort when set; "none" turns thinking off on thinking models.
    llm_reasoning_effort: str | None = None
    context_char_budget: int = 16000
    evidence_item_char_limit: int = 2400
    max_sentence_buffer_chars: int = 32768
    prompt_guard_threshold: float = 0.5
    topic_gate_margin: float = 0.0
    data_dir: str | None = None
    evidence_guard_threshold: float = 0.5
    nli_model: str = DEFAULT_NLI_MODEL
    nli_entailment_threshold: float = 0.5
    nli_contradiction_threshold: float = 0.5
    # Bearer token for a protected endpoint such as the Modal proxy; kept out of repr and logs.
    llm_api_key: str | None = field(default=None, repr=False)


def _parse_float(name: str, value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a positive finite number, got {value!r}")
    return number


def _parse_unit_float(name: str, value: str) -> float:
    number = _parse_float(name, value)
    if number > 1:
        raise ConfigError(f"{name} must be in (0, 1], got {value!r}")
    return number


def _parse_margin(name: str, value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if not -1 <= number <= 1:
        raise ConfigError(f"{name} must be in [-1, 1], got {value!r}")
    return number


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no", ""):
        return False
    raise ConfigError(f"{name} must be 0 or 1, got {value!r}")


def _parse_int(name: str, value: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if not low <= number <= high:
        raise ConfigError(f"{name} must be between {low} and {high}")
    return number


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load and validate settings, using explicit mappings without reading .env."""
    if env is None:
        # Existing environment variables win over .env.
        load_dotenv(find_dotenv(usecwd=True))
        env = os.environ

    use_mocks = _parse_bool("USE_MOCKS", env.get("USE_MOCKS", "0"))
    if not use_mocks:
        missing = [name for name in REQUIRED_WITHOUT_MOCKS if not env.get(name)]
        if missing:
            raise ConfigError(f"Missing with USE_MOCKS=0: {', '.join(missing)}")

    real = env.get("REAL_COMPONENTS", "")
    return Settings(
        database_url=env.get("DATABASE_URL") or None,
        llm_base_url=env.get("LLM_BASE_URL") or None,
        llm_model=env.get("LLM_MODEL") or None,
        llm_api_key=env.get("LLM_API_KEY") or None,
        embed_model=env.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL,
        use_mocks=use_mocks,
        real_components=frozenset(name.strip() for name in real.split(",") if name.strip()),
        request_timeout_s=_parse_float(
            "REQUEST_TIMEOUT_S", env.get("REQUEST_TIMEOUT_S") or str(DEFAULT_REQUEST_TIMEOUT_S)
        ),
        # LOG_DIR set but empty disables the log file; stdout logging stays on.
        log_dir=env.get("LOG_DIR", DEFAULT_LOG_DIR) or None,
        context_char_budget=_parse_int(
            "CONTEXT_CHAR_BUDGET", env.get("CONTEXT_CHAR_BUDGET", "16000"), 1000, 100000
        ),
        evidence_item_char_limit=_parse_int(
            "EVIDENCE_ITEM_CHAR_LIMIT", env.get("EVIDENCE_ITEM_CHAR_LIMIT", "2400"), 100, 10000
        ),
        max_sentence_buffer_chars=_parse_int(
            "MAX_SENTENCE_BUFFER_CHARS", env.get("MAX_SENTENCE_BUFFER_CHARS", "32768"), 1024, 200000
        ),
        prompt_guard_threshold=_parse_unit_float(
            "PROMPT_GUARD_THRESHOLD", env.get("PROMPT_GUARD_THRESHOLD", "0.5")
        ),
        topic_gate_margin=_parse_margin("TOPIC_GATE_MARGIN", env.get("TOPIC_GATE_MARGIN", "0.0")),
        data_dir=env.get("DATA_DIR") or None,
        evidence_guard_threshold=_parse_unit_float(
            "EVIDENCE_GUARD_THRESHOLD", env.get("EVIDENCE_GUARD_THRESHOLD", "0.5")
        ),
        nli_model=env.get("NLI_MODEL") or DEFAULT_NLI_MODEL,
        nli_entailment_threshold=_parse_unit_float(
            "NLI_ENTAILMENT_THRESHOLD", env.get("NLI_ENTAILMENT_THRESHOLD", "0.5")
        ),
        nli_contradiction_threshold=_parse_unit_float(
            "NLI_CONTRADICTION_THRESHOLD", env.get("NLI_CONTRADICTION_THRESHOLD", "0.5")
        ),
        llm_max_retries=_parse_int("LLM_MAX_RETRIES", env.get("LLM_MAX_RETRIES", "1"), 0, 3),
        llm_retry_backoff_s=_parse_float(
            "LLM_RETRY_BACKOFF_S", env.get("LLM_RETRY_BACKOFF_S", "0.2")
        ),
        llm_max_output_tokens=_parse_int(
            "LLM_MAX_OUTPUT_TOKENS", env.get("LLM_MAX_OUTPUT_TOKENS", "1024"), 64, 8192
        ),
        llm_reasoning_effort=env.get("LLM_REASONING_EFFORT") or None,
    )
