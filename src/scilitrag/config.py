import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_LOG_DIR = "logs"
REQUIRED_WITHOUT_MOCKS = ("DATABASE_URL", "LLM_BASE_URL", "LLM_MODEL")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
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


def _parse_float(name: str, value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a positive finite number, got {value!r}")
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
        embed_model=env.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL,
        use_mocks=use_mocks,
        real_components=frozenset(name.strip() for name in real.split(",") if name.strip()),
        request_timeout_s=_parse_float(
            "REQUEST_TIMEOUT_S", env.get("REQUEST_TIMEOUT_S") or str(DEFAULT_REQUEST_TIMEOUT_S)
        ),
        # LOG_DIR set but empty disables the log file; stdout logging stays on.
        log_dir=env.get("LOG_DIR", DEFAULT_LOG_DIR) or None,
        llm_max_retries=_parse_int("LLM_MAX_RETRIES", env.get("LLM_MAX_RETRIES", "1"), 0, 3),
        llm_retry_backoff_s=_parse_float(
            "LLM_RETRY_BACKOFF_S", env.get("LLM_RETRY_BACKOFF_S", "0.2")
        ),
        llm_max_output_tokens=_parse_int(
            "LLM_MAX_OUTPUT_TOKENS", env.get("LLM_MAX_OUTPUT_TOKENS", "1024"), 64, 8192
        ),
    )
