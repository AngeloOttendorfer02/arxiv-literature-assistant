import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
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


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no", ""):
        return False
    raise ConfigError(f"{name} must be 0 or 1, got {value!r}")


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
    )
