import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from scilitrag.config import Settings

# Components that need the service; it is only checked when one of them runs for real.
DB_COMPONENTS = frozenset({"retrieve", "sql_facts"})
LLM_COMPONENTS = frozenset({"plan", "generate"})
# Checks run in parallel and stay well below the 3 s compose healthcheck timeout.
TIMEOUT_S = 1.0


def _needed(settings: Settings, components: frozenset[str]) -> bool:
    return not settings.use_mocks or bool(settings.real_components & components)


def check_database(url: str | None) -> str:
    """Reachability of the database host and port; no login, no driver needed."""
    if not url:
        return "not configured"
    try:
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port or 5432
    except ValueError:
        return "invalid url"
    if not host:
        return "invalid url"
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT_S):
            return "ok"
    except OSError:
        return "unreachable"


def check_llm(base_url: str | None) -> str:
    if not base_url:
        return "not configured"
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=TIMEOUT_S):
            return "ok"
    except urllib.error.HTTPError as exc:
        # Reachable, but wrong path or credentials.
        return f"http {exc.code}"
    except ValueError:
        return "invalid url"
    except OSError:
        return "unreachable"


def dependency_checks(settings: Settings) -> dict[str, str]:
    checks = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if _needed(settings, DB_COMPONENTS):
            checks["database"] = pool.submit(check_database, settings.database_url)
        if _needed(settings, LLM_COMPONENTS):
            checks["llm"] = pool.submit(check_llm, settings.llm_base_url)
    return {
        "database": checks["database"].result() if "database" in checks else "skipped",
        "llm": checks["llm"].result() if "llm" in checks else "skipped",
    }
