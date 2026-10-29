"""Bounded, parameterized database reads for retrieval components."""

import logging
import math
import time

import psycopg
from psycopg.rows import dict_row

from scilitrag.config import Settings
from scilitrag.rag.errors import RagUserError

logger = logging.getLogger(__name__)


class DatabaseReader:
    def __init__(self, settings: Settings, *, connect=None, statement_timeout_s=None) -> None:
        self.settings = settings
        self.connect = connect or psycopg.connect
        # Interactive callers keep the request budget and 10-second cap. Offline
        # evaluation can explicitly allow slower reads without changing API settings.
        self.statement_timeout_s = (
            min(10.0, settings.request_timeout_s)
            if statement_timeout_s is None
            else statement_timeout_s
        )
        if not math.isfinite(self.statement_timeout_s) or self.statement_timeout_s <= 0:
            raise ValueError("The database statement timeout must be positive and finite.")

    def __call__(self, statement: str, params=None) -> list[dict]:
        if not self.settings.database_url:
            raise RagUserError("The database is not configured.")
        started = time.perf_counter()
        try:
            with self.connect(
                self.settings.database_url, connect_timeout=5, row_factory=dict_row
            ) as conn:
                # Set transaction characteristics before the first statement starts it.
                conn.read_only = True
                timeout_ms = max(1, int(self.statement_timeout_s * 1000))
                conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(timeout_ms),))
                return conn.execute(statement, params if params is not None else {}).fetchall()
        except psycopg.errors.QueryCanceled:
            raise RagUserError("The database search timed out. Try a narrower query.") from None
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            raise RagUserError(
                "The database schema is incomplete. Run the database preflight."
            ) from None
        except psycopg.Error:
            raise RagUserError("The literature database is unavailable or not ready.") from None
        finally:
            logger.info("db_read latency_ms=%.3f", (time.perf_counter() - started) * 1000)


def vector_literal(values: list[float], dimension: int) -> str:
    try:
        valid = len(values) == dimension and all(
            not isinstance(value, bool) and math.isfinite(value) for value in values
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise RagUserError("The query embedding does not match the database dimension.")
    if not any(value != 0 for value in values):
        raise RagUserError("The embedding model returned a zero vector.")
    return "[" + ",".join(str(float(value)) for value in values) + "]"
