"""Disposable database fixtures using the checked-in application schema."""

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row


@pytest.fixture(scope="module")
def fixture_reader():
    dsn = os.environ.get("RETRIEVAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set RETRIEVAL_TEST_DATABASE_URL to a disposable ParadeDB database.")
    schema = "retrieval_test_" + uuid4().hex
    root = Path(__file__).resolve().parents[1]
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        for extension in ("vector", "pg_search", "pg_trgm"):
            conn.execute(
                sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(sql.Identifier(extension))
            )
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            schema_sql = (root / "docker/db/schema.sql").read_text(encoding="utf-8")
            conn.execute(
                "\n".join(line for line in schema_sql.splitlines() if not line.startswith("\\"))
            )
            conn.execute(
                (root / "tests/fixtures/retrieval_synthetic.sql").read_text(encoding="utf-8")
            )
            search_indexes = root / "docker/db/retrieval_indexes.sql"
            if search_indexes.exists():
                conn.execute(search_indexes.read_text(encoding="utf-8"))

            def reader(statement, params=None):
                with conn.transaction():
                    conn.execute("SET TRANSACTION READ ONLY")
                    return conn.execute(statement, params if params is not None else {}).fetchall()

            yield reader
        finally:
            conn.execute("SET search_path TO public")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
