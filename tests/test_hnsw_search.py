"""D-14 filtered HNSW search against a disposable ParadeDB database (skipped without one)."""

import os
import re
from pathlib import Path
from uuid import uuid4

import pytest

INDEXES = Path(__file__).resolve().parents[1] / "docker/db/indexes.sql"


def configured(name):
    """The value indexes.sql sets as a database default."""
    match = re.search(rf"SET {re.escape(name)} = (\w+)", INDEXES.read_text(encoding="utf-8"))
    assert match, f"{name} is not configured in indexes.sql"
    return match[1]


def test_filtered_dense_search_returns_every_requested_candidate():
    psycopg = pytest.importorskip("psycopg")
    dsn = os.environ.get("RETRIEVAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set RETRIEVAL_TEST_DATABASE_URL to a disposable ParadeDB database.")
    schema = "hnsw_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            conn.execute(f'SET search_path TO "{schema}", public')
            conn.execute("CREATE TABLE items (id int PRIMARY KEY, year int, embedding vector(3))")
            # 950 old papers sit close to the query; the 100 recent ones lie farther away.
            conn.execute(
                "INSERT INTO items SELECT i, CASE WHEN i <= 950 THEN 2000 ELSE 2024 END, "
                "CASE WHEN i <= 950 THEN ARRAY[1, i / 10000.0, 0] "
                "ELSE ARRAY[0, 1, i / 10000.0] END::vector FROM generate_series(1, 1050) i"
            )
            conn.execute("CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)")
            conn.execute("ANALYZE items")
            conn.execute("SET enable_seqscan = off")
            conn.execute(f"SET hnsw.ef_search = {configured('hnsw.ef_search')}")
            conn.execute(f"SET hnsw.iterative_scan = {configured('hnsw.iterative_scan')}")
            query = (
                "SELECT id FROM items WHERE year >= 2024 "
                "ORDER BY embedding <=> '[1,0,0]'::vector LIMIT 100"
            )
            assert "items_embedding_idx" in str(conn.execute("EXPLAIN " + query).fetchall())
            assert len(conn.execute(query).fetchall()) == 100
        finally:
            conn.execute("SET search_path TO public")
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
