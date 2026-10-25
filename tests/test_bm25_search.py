"""D-15 BM25 behavior on the pinned ParadeDB image; needs a disposable test DB."""

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.backend import BM25_SQL, CHUNK_BM25_SQL, SearchBackend
from scilitrag.rag.search.readiness import validate_search_indexes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bm25_database():
    dsn = os.environ.get("RETRIEVAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set RETRIEVAL_TEST_DATABASE_URL to a disposable ParadeDB database.")
    schema = "bm25_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        for extension in ("vector", "pg_search"):
            conn.execute(
                sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(sql.Identifier(extension))
            )
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            schema_sql = (ROOT / "docker/db/schema.sql").read_text(encoding="utf-8")
            conn.execute("\n".join(s for s in schema_sql.splitlines() if not s.startswith("\\")))
            conn.execute("""
                INSERT INTO papers (corpus_id, title, abstract, year) VALUES
                    (1, 'Quasar retrieval', NULL, 2021),
                    (2, NULL, 'Quasar experiments', 2022),
                    (3, 'QUASAR', 'Quasar retrieval', 2018),
                    (4, '', '', 2023),
                    (5, NULL, NULL, 2023),
                    (6, 'Unrelated paper', 'Vision model', 2024);
                INSERT INTO chunks (chunk_id, corpus_id, seq_no, text) VALUES
                    ('1-0', 1, 0, 'Nebula experiment'),
                    ('2-0', 2, 0, 'Nebula experiment'),
                    ('2-1', 2, 1, 'Completely unrelated content');
            """)
            conn.execute((ROOT / "docker/db/retrieval_indexes.sql").read_text(encoding="utf-8"))

            def reader(statement, params=None):
                with conn.transaction():
                    conn.execute("SET TRANSACTION READ ONLY")
                    return conn.execute(statement, params or {}).fetchall()

            yield conn, reader
        finally:
            # Recover from a failed multi-statement migration before cleaning up.
            conn.rollback()
            conn.execute("SET search_path TO public")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_title_and_abstract_matches_include_nullable_fields_and_respect_years(bm25_database):
    _, reader = bm25_database
    backend = SearchBackend(reader, None)
    all_hits = backend.bm25(Plan(scenario="S1", k=5, topic="quasar"))
    assert {row["corpus_id"] for row in all_hits} == {1, 2, 3}
    assert all(row["score"] > 0 for row in all_hits)
    assert [row["score"] for row in all_hits] == sorted(
        (row["score"] for row in all_hits), reverse=True
    )
    filtered = backend.bm25(Plan(scenario="S1", k=5, topic="quasar", year_from=2020, year_to=2023))
    assert {row["corpus_id"] for row in filtered} == {1, 2}
    assert backend.bm25(Plan(scenario="S1", k=5, topic="nonexistenttoken")) == []


def test_title_score_is_halved_while_abstract_score_is_unchanged(bm25_database):
    _, reader = bm25_database
    baseline = reader(
        "SELECT p.corpus_id, pdb.score(p.corpus_id) AS score FROM papers p "
        "WHERE p.title ||| %(topic)s OR p.abstract ||| %(topic)s",
        {"topic": "quasar"},
    )
    unweighted = {row["corpus_id"]: row["score"] for row in baseline}
    rows = SearchBackend(reader, None).bm25(Plan(scenario="S1", k=5, topic="quasar"))
    weighted = {row["corpus_id"]: row["score"] for row in rows}
    assert weighted.keys() == unweighted.keys()
    # Paper 1 matches only the title; paper 2 matches only the abstract.
    assert weighted[1] == pytest.approx(unweighted[1] * 0.5)
    assert weighted[2] == pytest.approx(unweighted[2])


def test_title_matches_respect_author_paper_filters_and_limits(bm25_database):
    conn, reader = bm25_database
    conn.execute("INSERT INTO authors (name, normalized_name) VALUES ('Ada', 'ada')")
    conn.execute("INSERT INTO paper_authors (author_id, corpus_id) VALUES (1, 1)")
    backend = SearchBackend(reader, None)
    rows = backend.bm25(Plan(scenario="S1", k=5, topic="quasar", author_ref="Ada"))
    assert [row["corpus_id"] for row in rows] == [1]
    rows = backend.bm25(Plan(scenario="S1", k=5, topic="quasar", paper_ref="2"))
    assert [row["corpus_id"] for row in rows] == [2]
    assert len(backend.bm25(Plan(scenario="S1", k=5, topic="quasar"), limit=1)) == 1


def test_chunk_bm25_searches_text_only_within_selected_papers(bm25_database):
    _, reader = bm25_database
    rows = reader(CHUNK_BM25_SQL, {"paper_ids": [2], "topic": "nebula", "candidates": 10})
    assert [row["chunk_id"] for row in rows] == ["2-0"]
    assert rows[0]["score"] > 0
    assert (
        reader(CHUNK_BM25_SQL, {"paper_ids": [2], "topic": "nonexistenttoken", "candidates": 10})
        == []
    )


def test_installed_indexes_are_ready_reapplicable_and_used(bm25_database):
    conn, reader = bm25_database
    conn.execute((ROOT / "docker/db/retrieval_indexes.sql").read_text(encoding="utf-8"))
    validate_search_indexes(reader)
    params = SearchBackend(reader, None).params(Plan(scenario="S1", k=5, topic="quasar"), 10)
    plan = reader("EXPLAIN (FORMAT JSON) " + BM25_SQL, params)
    assert "papers_title_abstract_bm25_idx" in str(plan)
    plan = reader(
        "EXPLAIN (FORMAT JSON) " + CHUNK_BM25_SQL,
        {"paper_ids": [2], "topic": "nebula", "candidates": 10},
    )
    assert "chunks_text_bm25_idx" in str(plan)


def test_legacy_index_migration_preserves_data_and_enables_title_search(bm25_database):
    conn, reader = bm25_database
    conn.execute("DROP INDEX papers_title_abstract_bm25_idx")
    conn.execute("""
        CREATE INDEX papers_abstract_bm25_idx ON papers
        USING paradedb (corpus_id, abstract, year)
    """)
    with pytest.raises(RagUserError, match="002_bm25_title_abstract.sql"):
        validate_search_indexes(reader)
    migration = (ROOT / "docker/db/migrations/002_bm25_title_abstract.sql").read_text(
        encoding="utf-8"
    )
    conn.execute(migration)
    conn.execute(migration)
    validate_search_indexes(reader)
    assert reader("SELECT count(*) AS n FROM papers")[0]["n"] == 6
    assert reader("SELECT count(*) AS n FROM chunks")[0]["n"] == 3
    rows = SearchBackend(reader, None).bm25(Plan(scenario="S1", k=5, topic="quasar"))
    assert {row["corpus_id"] for row in rows} == {1, 2, 3}
