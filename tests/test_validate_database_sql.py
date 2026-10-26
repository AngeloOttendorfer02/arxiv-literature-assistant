"""D-17 regression checks using real SQL in isolated disposable-database schemas."""

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from validate_database import INFO_CHECKS, NULL_CHECKS, foreign_keys

ROOT = Path(__file__).resolve().parents[1]
CHECKS = dict(NULL_CHECKS + INFO_CHECKS)


@pytest.fixture
def validation_db():
    dsn = os.environ.get("RETRIEVAL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set RETRIEVAL_TEST_DATABASE_URL to a disposable ParadeDB database.")
    schema = "validation_test_" + uuid4().hex
    other_schema = schema + "_other"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(other_schema)))
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            schema_sql = (ROOT / "docker/db/schema.sql").read_text(encoding="utf-8")
            conn.execute("\n".join(s for s in schema_sql.splitlines() if not s.startswith("\\")))
            conn.execute(
                "UPDATE schema_metadata SET dataset_revision='r1', "
                "embed_model='model', embed_fingerprint='fingerprint'"
            )
            yield conn, other_schema
        finally:
            conn.rollback()
            conn.execute("SET search_path TO public")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(other_schema))
            )


@pytest.mark.parametrize("value", [None, "", " \t\n\r "])
@pytest.mark.parametrize("field", ["dataset_revision", "embed_model", "embed_fingerprint"])
def test_metadata_rejects_missing_and_whitespace_values(validation_db, field, value):
    conn, _ = validation_db
    conn.execute(
        sql.SQL("UPDATE schema_metadata SET {}=%s").format(sql.Identifier(field)), (value,)
    )
    assert conn.execute(CHECKS["metadata not published"]).fetchone()[0] == 1


def test_metadata_requires_a_published_model(validation_db):
    conn, _ = validation_db
    assert conn.execute(CHECKS["metadata not published"]).fetchone()[0] == 0
    conn.execute("UPDATE schema_metadata SET embed_model='pending:model'")
    assert conn.execute(CHECKS["metadata not published"]).fetchone()[0] == 1


@pytest.mark.parametrize("value, expected", [("", 1), (" \t\n\r ", 1), ("Text\tcontent", 0)])
@pytest.mark.parametrize("field", ["text", "name", "normalized_name"])
def test_blank_required_text_recognizes_whitespace(validation_db, field, value, expected):
    conn, _ = validation_db
    if field == "text":
        conn.execute("INSERT INTO papers (corpus_id) VALUES (1)")
        conn.execute(
            "INSERT INTO chunks (chunk_id, corpus_id, seq_no, text) VALUES ('1-0', 1, 0, %s)",
            (value,),
        )
        check = "chunks with blank text"
    else:
        conn.execute(
            "INSERT INTO authors (name, normalized_name) VALUES (%s, %s)",
            (value, "valid") if field == "name" else ("Valid", value),
        )
        check = "authors with blank name"
    assert conn.execute(CHECKS[check]).fetchone()[0] == expected


def test_blank_paper_info_recognizes_whitespace_and_null(validation_db):
    conn, _ = validation_db
    conn.execute("INSERT INTO papers (corpus_id, title, abstract) VALUES (1, %s, NULL)", ("\t\n",))
    assert conn.execute(CHECKS["papers without title and abstract"]).fetchone()[0] == 1


def key_results(conn):
    return {name: ok for name, ok, _ in foreign_keys(lambda query: conn.execute(query).fetchall())}


def test_declared_schema_keys_pass(validation_db):
    conn, _ = validation_db
    assert all(key_results(conn).values())


@pytest.mark.parametrize("problem", ["wrong_column", "wrong_schema", "composite", "unvalidated"])
def test_incorrect_foreign_key_definition_fails_even_without_orphans(validation_db, problem):
    conn, other_schema = validation_db
    conn.execute("ALTER TABLE chunks DROP CONSTRAINT chunks_corpus_id_fkey")
    if problem == "wrong_column":
        conn.execute("ALTER TABLE papers ADD COLUMN alternate_id bigint UNIQUE")
        conn.execute(
            "ALTER TABLE chunks ADD FOREIGN KEY (corpus_id) REFERENCES papers(alternate_id)"
        )
    elif problem == "wrong_schema":
        conn.execute(
            sql.SQL("CREATE TABLE {}.papers (corpus_id bigint PRIMARY KEY)").format(
                sql.Identifier(other_schema)
            )
        )
        conn.execute(
            sql.SQL(
                "ALTER TABLE chunks ADD FOREIGN KEY (corpus_id) REFERENCES {}.papers(corpus_id)"
            ).format(sql.Identifier(other_schema))
        )
    elif problem == "composite":
        conn.execute("ALTER TABLE papers ADD COLUMN alternate_id bigint")
        conn.execute("ALTER TABLE papers ADD UNIQUE (corpus_id, alternate_id)")
        conn.execute("ALTER TABLE chunks ADD COLUMN alternate_id bigint")
        conn.execute(
            "ALTER TABLE chunks ADD FOREIGN KEY (corpus_id, alternate_id) "
            "REFERENCES papers(corpus_id, alternate_id)"
        )
    else:
        conn.execute(
            "ALTER TABLE chunks ADD FOREIGN KEY (corpus_id) REFERENCES papers(corpus_id) NOT VALID"
        )
    results = key_results(conn)
    assert results.pop("fk chunks.corpus_id") is False
    assert all(results.values())
