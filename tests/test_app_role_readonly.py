"""S-08: the database itself rejects writes from the application login (DATABASE_URL)."""

import os

import psycopg
import pytest
from psycopg import errors

# One column per table that exists in docker/db/schema.sql, for UPDATE statements.
TABLES = {
    "papers": "title",
    "authors": "name",
    "paper_authors": "corpus_id",
    "citations": "cited_id",
    "chunks": "section",
    "schema_metadata": "schema_version",
}
# Every attempt runs in a transaction that is rolled back, and row statements match no row,
# so a wrongly privileged role makes the test fail without changing data.
DDL = [
    "CREATE TABLE public.s08_probe (id int)",
    "DROP TABLE public.papers",
    "ALTER TABLE public.papers ADD COLUMN s08_probe int",
]


@pytest.fixture(scope="module")
def app_conn():
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("Set DATABASE_URL to a database with the SELECT-only app login (roles.sql).")
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        assert conn.execute("SHOW transaction_read_only").fetchone()[0] == "off", (
            "A read-only session would hide the missing privilege this test checks."
        )
        yield conn


def statements():
    for table, column in TABLES.items():
        # An explicit column list: authors.author_id is GENERATED ALWAYS and would fail first.
        yield f"INSERT INTO public.{table} ({column}) SELECT {column} FROM public.{table} WHERE false"
        yield f"UPDATE public.{table} SET {column} = {column} WHERE false"
        yield f"DELETE FROM public.{table} WHERE false"
        yield f"TRUNCATE public.{table}"
    yield from DDL


def test_app_login_can_read(app_conn):
    assert app_conn.execute("SELECT count(*) FROM public.papers").fetchone()[0] >= 0


def test_app_login_is_not_a_superuser_or_owner(app_conn):
    row = app_conn.execute(
        "SELECT rolsuper, rolcreatedb, rolcreaterole, "
        "(SELECT tableowner = current_user FROM pg_tables "
        " WHERE schemaname = 'public' AND tablename = 'papers') "
        "FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    assert row == (False, False, False, False)


@pytest.mark.parametrize("statement", list(statements()))
def test_write_with_app_login_is_rejected(app_conn, statement):
    with pytest.raises(errors.InsufficientPrivilege), app_conn.transaction(force_rollback=True):
        app_conn.execute(statement)
