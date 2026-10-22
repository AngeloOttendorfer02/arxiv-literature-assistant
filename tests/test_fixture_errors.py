"""Import diagnostics redact credentials without opening a database connection."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import quote

import load_fixture
import psycopg
import pytest


def test_client_error_without_sqlstate_keeps_its_cause():
    error = psycopg.OperationalError("connection timed out")
    detail = load_fixture.database_error_detail(error, "postgresql://demo:secret@localhost/db")
    assert detail == "OperationalError (SQLSTATE unavailable): connection timed out"


def test_server_error_uses_primary_message_without_copy_context():
    class ServerError(psycopg.errors.InsufficientPrivilege):
        @property
        def diag(self):
            return SimpleNamespace(message_primary="permission denied for table authors")

    error = ServerError("permission denied\nCONTEXT: COPY authors, private row contents")
    detail = load_fixture.database_error_detail(error, "dbname=test")
    assert "42501" in detail
    assert "permission denied for table authors" in detail
    assert "CONTEXT" not in detail
    assert "private row contents" not in detail


def test_passwords_tokens_and_connection_urls_are_redacted(monkeypatch):
    password = "example p@ss/word"
    encoded = quote(password, safe="")
    dsn = f"postgresql://demo:{encoded}@localhost/test"
    monkeypatch.setenv("APP_DB_PASSWORD", "another-password")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    error = psycopg.OperationalError(
        f"Failure: {dsn}; decoded {password}; encoded {encoded}; "
        "another-password hf_test_token postgresql://other:unknown@remote/test"
    )
    detail = load_fixture.database_error_detail(error, dsn)
    for hidden in (
        password,
        encoded,
        dsn,
        "another-password",
        "hf_test_token",
        "unknown",
        "remote",
    ):
        assert hidden not in detail
    assert "<CONNECTION>" in detail
    assert "<REDACTED>" in detail


def test_conninfo_password_and_unknown_password_assignment_are_redacted():
    dsn = "host=localhost dbname=test user=demo password='a b\\'c'"
    error = psycopg.ProgrammingError(f"Failure: {dsn}; decoded a b'c; password=unknown-password")
    detail = load_fixture.database_error_detail(error, dsn)
    assert "a b'c" not in detail
    assert "unknown-password" not in detail
    assert dsn not in detail


def test_invalid_dsn_does_not_break_error_reporting():
    error = psycopg.ProgrammingError('invalid connection string; password="unknown value"')
    detail = load_fixture.database_error_detail(error, "not-a-valid-dsn")
    assert "ProgrammingError" in detail
    assert "invalid connection string" in detail
    assert "unknown value" not in detail


@pytest.mark.parametrize("during_import", [False, True])
def test_cli_reports_redacted_connection_or_copy_failure(
    tmp_path, monkeypatch, capsys, during_import
):
    dsn = "postgresql://demo:fake-password@localhost/test"
    monkeypatch.setenv("INGEST_DATABASE_URL", dsn)
    monkeypatch.setattr(load_fixture, "data_directory", lambda: tmp_path)
    monkeypatch.setattr(load_fixture, "fixture_records", lambda folder: ([], []))
    monkeypatch.setattr(load_fixture, "fixture_authors", lambda folder: ([], [], {}))
    monkeypatch.setattr("sys.argv", ["load_fixture.py"])

    def fail(*args, **kwargs):
        if during_import:
            raise psycopg.DataError(f"COPY adaptation failed: fake-password {dsn}")
        raise psycopg.OperationalError(f"connection timed out: fake-password {dsn}")

    context = MagicMock()
    if during_import:
        monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: context)
        monkeypatch.setattr(load_fixture, "import_records", fail)
    else:
        monkeypatch.setattr(psycopg, "connect", fail)
        monkeypatch.setattr(
            load_fixture, "import_records", lambda *a: pytest.fail("Unexpected import")
        )
    with pytest.raises(SystemExit) as caught:
        load_fixture.main()
    assert caught.value.code == 1
    output = capsys.readouterr().err
    assert ("DataError" if during_import else "OperationalError") in output
    assert ("COPY adaptation failed" if during_import else "connection timed out") in output
    assert "fake-password" not in output
    assert dsn not in output
    if during_import:
        assert context.__exit__.call_args.args[0] is psycopg.DataError
