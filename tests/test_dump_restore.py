"""D-16 dump and restore steps with a fake docker compose, no database needed."""

import subprocess
from types import SimpleNamespace

import dump_database
import pytest
import restore_database
from select_fixture import digest


class FakeCompose:
    def __init__(self, values):
        self.values = values  # answers for psql -c queries, in call order
        self.calls = []

    def __call__(self, command, check, **kwargs):
        args = command[5:]  # strip "docker compose exec -T db"
        self.calls.append(args)
        if args[0] == "pg_dump":
            kwargs["stdout"].write(b"PGDMP fake dump")
        if "-tA" in args:
            return SimpleNamespace(stdout=self.values.pop(0))
        return SimpleNamespace(stdout="")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    def install(*values):
        compose = FakeCompose(list(values))
        monkeypatch.setattr(subprocess, "run", compose)
        return compose

    for module in (dump_database, restore_database):
        monkeypatch.setattr(module, "data_directory", lambda: tmp_path)
    return install


def run(module, monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", [module.__name__, *argv])
    module.main()


def test_dump_writes_file_and_checksum_without_roles_or_grants(tmp_path, fake, monkeypatch):
    compose = fake("ibm-granite/granite-embedding-english-r2")
    run(dump_database, monkeypatch)
    dump = tmp_path / "dumps/scilitrag_v1.dump"
    assert dump.read_bytes() == b"PGDMP fake dump"
    assert (tmp_path / "dumps/scilitrag_v1.dump.sha256").read_text() == (
        f"{digest(dump)}  scilitrag_v1.dump\n"
    )
    pg_dump = compose.calls[-1]
    assert pg_dump[0] == "pg_dump" and "--no-owner" in pg_dump and "--no-acl" in pg_dump
    assert {arg for arg in pg_dump if arg.startswith("--table=")} == {
        f"--table=public.{table}" for table in dump_database.TABLES
    }


@pytest.mark.parametrize("model", ["", "pending:ibm-granite/granite-embedding-english-r2@x"])
def test_dump_refuses_unpublished_embeddings(fake, monkeypatch, model):
    compose = fake(model)
    with pytest.raises(SystemExit):
        run(dump_database, monkeypatch)
    assert all(call[0] != "pg_dump" for call in compose.calls)


def test_dump_never_overwrites(tmp_path, fake, monkeypatch):
    (tmp_path / "dumps").mkdir()
    (tmp_path / "dumps/scilitrag_v1.dump").write_bytes(b"old")
    compose = fake()
    with pytest.raises(SystemExit):
        run(dump_database, monkeypatch)
    assert compose.calls == [] and (tmp_path / "dumps/scilitrag_v1.dump").read_bytes() == b"old"


def write_dump(tmp_path, checksum=None):
    dump = tmp_path / "v1.dump"
    dump.write_bytes(b"PGDMP fake dump")
    (tmp_path / "v1.dump.sha256").write_text(f"{checksum or digest(dump)}  v1.dump\n")
    return dump


def test_restore_runs_roles_before_indexes_and_analyzes(tmp_path, fake, monkeypatch, capsys):
    dump = write_dump(tmp_path)
    compose = fake("f", "56115 papers, ...")
    run(restore_database, monkeypatch, str(dump))
    steps = [
        "query" if "-tA" in call else call[-1] if call[0] == "psql" else call[0]
        for call in compose.calls
    ]
    assert steps == [
        "query",  # empty-database check
        "pg_restore",
        "/opt/scilitrag/db/roles.sql",
        "/opt/scilitrag/db/indexes.sql",
        "/opt/scilitrag/db/retrieval_indexes.sql",
        "ANALYZE papers, authors, paper_authors, citations, chunks;",
        "query",  # final counts
    ]
    restore_call = compose.calls[1]
    assert "--no-owner" in restore_call and "--exit-on-error" in restore_call
    assert "checksum ok" in capsys.readouterr().out


def test_restore_refuses_a_database_with_tables(tmp_path, fake, monkeypatch):
    compose = fake("t")
    with pytest.raises(SystemExit):
        run(restore_database, monkeypatch, str(write_dump(tmp_path)))
    assert all(call[0] != "pg_restore" for call in compose.calls)


def test_restore_refuses_a_damaged_copy(tmp_path, fake, monkeypatch):
    compose = fake()
    with pytest.raises(SystemExit):
        run(restore_database, monkeypatch, str(write_dump(tmp_path, checksum="0" * 64)))
    assert compose.calls == []
