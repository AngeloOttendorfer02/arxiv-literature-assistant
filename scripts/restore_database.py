"""Restore a D-16 dump into a fresh database, then create roles and build the indexes.

Usage: uv run --locked --group data python scripts/restore_database.py data/dumps/scilitrag_v1.dump
Needs a fresh db volume (docker compose up -d db) and the role variables in .env.
Steps: pg_restore (schema and data), roles.sql, indexes.sql, retrieval_indexes.sql, ANALYZE.
The SQL files come from this repository, so indexes always match the current code.
"""

import argparse
import subprocess
from pathlib import Path

from dump_database import admin, compose, psql_value
from select_fixture import data_directory, digest

SQL_FILES = ("roles.sql", "indexes.sql", "retrieval_indexes.sql")
COUNTS = (
    "SELECT (SELECT count(*) FROM papers) || ' papers, ' || "
    "(SELECT count(*) FROM papers WHERE embedding IS NOT NULL) || ' with vectors, ' || "
    "(SELECT count(*) FROM chunks) || ' chunks, ' || "
    "(SELECT count(*) FROM citations) || ' citations, ' || "
    "(SELECT count(*) FROM authors) || ' authors; embed_model ' || "
    "(SELECT embed_model FROM schema_metadata WHERE id = 1)"
)


def check_checksum(path):
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        return "no .sha256 file, not checked"
    if sidecar.read_text(encoding="utf-8").split()[0] != digest(path):
        raise ValueError(f"{path.name} does not match {sidecar.name}; the copy is damaged.")
    return "checksum ok"


def restore(path):
    user, database = admin()
    with path.open("rb") as dump:
        compose(
            "pg_restore",
            "-U",
            user,
            "-d",
            database,
            "--no-owner",
            "--no-acl",
            "--exit-on-error",
            stdin=dump,
        )
    for name in SQL_FILES:
        print(f"Running {name} ...", flush=True)
        compose("psql", "-X", "-q", "-U", user, "-d", database, "-f", f"/opt/scilitrag/db/{name}")
    compose(
        "psql",
        "-X",
        "-q",
        "-U",
        user,
        "-d",
        database,
        "-c",
        "ANALYZE papers, authors, paper_authors, citations, chunks;",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, nargs="?")
    args = parser.parse_args()
    path = args.dump or data_directory() / "dumps" / "scilitrag_v1.dump"
    if not path.is_file():
        parser.exit(1, f"{path} not found.\n")
    try:
        print(f"{path.name}: {check_checksum(path)}.", flush=True)
        if psql_value("SELECT to_regclass('public.papers') IS NOT NULL") == "t":
            parser.exit(
                1,
                "The database already has project tables; restore needs a fresh volume. "
                "`docker compose down -v` deletes the current data.\n",
            )
        restore(path)
        print(f"Restored: {psql_value(COUNTS)}.")
    except ValueError as error:
        parser.exit(1, f"Restore refused: {error}\n")
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Restore failed: {error}\n")
    print(
        "Check with: uv run --locked --group rag python scripts/check_database.py --require-search"
    )


if __name__ == "__main__":
    main()
