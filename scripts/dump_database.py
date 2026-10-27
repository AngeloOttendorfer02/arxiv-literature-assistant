"""Write the D-16 dump: schema and data of the six project tables (pg_dump custom format).

Usage: uv run --locked --group data python scripts/dump_database.py [--name scilitrag_v2]
Run after scripts/embed_database.py has published embed_model and before building indexes,
so every restore builds the indexes of the current repository. Roles and grants are not in
the dump; each restore creates its own roles from .env. LitSearch data: share privately only.
"""

import argparse
import os
import subprocess

from select_fixture import data_directory, digest

TABLES = ("papers", "authors", "paper_authors", "citations", "chunks", "schema_metadata")


def compose(*args, **kwargs):
    """Run a command in the compose db container."""
    return subprocess.run(["docker", "compose", "exec", "-T", "db", *args], check=True, **kwargs)


def admin():
    user = os.environ.get("POSTGRES_USER") or "postgres"
    return user, os.environ.get("POSTGRES_DB") or "scilitrag"


def psql_value(sql):
    user, database = admin()
    result = compose(
        "psql", "-X", "-tA", "-U", user, "-d", database, "-c", sql, capture_output=True, text=True
    )
    return result.stdout.strip()


def write_dump(target):
    user, database = admin()
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as out:
        compose(
            "pg_dump",
            "-U",
            user,
            "-d",
            database,
            "-Fc",
            "--no-owner",
            "--no-acl",
            *(f"--table=public.{table}" for table in TABLES),
            stdout=out,
        )
    partial.replace(target)
    checksum = digest(target)
    # sha256sum format, so the file can also be checked without Python.
    target.with_name(target.name + ".sha256").write_text(
        f"{checksum}  {target.name}\n", encoding="utf-8", newline="\n"
    )
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="scilitrag_v2")
    args = parser.parse_args()
    target = data_directory() / "dumps" / f"{args.name}.dump"
    if target.exists():
        parser.exit(1, f"{target} exists; it will not be overwritten.\n")
    try:
        model = psql_value(
            "SELECT coalesce(embed_model, '') FROM public.schema_metadata WHERE id = 1"
        )
        if not model or model.startswith("pending:"):
            parser.exit(1, "Embeddings are not published; run scripts/embed_database.py first.\n")
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = write_dump(target)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Dump failed: {error}\n")
    print(f"Dump: {target} ({target.stat().st_size / 2**20:.0f} MiB), embed_model {model}.")
    print(f"SHA-256: {checksum} (also in {target.name}.sha256).")
    print("Share privately within the team only; LitSearch must not be redistributed.")


if __name__ == "__main__":
    main()
