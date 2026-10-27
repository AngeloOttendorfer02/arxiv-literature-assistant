"""D-17 validation of the loaded database: row counts, foreign keys, nulls and e-mails.

Usage: uv run --locked --group data --group rag python scripts/validate_database.py
Reads through the SELECT-only app login (DATABASE_URL) in a read-only transaction.
Exits with 1 if a check fails; INFO lines report documented exceptions.
"""

import argparse
import json
from time import perf_counter

from email_masking import count_emails

ENTITIES = ("papers", "authors", "paper_authors", "citations", "chunks")
MIN_ROWS, MIN_ENTITIES = 1000, 3
# (child table, column, parent table, parent key) as declared in schema.sql.
FOREIGN_KEYS = (
    ("paper_authors", "author_id", "authors", "author_id"),
    ("paper_authors", "corpus_id", "papers", "corpus_id"),
    ("citations", "citing_id", "papers", "corpus_id"),
    ("citations", "cited_id", "papers", "corpus_id"),
    ("chunks", "corpus_id", "papers", "corpus_id"),
)
DECLARED_KEYS = """
    SELECT child.relname, a.attname, parent.relname, target.attname, c.convalidated
    FROM pg_constraint c
    JOIN pg_class child ON child.oid = c.conrelid
    JOIN pg_class parent ON parent.oid = c.confrelid
    JOIN pg_namespace n ON n.oid = child.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
    JOIN pg_attribute target ON target.attrelid = c.confrelid AND target.attnum = c.confkey[1]
    WHERE c.contype = 'f' AND n.nspname = current_schema()
      AND parent.relnamespace = child.relnamespace
      AND cardinality(c.conkey) = 1 AND cardinality(c.confkey) = 1
"""
# Required values; each query counts violations and must return 0.
NULL_CHECKS = (
    ("papers without embedding", "SELECT count(*) FROM papers WHERE embedding IS NULL"),
    ("chunks without embedding", "SELECT count(*) FROM chunks WHERE embedding IS NULL"),
    (
        "chunks with blank text",
        "SELECT count(*) FROM chunks WHERE coalesce(text, '') ~ '^[[:space:]]*$'",
    ),
    (
        "authors with blank name",
        (
            "SELECT count(*) FROM authors WHERE coalesce(name, '') ~ '^[[:space:]]*$' "
            "OR coalesce(normalized_name, '') ~ '^[[:space:]]*$'"
        ),
    ),
    (
        "metadata not published",
        (
            "SELECT 1 - count(*) FROM schema_metadata WHERE id = 1 AND schema_version = 1 "
            "AND dataset_revision ~ '[^[:space:]]' AND embed_model ~ '[^[:space:]]' "
            "AND embed_model NOT LIKE 'pending:%' AND embed_fingerprint ~ '[^[:space:]]'"
        ),
    ),
)
# Documented exceptions, reported but not failed (D021: protected targets keep empty text).
INFO_CHECKS = (
    (
        "papers without title and abstract",
        (
            "SELECT count(*) FROM papers WHERE coalesce(title, '') ~ '^[[:space:]]*$' "
            "AND coalesce(abstract, '') ~ '^[[:space:]]*$'"
        ),
    ),
    ("papers without year", "SELECT count(*) FROM papers WHERE year IS NULL"),
)
TEXT_FIELDS = (
    ("papers.title and abstract", "SELECT title, abstract FROM papers"),
    ("chunks.section and text", "SELECT section, text FROM chunks"),
    ("authors.name", "SELECT name, first_name, last_name FROM authors"),
)


def row_counts(query):
    """Rows per table and whether enough entities reach the minimum size."""
    counts = {table: query(f"SELECT count(*) FROM {table}")[0][0] for table in ENTITIES}
    large = [table for table, count in counts.items() if count >= MIN_ROWS]
    detail = ", ".join(f"{table} {count:,}" for table, count in counts.items())
    ok = len(large) >= MIN_ENTITIES
    return [("rows", ok, f"{detail}; {len(large)} entities with at least {MIN_ROWS:,}")]


def foreign_keys(query):
    """Declared, validated constraints and zero orphans for every expected key."""
    declared = {
        (child, column, parent, key)
        for child, column, parent, key, valid in query(DECLARED_KEYS)
        if valid
    }
    results = []
    for child, column, parent, key in FOREIGN_KEYS:
        orphans = query(
            f"SELECT count(*) FROM {child} c LEFT JOIN {parent} p ON p.{key} = c.{column} "
            f"WHERE p.{key} IS NULL"
        )[0][0]
        valid = (child, column, parent, key) in declared
        ok = valid and orphans == 0
        state = "declared and validated" if valid else "not declared or not validated"
        results.append((f"fk {child}.{column}", ok, f"{state}; {orphans} orphans"))
    return results


def null_checks(query):
    results = [(name, (count := query(sql)[0][0]) == 0, f"{count}") for name, sql in NULL_CHECKS]
    results += [(name, None, f"{query(sql)[0][0]}") for name, sql in INFO_CHECKS]
    return results


def email_checks(stream):
    """Recognized addresses left in indexed and displayed text; must be 0 (O13, D-11)."""
    results = []
    for name, sql in TEXT_FIELDS:
        found = sum(count_emails(value) for row in stream(sql) for value in row)
        results.append((f"e-mails in {name}", found == 0, f"{found}"))
    return results


def validate(query, stream):
    return row_counts(query) + foreign_keys(query) + null_checks(query) + email_checks(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Also write the results to this file.")
    args = parser.parse_args()
    started = perf_counter()
    import psycopg

    from scilitrag.config import load_settings

    settings = load_settings()
    if not settings.database_url:
        parser.error("Set DATABASE_URL (the read-only app login) in .env.")
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            conn.read_only = True

            def query(sql):
                return conn.execute(sql).fetchall()

            def stream(sql):
                with conn.cursor(name="validate_texts") as cursor:
                    cursor.itersize = 2000
                    cursor.execute(sql)
                    yield from cursor

            results = validate(query, stream)
    except psycopg.Error as error:
        parser.exit(2, f"Validation could not read the database: {type(error).__name__}\n")
    for name, ok, detail in results:
        label = "INFO" if ok is None else "PASS" if ok else "FAIL"
        print(f"{label:4}  {name}: {detail}")
    failed = [name for name, ok, _ in results if ok is False]
    print(f"{len(failed)} failed checks; {perf_counter() - started:.1f}s; read-only, no changes.")
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as stream_out:
            records = [{"check": n, "status": ok, "detail": d} for n, ok, d in results]
            json.dump(records, stream_out, indent=2)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
