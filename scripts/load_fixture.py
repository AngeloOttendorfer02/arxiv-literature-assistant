"""Load D-07 paper metadata and internal citations into an empty D-04 database.

Usage: uv run --locked --group data --group rag python scripts/load_fixture.py
Use --check-only to validate local fixture files without connecting to PostgreSQL.
Authors, chunks and embeddings are populated by later tasks.
"""

import argparse
import json
import os
import re
from time import perf_counter

from filter_data import batches
from select_fixture import DATASET, FILES, REVISION, data_directory, digest

PAPER_COPY = "COPY public.papers (corpus_id, title, abstract, year) FROM STDIN"
CITATION_COPY = "COPY public.citations (citing_id, cited_id) FROM STDIN"
# O13: e-mails are masked at import; same pattern and placeholder as the mask_pii mock.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def mask_emails(text):
    return EMAIL.sub("<EMAIL>", text) if text else text


def fixture_records(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("fixture_version") != 0
        or manifest.get("dataset") != DATASET
        or manifest.get("dataset_revision") != REVISION
    ):
        raise ValueError("Fixture version or pinned dataset revision does not match.")
    for name in FILES:
        if digest(folder / name) != manifest["sha256"][name]:
            raise ValueError(f"Fixture checksum mismatch: {name}")
    papers, graph = [], {}
    for batch in batches([folder / "corpus_clean.arrow"]):
        for row in batch.select(["corpusid", "title", "abstract", "year", "citations"]).to_pylist():
            cid = row["corpusid"]
            if type(cid) is not int or cid in graph:
                raise ValueError("Fixture has invalid or duplicate corpus IDs.")
            if any(
                row[name] is not None and not isinstance(row[name], str)
                for name in ("title", "abstract")
            ):
                raise ValueError("Title and abstract must be text or NULL.")
            if row["year"] is not None and type(row["year"]) is not int:
                raise ValueError("Fixture years must be integers or NULL.")
            if not isinstance(row["citations"], list) or any(
                type(cid) is not int for cid in row["citations"]
            ):
                raise ValueError("Citation lists must contain integer corpus IDs.")
            papers.append(
                (cid, mask_emails(row["title"]), mask_emails(row["abstract"]), row["year"])
            )
            graph[cid] = set(row["citations"])
    ids = set(graph)
    if not 200 <= len(ids) <= 500 or sorted(ids) != manifest["corpusids"]:
        raise ValueError("Fixture must have 200 to 500 papers matching the manifest.")
    queries = [row for batch in batches([folder / "query.arrow"]) for row in batch.to_pylist()]
    if len(queries) != 20 or any(
        not row["corpusids"] or set(row["corpusids"]) - ids for row in queries
    ):
        raise ValueError("Fixture must have 20 queries with all target papers present.")
    s2orc_ids = [
        cid
        for batch in batches([folder / "corpus_s2orc.arrow"])
        for cid in batch.column("corpusid").to_pylist()
    ]
    if len(s2orc_ids) != len(ids) or set(s2orc_ids) != ids:
        raise ValueError("Fixture corpora are not aligned.")
    citations = sorted((citing, cited) for citing in ids for cited in graph[citing] & ids)
    if len(citations) != manifest["citation_count"]:
        raise ValueError("Fixture citation count differs from the manifest.")
    return sorted(papers), citations


def import_records(connection, papers, citations):
    connection.execute("SET LOCAL lock_timeout = '5s'")
    connection.execute("SET LOCAL statement_timeout = '50s'")
    # Prevent a concurrent import from passing the same empty-database check.
    connection.execute(
        "LOCK TABLE public.papers, public.authors, public.paper_authors, "
        "public.citations, public.chunks IN SHARE ROW EXCLUSIVE MODE"
    )
    occupied = connection.execute(
        "SELECT EXISTS (SELECT 1 FROM public.papers) "
        "OR EXISTS (SELECT 1 FROM public.authors) "
        "OR EXISTS (SELECT 1 FROM public.paper_authors) "
        "OR EXISTS (SELECT 1 FROM public.citations) "
        "OR EXISTS (SELECT 1 FROM public.chunks)"
    ).fetchone()[0]
    if occupied:
        raise ValueError("Database is not empty; no data will be replaced or appended.")
    metadata = connection.execute(
        "SELECT schema_version, embed_model FROM public.schema_metadata WHERE id = 1 FOR UPDATE"
    ).fetchone()
    if metadata != (1, None):
        raise ValueError("Expected schema version 1 with no published embedding model.")
    with connection.cursor() as cursor:
        for statement, records in ((PAPER_COPY, papers), (CITATION_COPY, citations)):
            with cursor.copy(statement) as copy:
                for record in records:
                    copy.write_row(record)
    connection.execute(
        "UPDATE public.schema_metadata SET dataset_revision = %s WHERE id = 1", (REVISION,)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    try:
        folder = data_directory() / "fixtures" / "v0"
        papers, citations = fixture_records(folder)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Fixture validation failed: {error}\n")
    if args.check_only:
        print(f"Valid fixture: {len(papers)} papers, 20 queries, {len(citations)} citations.")
        print(f"Manifest SHA-256: {digest(folder / 'manifest.json')}")
        print(f"Validation: {perf_counter() - started:.2f}s. No database connection.")
        return
    dsn = os.environ.get("INGEST_DATABASE_URL")
    if not dsn:
        parser.error("Set INGEST_DATABASE_URL in .env; the app URL is never used for imports.")
    import psycopg

    try:
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            import_records(connection, papers, citations)
    except ValueError as error:
        parser.exit(1, f"Fixture import refused: {error}\n")
    except psycopg.Error as error:
        parser.exit(
            1,
            f"Database import failed (SQLSTATE {error.sqlstate or 'unavailable'}); transaction rolled back.\n",
        )
    elapsed = perf_counter() - started
    print(f"Loaded {len(papers)} papers and {len(citations)} citations in {elapsed:.2f}s.")
    print("D-07 runtime target: " + ("under 60s." if elapsed < 60 else "not met (60s or longer)."))
    print("Authors, chunks and embeddings are not populated; hybrid search is not ready yet.")


if __name__ == "__main__":
    main()
