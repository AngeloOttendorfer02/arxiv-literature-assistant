"""Load D-07 papers, D-09 internal citations and D-08 authors into an empty D-04 database.

Usage: uv run --locked --group data --group rag python scripts/load_fixture.py
Use --check-only to validate local fixture files without connecting to PostgreSQL.
Chunks and embeddings are populated by later tasks.
"""

import argparse
import json
import os
import re
from time import perf_counter
from urllib.parse import quote, quote_plus

from author_names import extract_authors
from citations import bibliography_counts, bibliography_ids, build_citations, direction_counts
from filter_data import batches
from select_fixture import DATASET, FILES, REVISION, data_directory, digest

PAPER_COPY = "COPY public.papers (corpus_id, title, abstract, year) FROM STDIN"
CITATION_COPY = "COPY public.citations (citing_id, cited_id) FROM STDIN"
AUTHOR_COPY = "COPY public.authors (first_name, last_name, name, normalized_name) FROM STDIN"
PAPER_AUTHOR_COPY = "COPY public.paper_authors (author_id, corpus_id) FROM STDIN"
# O13: e-mails are masked at import; same pattern and placeholder as the mask_pii mock.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def mask_emails(text):
    return EMAIL.sub("<EMAIL>", text) if text else text


def database_error_detail(error, dsn):
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    # Server context can include COPY row contents; prefer the primary message.
    detail = error.diag.message_primary or str(error) or "No error details available."
    secrets = {
        value
        for key, value in os.environ.items()
        if value and (key.endswith(("PASSWORD", "TOKEN", "SECRET")) or key == "PGPASSWORD")
    }
    try:
        password = conninfo_to_dict(dsn).get("password")
        if password:
            secrets.add(password)
    except psycopg.Error:
        pass
    detail = detail.replace(dsn, "<CONNECTION>")
    detail = re.sub(r"(?i)\bpostgres(?:ql)?://[^\s<>]+", "<CONNECTION>", detail)
    detail = re.sub(
        r"""(?i)\bpassword\s*=\s*(?:'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|[^\s]+)""",
        "password=<REDACTED>",
        detail,
    )
    variants = {
        variant
        for value in secrets
        for variant in (value, quote(value, safe=""), quote_plus(value))
    }
    for value in sorted(variants, key=len, reverse=True):
        detail = detail.replace(value, "<REDACTED>")
    detail = " ".join(detail.split())[:1000]
    return f"{type(error).__name__} (SQLSTATE {error.sqlstate or 'unavailable'}): {detail}"


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
            graph[cid] = row["citations"]
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
    # The manifest counts source pairs before D-09 drops self-citations.
    if sum(len(set(graph[cid]) & ids) for cid in ids) != manifest["citation_count"]:
        raise ValueError("Fixture citation count differs from the manifest.")
    citations, citation_counts = build_citations(graph, ids)
    return sorted(papers), citations, citation_counts


def fixture_bibliographies(folder):
    return {
        row["corpusid"]: bibliography_ids(row)
        for batch in batches([folder / "corpus_s2orc.arrow"])
        for row in batch.select(["corpusid", "content"]).to_pylist()
    }


def fixture_authors(folder):
    rows = (
        row
        for batch in batches([folder / "corpus_s2orc.arrow"])
        for row in batch.select(["corpusid", "content"]).to_pylist()
    )
    return extract_authors(rows)


def import_authors(connection, cursor, authors, links):
    with cursor.copy(AUTHOR_COPY) as copy:
        for author in authors:
            copy.write_row(author)
    rows = connection.execute(
        "SELECT author_id, normalized_name FROM public.authors ORDER BY author_id"
    ).fetchall()
    # IDs follow the sorted insert order only on a fresh identity sequence.
    if [author_id for author_id, _ in rows] != list(range(1, len(authors) + 1)) or [
        name for _, name in rows
    ] != [author[3] for author in authors]:
        raise ValueError(
            "Author IDs must start at 1; reset with TRUNCATE ... RESTART IDENTITY first."
        )
    ids = {name: author_id for author_id, name in rows}
    with cursor.copy(PAPER_AUTHOR_COPY) as copy:
        for name, corpus_id in links:
            copy.write_row((ids[name], corpus_id))


def import_records(connection, papers, citations, authors=(), links=()):
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
        if authors:
            import_authors(connection, cursor, authors, links)
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
        papers, citations, citation_counts = fixture_records(folder)
        authors, links, author_counts = fixture_authors(folder)
        years = {paper[0]: paper[3] for paper in papers}
        direction = direction_counts(citations, years)
        direction.update(bibliography_counts(citations, fixture_bibliographies(folder)))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Fixture validation failed: {error}\n")
    print(
        f"Citations: {len(citations)} from {citation_counts.get('entries', 0)} list entries; "
        f"dropped {citation_counts.get('outside_corpus', 0)} outside the fixture, "
        f"{citation_counts.get('self_citation', 0)} self-citations, "
        f"{citation_counts.get('duplicate', 0)} duplicates."
    )
    print(
        f"Direction: {direction.get('in_bibliography', 0)} of {len(citations)} in the citing "
        f"paper's S2ORC bibliography; citing paper newer in {direction.get('citing_newer', 0)}, "
        f"same year in {direction.get('same_year', 0)}, older in "
        f"{direction.get('citing_older', 0)}, year missing in {direction.get('year_missing', 0)}."
    )
    print(
        f"Authors: {len(authors)} with {len(links)} paper links from "
        f"{author_counts.get('author_blocks', 0)} author blocks; dropped "
        f"{author_counts.get('missing_first_or_last', 0)} without first or last name, "
        f"{author_counts.get('affiliation_or_header', 0)} affiliations or headers, "
        f"{author_counts.get('duplicate_in_paper', 0)} repeated in the same paper."
    )
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
            import_records(connection, papers, citations, authors, links)
    except ValueError as error:
        parser.exit(1, f"Fixture import refused: {error}\n")
    except psycopg.Error as error:
        parser.exit(
            1,
            f"Database import failed: {database_error_detail(error, dsn)}; transaction rolled back.\n",
        )
    elapsed = perf_counter() - started
    print(
        f"Loaded {len(papers)} papers, {len(citations)} citations, {len(authors)} authors "
        f"and {len(links)} paper links in {elapsed:.2f}s."
    )
    print("D-07 runtime target: " + ("under 60s." if elapsed < 60 else "not met (60s or longer)."))
    print("Chunks and embeddings are not populated; hybrid search is not ready yet.")


if __name__ == "__main__":
    main()
