"""Fill papers.embedding and chunks.embedding with the EMBED_MODEL vectors (D-14).

Usage: uv run --locked --group data --group rag --group embed python scripts/embed_database.py
Before the first batch, schema_metadata.embed_fingerprint records the model, revision
and inputs. It is retained after embed_model is published, so resumed and completed
runs must match it before reusing vectors. Vectors of unknown origin are refused.
"""

import argparse
import os
from time import perf_counter

from load_fixture import database_error_detail
from select_fixture import data_directory

from scilitrag.config import DEFAULT_EMBED_MODEL
from scilitrag.data.embedding import (
    MODEL_REVISIONS,
    check_vectors,
    document_text,
    encode,
    fingerprint,
)

PAPERS_TODO = (
    "SELECT corpus_id, title, abstract FROM papers WHERE embedding IS NULL ORDER BY corpus_id"
)
CHUNKS_TODO = "SELECT chunk_id, text FROM chunks WHERE embedding IS NULL ORDER BY chunk_id"
PAPER_UPDATE = "UPDATE papers SET embedding = %s::vector WHERE corpus_id = %s"
CHUNK_UPDATE = "UPDATE chunks SET embedding = %s::vector WHERE chunk_id = %s"
REMAINING = (
    "SELECT (SELECT count(*) FROM papers WHERE embedding IS NULL)"
    " + (SELECT count(*) FROM chunks WHERE embedding IS NULL)"
)
EXISTING = (
    "SELECT (SELECT count(*) FROM papers WHERE embedding IS NOT NULL)"
    " + (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL)"
)
METADATA = "SELECT embed_model, embed_fingerprint FROM schema_metadata WHERE id = 1 FOR UPDATE"
SET_MODEL = "UPDATE schema_metadata SET embed_model = %s WHERE id = 1"
SET_RUN = "UPDATE schema_metadata SET embed_model = %s, embed_fingerprint = %s WHERE id = 1"
PENDING = "pending:"
RESET_HINT = (
    "reset with UPDATE papers SET embedding = NULL; UPDATE chunks SET embedding = NULL; "
    "UPDATE schema_metadata SET embed_model = NULL, embed_fingerprint = NULL WHERE id = 1;"
)


def vector_text(vector):
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"


def embed_rows(connection, model, embed, query, update, to_text, batch_size):
    """Embed the rows of one query batch by batch, committing after each batch."""
    rows = connection.execute(query).fetchall()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embed(model, [to_text(row) for row in batch])
        check_vectors(vectors, len(batch))
        with connection.cursor() as cursor:
            cursor.executemany(
                update, [(vector_text(v), row[0]) for v, row in zip(vectors, batch, strict=True)]
            )
        connection.commit()
    return len(rows)


def start_run(connection, model):
    """Verify provenance before reusing vectors, including those in a completed database."""
    expected = fingerprint(model)
    marker = PENDING + expected
    published, saved_fingerprint = connection.execute(METADATA).fetchone()
    if published is None:
        if saved_fingerprint is not None:
            raise ValueError(f"An embedding fingerprint exists without a model; {RESET_HINT}")
        existing = connection.execute(EXISTING).fetchone()[0]
        if existing:
            raise ValueError(f"{existing} vectors of unknown origin exist; {RESET_HINT}")
    elif saved_fingerprint != expected:
        raise ValueError(f"The embedding fingerprint is missing or does not match; {RESET_HINT}")
    elif published not in (marker, model):
        raise ValueError(f"The database holds vectors for {published}; {RESET_HINT}")
    connection.execute(SET_RUN, (marker, expected))
    connection.commit()


def publish_model(connection, model):
    """Publish the model only when every paper and chunk has a vector from this run."""
    expected = fingerprint(model)
    published, saved_fingerprint = connection.execute(METADATA).fetchone()
    if published != PENDING + expected or saved_fingerprint != expected:
        raise ValueError(f"Unexpected embedding state {published}; {RESET_HINT}")
    remaining = connection.execute(REMAINING).fetchone()[0]
    if remaining:
        raise ValueError(f"{remaining} rows still have no vector.")
    # Keep embed_fingerprint: future runs must verify it even after publication.
    connection.execute(SET_MODEL, (model,))
    connection.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    started = perf_counter()
    data_directory()  # loads .env like the other WP1 scripts
    model = os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL
    if model not in MODEL_REVISIONS:
        parser.exit(1, f"Embedding failed: no pinned revision for {model}.\n")
    dsn = os.environ.get("INGEST_DATABASE_URL")
    if not dsn:
        parser.error("Set INGEST_DATABASE_URL in .env; the app URL is never used for imports.")
    import psycopg

    try:
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            start_run(connection, model)
            papers = embed_rows(
                connection,
                model,
                encode,
                PAPERS_TODO,
                PAPER_UPDATE,
                lambda row: document_text(row[1], row[2]),
                args.batch_size,
            )
            print(f"Papers embedded: {papers} ({perf_counter() - started:.0f}s).")
            chunks = embed_rows(
                connection,
                model,
                encode,
                CHUNKS_TODO,
                CHUNK_UPDATE,
                lambda row: row[1],
                args.batch_size,
            )
            print(f"Chunks embedded: {chunks} ({perf_counter() - started:.0f}s).")
            publish_model(connection, model)
    except ValueError as error:
        parser.exit(1, f"Embedding refused: {error}\n")
    except psycopg.Error as error:
        parser.exit(1, f"Embedding failed: {database_error_detail(error, dsn)}\n")
    print(f"Published embed_model {model} (revision {MODEL_REVISIONS[model][:12]}).")
    print("Run ANALYZE papers, chunks; as administrator afterwards.")


if __name__ == "__main__":
    main()
