"""Validate the shared relational schema independently of corpus readiness."""

import re
from dataclasses import dataclass

from scilitrag.rag.errors import RagUserError

REQUIRED_COLUMNS = {
    "papers": {"corpus_id", "title", "abstract", "year", "embedding"},
    "authors": {"author_id", "name", "normalized_name"},
    "paper_authors": {"corpus_id", "author_id"},
    "citations": {"citing_id", "cited_id"},
    "chunks": {"chunk_id", "corpus_id", "section", "text", "seq_no", "embedding"},
    "schema_metadata": {"id", "schema_version", "dataset_revision", "embed_model"},
}
VECTOR_TYPES_SQL = """
    SELECT c.relname AS table_name, format_type(a.atttypid,a.atttypmod) AS vector_type
    FROM pg_attribute a
    JOIN pg_class c ON c.oid=a.attrelid
    JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname=current_schema() AND c.relname=ANY(%(tables)s)
      AND a.attname='embedding' AND NOT a.attisdropped
"""


@dataclass(frozen=True)
class DatabaseMetadata:
    schema_version: int
    dataset_revision: str | None
    embed_model: str | None
    embed_dim: int


def validate_database(
    reader, model: str | None = None, *, check_columns: bool = True, require_search: bool = True
) -> DatabaseMetadata:
    if check_columns:
        columns = reader(
            "SELECT table_name,column_name FROM information_schema.columns "
            "WHERE table_schema=current_schema() AND table_name=ANY(%(tables)s)",
            {"tables": list(REQUIRED_COLUMNS)},
        )
        actual = {name: set() for name in REQUIRED_COLUMNS}
        for column in columns:
            if column["table_name"] in actual:
                actual[column["table_name"]].add(column["column_name"])
        missing = [
            f"{name}.{column}"
            for name, required in REQUIRED_COLUMNS.items()
            for column in sorted(required - actual[name])
        ]
        if missing:
            raise RagUserError("Database interface is missing: " + ", ".join(missing))
        extensions = reader(
            "SELECT extname FROM pg_extension WHERE extname=ANY(%(names)s)",
            {"names": ["vector", "pg_search", "pg_trgm"]},
        )
        if {row["extname"] for row in extensions} != {"vector", "pg_search", "pg_trgm"}:
            raise RagUserError("The database needs vector, pg_search and pg_trgm.")
    types = reader(VECTOR_TYPES_SQL, {"tables": ["papers", "chunks"]})
    dimensions = {}
    for row in types:
        match = re.fullmatch(r"vector\((\d+)\)", row["vector_type"])
        if match:
            dimensions[row["table_name"]] = int(match[1])
    # Dimension is declared by the shared schema; no duplicate metadata column is needed.
    if dimensions != {"papers": 768, "chunks": 768}:
        raise RagUserError("Both papers.embedding and chunks.embedding must be vector(768).")
    rows = reader(
        "SELECT schema_version,dataset_revision,embed_model FROM schema_metadata WHERE id=1"
    )
    if len(rows) != 1 or rows[0]["schema_version"] != 1:
        raise RagUserError("The database needs schema_metadata id=1 with schema_version=1.")
    row = rows[0]
    if require_search:
        if not row["dataset_revision"] or not row["embed_model"]:
            raise RagUserError(
                "Search is not ready: load corpus embeddings and publish dataset_revision and embed_model."
            )
        if model and row["embed_model"] != model:
            raise RagUserError("EMBED_MODEL does not match the database model.")
    return DatabaseMetadata(1, row["dataset_revision"], row["embed_model"], 768)
