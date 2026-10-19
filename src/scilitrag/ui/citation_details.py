"""Build citation drawer data from shared contracts without importing RAG code."""

from typing import Any

from scilitrag.contracts import Citation, SqlFacts


def citation_details(citation: Citation, tables: list[SqlFacts]) -> dict[str, Any]:
    """Join a citation to its SQL result, when present, using the exact query ID."""
    table = (
        next((item for item in tables if item.query_id == citation.query_id), None)
        if citation.query_id
        else None
    )
    return {
        "kind": "sql" if citation.query_id else "chunk" if citation.chunk_id else "paper",
        "title": citation.title or "Untitled",
        "year": citation.year,
        "corpus_id": citation.corpus_id,
        "chunk_id": citation.chunk_id,
        "author_ids": list(citation.author_ids),
        "section": citation.section,
        "text": citation.text,
        "source_hash": citation.source_hash,
        "query_id": citation.query_id,
        "sql": table.sql if table else None,
        "parameters": table.parameters if table else {},
        "scope": table.scope if table else None,
        "columns": table.columns if table else [],
        "rows": table.rows if table else [],
        "dataset_revision": table.dataset_revision if table else None,
    }
