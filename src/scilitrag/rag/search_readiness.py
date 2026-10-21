"""Check the explicit BM25 index deployment before real search starts."""

from scilitrag.rag.errors import RagUserError

SEARCH_INDEXES = {"papers_abstract_bm25_idx", "chunks_text_bm25_idx"}
INDEX_SQL = """
    SELECT index_relation.relname AS index_name
    FROM pg_index i
    JOIN pg_class index_relation ON index_relation.oid=i.indexrelid
    JOIN pg_class source_relation ON source_relation.oid=i.indrelid
    JOIN pg_namespace n ON n.oid=source_relation.relnamespace
    JOIN pg_am am ON am.oid=index_relation.relam
    WHERE n.nspname=current_schema() AND i.indisvalid AND i.indisready
      AND am.amname IN ('bm25','paradedb')
      AND index_relation.relname=ANY(%(names)s)
"""


def validate_search_indexes(reader, *, required: set[str] | None = None) -> None:
    expected = SEARCH_INDEXES if required is None else required
    rows = reader(INDEX_SQL, {"names": sorted(expected)})
    missing = expected - {row["index_name"] for row in rows}
    if missing:
        raise RagUserError(
            "Search indexes are not ready: "
            + ", ".join(sorted(missing))
            + ". Apply docker/db/retrieval_indexes.sql after loading data."
        )
