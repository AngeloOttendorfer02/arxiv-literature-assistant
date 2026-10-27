"""Check the explicit BM25 index deployment before real search starts."""

from scilitrag.rag.errors import RagUserError

SEARCH_INDEX_FIELDS = {
    "papers_title_abstract_bm25_idx": ("papers", {"corpus_id", "title", "abstract", "year"}),
    "chunks_text_bm25_idx": ("chunks", {"chunk_id", "text", "corpus_id"}),
}
SEARCH_INDEXES = set(SEARCH_INDEX_FIELDS)
INDEX_SQL = """
    SELECT index_relation.relname AS index_name, source_relation.relname AS table_name,
           ARRAY(SELECT a.attname FROM unnest(i.indkey) AS k(attnum)
                 JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
                 WHERE k.attnum > 0) AS columns
    FROM pg_index i
    JOIN pg_class index_relation ON index_relation.oid=i.indexrelid
    JOIN pg_class source_relation ON source_relation.oid=i.indrelid
    JOIN pg_namespace n ON n.oid=source_relation.relnamespace
    JOIN pg_am am ON am.oid=index_relation.relam
    WHERE n.nspname=current_schema() AND i.indisvalid AND i.indisready
      AND i.indpred IS NULL
      AND am.amname IN ('bm25','paradedb')
      AND index_relation.relname=ANY(%(names)s)
"""


def validate_search_indexes(reader, *, required: set[str] | None = None) -> None:
    expected = SEARCH_INDEXES if required is None else required
    rows = reader(INDEX_SQL, {"names": sorted(expected)})
    ready = set()
    for row in rows:
        name = row["index_name"]
        definition = SEARCH_INDEX_FIELDS.get(name)
        if definition is None or (
            row["table_name"] == definition[0] and definition[1] <= set(row["columns"])
        ):
            ready.add(name)
    missing = expected - ready
    if missing:
        raise RagUserError(
            "Search indexes are not ready: "
            + ", ".join(sorted(missing))
            + ". Apply docker/db/retrieval_indexes.sql after loading data. "
            "For an existing abstract-only index, first apply "
            "docker/db/migrations/002_bm25_title_abstract.sql."
        )
