import pytest

from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.readiness import (
    SEARCH_INDEX_FIELDS,
    SEARCH_INDEXES,
    validate_search_indexes,
)


def index_rows():
    return [
        {"index_name": name, "table_name": table, "columns": sorted(columns)}
        for name, (table, columns) in SEARCH_INDEX_FIELDS.items()
    ]


def test_missing_search_indexes_have_an_installation_hint():
    with pytest.raises(RagUserError, match="retrieval_indexes.sql"):
        validate_search_indexes(lambda *_: [])


def test_index_probe_requires_valid_ready_indexes_in_the_active_schema():
    def reader(statement, params):
        assert "i.indisvalid AND i.indisready" in statement
        assert "current_schema()" in statement
        assert params["names"] == sorted(SEARCH_INDEXES)
        assert "i.indpred IS NULL" in statement
        return index_rows()

    validate_search_indexes(reader)


@pytest.mark.parametrize("problem", ["missing_title", "wrong_table", "legacy_name"])
def test_stale_or_misplaced_paper_index_is_rejected(problem):
    rows = index_rows()
    paper = next(row for row in rows if row["table_name"] == "papers")
    if problem == "missing_title":
        paper["columns"].remove("title")
    elif problem == "wrong_table":
        paper["table_name"] = "chunks"
    else:
        paper["index_name"] = "papers_abstract_bm25_idx"
    with pytest.raises(RagUserError, match="002_bm25_title_abstract.sql"):
        validate_search_indexes(lambda *_: rows)
