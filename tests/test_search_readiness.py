import pytest

from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.readiness import SEARCH_INDEXES, validate_search_indexes


def test_missing_search_indexes_have_an_installation_hint():
    with pytest.raises(RagUserError, match="retrieval_indexes.sql"):
        validate_search_indexes(lambda *_: [])


def test_index_probe_requires_valid_ready_indexes_in_the_active_schema():
    def reader(statement, params):
        assert "i.indisvalid AND i.indisready" in statement
        assert "current_schema()" in statement
        assert params["names"] == sorted(SEARCH_INDEXES)
        return [{"index_name": name} for name in SEARCH_INDEXES]

    validate_search_indexes(reader)
