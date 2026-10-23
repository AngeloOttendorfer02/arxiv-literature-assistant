import pytest

from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.entities import EntityLinker, normalize_name


def test_normalization_matches_shared_ingestion_rule():
    assert normalize_name(" ALICE  Smith ") == "alice smith"
    assert normalize_name("Alice-Smith") == "alice smith"


def test_paper_context_disambiguates_an_author():
    calls = []

    def reader(sql, params):
        calls.append((sql, params))
        if "FROM papers" in sql:
            return [{"corpus_id": 17, "title": "A"}]
        assert params["name"] == "alice smith" and params["paper_id"] == 17
        return [{"author_id": 3, "name": "Alice Smith"}]

    linker = EntityLinker(reader)
    assert linker.author("Alice Smith", paper_ref="17") == 3
    assert linker.author("Alice Smith", paper_ref="17") == 3 and len(calls) == 2


def test_namesake_authors_require_an_explicit_choice():
    rows = [{"author_id": 1, "name": "A Smith"}, {"author_id": 2, "name": "A Smith"}]
    with pytest.raises(RagUserError, match="author ID: 1"):
        EntityLinker(lambda *_: rows).author("A Smith")


def test_numeric_author_id_is_bound():
    def reader(sql, params):
        assert params["id"] == 42 and "42" not in sql
        return [{"author_id": 42, "name": "Alice"}]

    assert EntityLinker(reader).author("42") == 42


def test_missing_author_does_not_guess_from_partial_names():
    with pytest.raises(RagUserError, match="not found"):
        EntityLinker(lambda *_: []).author("Smith")
