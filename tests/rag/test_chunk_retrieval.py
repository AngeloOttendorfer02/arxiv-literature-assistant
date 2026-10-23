import pytest

from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.backend import SearchBackend
from scilitrag.rag.search.passages import select_passages


def test_two_queries_cover_multiple_papers_with_scoped_results():
    calls = []
    rows = [
        {"chunk_id": "1-a", "corpus_id": 1, "section": "Method", "text": "Method evidence"},
        {"chunk_id": "2-a", "corpus_id": 2, "section": "Results", "text": "Experimental results"},
    ]

    def reader(sql, params):
        calls.append((sql, params))
        return rows

    backend = SearchBackend(reader, lambda _: [[1, 0]], dimension=2)
    result = backend.chunks(Plan(scenario="S1", topic="retrieval", k=2), [1, 2, 1])
    assert len(calls) == 2 and len(result) == 2
    assert all(call[1]["paper_ids"] == [1, 2] for call in calls)
    assert "LATERAL" in calls[0][0] and "LATERAL" in calls[1][0]


def test_overlap_removal_preserves_other_sections():
    rows = [
        {"chunk_id": "a", "section": "Method", "text": "the same method words"},
        {"chunk_id": "b", "section": "Method", "text": "the same method words"},
        {"chunk_id": "c", "section": "Results", "text": "measured recall improves"},
    ]
    assert [r["chunk_id"] for r in select_passages(rows, limit=2)] == ["a", "c"]


def test_out_of_scope_chunks_are_rejected():
    backend = SearchBackend(
        lambda *_: [{"chunk_id": "bad", "corpus_id": 99, "text": "x"}],
        lambda _: [[1, 0]],
        dimension=2,
    )
    with pytest.raises(RagUserError, match="outside"):
        backend.chunks(Plan(scenario="S1", topic="x", k=1), [1])


def test_empty_scope_performs_no_embedding_or_database_work():
    def forbidden(*args):
        raise AssertionError("unexpected work")

    assert SearchBackend(forbidden, forbidden).chunks(Plan(scenario="S1", k=1), []) == []
