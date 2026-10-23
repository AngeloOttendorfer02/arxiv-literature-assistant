from scilitrag.contracts import Plan
from scilitrag.rag.search.backend import SearchBackend


def test_real_chunk_queries_stay_within_selected_papers(fixture_reader):
    backend = SearchBackend(fixture_reader, lambda _: [[1.0] + [0.0] * 767], dimension=768)
    rows = backend.chunks(Plan(scenario="S1", topic="retrieval", k=2), [2])
    assert {row["corpus_id"] for row in rows} == {2}
    assert rows[0]["chunk_id"] == "2-0"
