from scilitrag.contracts import Plan
from scilitrag.rag.search_backend import SearchBackend


def test_real_dense_and_bm25_apply_year_bounds(fixture_reader):
    backend = SearchBackend(fixture_reader, lambda _: [[1.0] + [0.0] * 767], dimension=768)
    p = Plan(scenario="S1", topic="retrieval", year_from=2020, year_to=2023, k=3)
    assert {r["corpus_id"] for r in backend.dense(p)} == {2, 3, 5}
    assert {r["corpus_id"] for r in backend.bm25(p)} == {2, 3, 5}
