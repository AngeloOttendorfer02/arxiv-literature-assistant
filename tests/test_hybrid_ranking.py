import pytest

from scilitrag.contracts import Plan
from scilitrag.rag.search.backend import SearchBackend
from scilitrag.rag.search.ranking import reciprocal_rank_fusion


def test_rrf_rewards_consensus_without_raw_score_mixing():
    a = [
        {"corpus_id": 1, "score": 1000},
        {"corpus_id": 2, "score": 1},
        {"corpus_id": 2, "score": 1},
    ]
    b = [{"corpus_id": 3, "score": 0.9}, {"corpus_id": 2, "score": 0.8}]
    result = reciprocal_rank_fusion(a, b)
    assert [r["corpus_id"] for r in result] == [2, 1, 3]
    assert result[0]["score"] == pytest.approx(2 / 62)
    assert a[0]["score"] == 1000


def test_hybrid_trace_captures_source_counts_and_timings():
    def reader(sql, params):
        return [{"corpus_id": 1, "score": 0.9}]

    backend = SearchBackend(reader, lambda _: [[1, 0]], dimension=2)
    assert backend.hybrid(Plan(scenario="S1", topic="retrieval", k=3))[0]["corpus_id"] == 1
    trace = backend.last_trace.public_fields()
    assert trace["dense_candidates"] == trace["bm25_candidates"] == trace["returned"] == 1
    assert trace["rrf_k"] == 60 and trace["dense_ms"] >= 0 and trace["bm25_ms"] >= 0


def test_chunk_identity_can_be_used_without_collapsing_a_paper():
    rows = [{"chunk_id": "a", "corpus_id": 1}, {"chunk_id": "b", "corpus_id": 1}]
    assert len(reciprocal_rank_fusion(rows, key_field="chunk_id")) == 2
