from scilitrag.contracts import Plan
from scilitrag.rag.search.backend import SearchBackend


def test_dense_and_bm25_bind_identical_filters():
    calls = []

    def reader(sql, params):
        calls.append((sql, params))
        return []

    backend = SearchBackend(reader, lambda _: [[1, 0]], dimension=2)
    plan = Plan(scenario="S1", topic="rare ' OR true --", year_from=2020, year_to=2023, k=3)
    backend.dense(plan)
    backend.bm25(plan)
    assert len(calls) == 2
    for sql, params in calls:
        assert params["low"] == 2020 and params["high"] == 2023
        assert params["topic"] == plan.topic and plan.topic not in sql
        assert "p.year >= %(low)s" in sql and "p.year <= %(high)s" in sql
    assert "<=>" in calls[0][0] and "|||" in calls[1][0]


def test_linked_paper_and_author_restrict_both_sources():
    def reader(sql, params):
        if "FROM papers WHERE" in sql:
            return [{"corpus_id": 7, "title": "A"}]
        if "FROM authors" in sql:
            assert params["paper_id"] == 7
            return [{"author_id": 9, "name": "Alice"}]
        assert params["paper_id"] == 7 and params["author_id"] == 9
        return []

    backend = SearchBackend(reader, lambda _: [[1, 0]], dimension=2)
    p = Plan(scenario="S1", topic="A", paper_ref="A", author_ref="Alice", k=1)
    backend.dense(p)
    backend.bm25(p)
