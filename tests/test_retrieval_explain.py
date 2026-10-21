from explain_retrieval import explain_with

from scilitrag.contracts import Plan
from scilitrag.rag.search_backend import SearchBackend


def test_explain_uses_real_templates_and_parameterized_filters():
    calls = []

    def reader(statement, params):
        assert statement.startswith("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT")
        assert "private topic" not in statement
        assert params["low"] == 2020 and params["topic"] == "private topic"
        calls.append(statement)
        return [{"QUERY PLAN": [{"Plan": {"Node Type": "test"}}]}]

    result = explain_with(
        Plan(scenario="S1", topic="private topic", year_from=2020, k=2),
        SearchBackend(reader, lambda _: [[1.0, 0.0]], dimension=2),
    )
    assert set(result) == {"dense", "bm25"} and len(calls) == 2
