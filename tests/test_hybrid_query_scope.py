from scilitrag.contracts import Plan
from scilitrag.rag.search_backend import SearchBackend


def test_hybrid_embeds_once_and_binds_one_shared_filter_set():
    embeddings, calls = [], []

    def embed(texts):
        embeddings.append(texts)
        return [[1.0, 0.0]]

    def reader(statement, parameters):
        calls.append((statement, dict(parameters)))
        return [{"corpus_id": 1, "score": 1.0}]

    backend = SearchBackend(reader, embed, dimension=2)
    backend.hybrid(Plan(scenario="S1", topic="retrieval", year_from=2020, k=2))
    assert embeddings == [["retrieval"]]
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    assert calls[0][1]["low"] == 2020


def test_s3_context_does_not_filter_citing_papers_by_credited_author():
    backend = SearchBackend(lambda *_: [], lambda _: [[1.0, 0.0]], dimension=2)
    params = backend.params(Plan(scenario="S3", topic="retrieval", author_ref="123", k=2), 50)
    assert params["author_id"] is None and params["paper_id"] is None
