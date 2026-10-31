import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

import compare_retrieval as comparison
from compare_embeddings import split_queries

from scilitrag.config import load_settings
from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search.backend import BM25_SQL, DENSE_SQL, SearchBackend
from scilitrag.rag.search.contract import DatabaseMetadata


def query(id_, split="dev", relevant=None):
    return {
        "id": id_,
        "query": "private query text with year 2020",
        "query_set": "manual_acl",
        "split": split,
        "relevant": [1] if relevant is None else relevant,
    }


def test_evaluation_uses_app_sql_and_preserves_hybrid_top20_without_extra_embedding_calls(
    monkeypatch,
):
    calls, embedded = [], []
    fused = []
    fusion = comparison.reciprocal_rank_fusion

    def capture_fusion(*rankings, **kwargs):
        rows = fusion(*rankings, **kwargs)
        fused.extend(row["corpus_id"] for row in rows)
        return rows

    monkeypatch.setattr(comparison, "reciprocal_rank_fusion", capture_fusion)
    dense = [{"corpus_id": i} for i in range(1, 101)]
    sparse = [{"corpus_id": i} for i in range(100, 0, -1)]

    def reader(statement, params):
        assert statement in (DENSE_SQL, BM25_SQL)
        assert params["limit"] == 100
        assert params["low"] is params["high"] is None
        assert params["paper_id"] is params["author_id"] is None
        calls.append(statement)
        return dense if statement == DENSE_SQL else sparse

    def embed(texts):
        embedded.extend(texts)
        return [[1.0, 0.0]]

    backend = SearchBackend(reader, embed, dimension=2)
    result = comparison.evaluate([query("q000", relevant=[100])], backend)
    assert calls == [DENSE_SQL, BM25_SQL]
    assert embedded == [query("q000")["query"]]
    assert result["scores"]["dense"]["dev"]["mrr"] == pytest.approx(1 / 100)
    assert result["scores"]["dense"]["dev"]["recall@20"] == 0
    plan = Plan(scenario="S1", topic=query("q000")["query"], k=20)
    expected = [row["corpus_id"] for row in backend.hybrid(plan)]
    assert fused[:20] == expected
    assert len(fused) == 100
    assert result["scores"]["dense"]["test"] is None
    assert "private query text" not in json.dumps(result)


def test_split_averages_include_empty_rankings_and_multiple_targets():
    class Backend:
        def dense(self, plan, limit):
            return [] if plan.query == "miss" else [{"corpus_id": 1}]

        bm25 = dense

    queries = [query("q000", relevant=[1, 2]), query("q001"), query("q002", "test")]
    queries[1]["query"] = "miss"
    result = comparison.evaluate(queries, Backend())
    assert result["split_counts"] == {"dev": 2, "test": 1, "all": 3}
    for method in comparison.METHODS:
        assert result["scores"][method]["dev"]["recall@5"] == 0.25
        assert result["scores"][method]["test"]["recall@5"] == 1
        assert result["scores"][method]["all"]["recall@20"] == 0.5
        assert result["per_query"][1]["metrics"][method]["mrr"] == 0


def test_search_error_aborts_instead_of_silently_skipping_a_query():
    def fail(*args, **kwargs):
        raise RagUserError("database unavailable")

    with pytest.raises(RagUserError, match="q000"):
        comparison.evaluate([query("q000")], SimpleNamespace(dense=fail))


def corpus_backend(ids, missing_vector=False, revision=comparison.REVISION):
    return SimpleNamespace(
        metadata=DatabaseMetadata(1, revision, "test-model", 768),
        reader=lambda statement: [
            {"corpus_id": cid, "embedded": not missing_vector} for cid in ids
        ],
    )


def test_corpus_preflight_requires_the_full_revision_and_all_vectors_and_targets():
    manifest = {"retained_corpusids": [1, 2]}
    queries = [query("q000")]
    assert comparison.validate_corpus(corpus_backend([1, 2]), queries, manifest) == 2
    for backend, invalid_queries, message in [
        (corpus_backend([1]), queries, "full corpus"),
        (corpus_backend([1, 2], revision="other"), queries, "revision"),
        (corpus_backend([1, 2], missing_vector=True), queries, "embeddings"),
        (corpus_backend([1, 2]), [query("q000", relevant=[3])], "target"),
    ]:
        with pytest.raises(ValueError, match=message):
            comparison.validate_corpus(backend, invalid_queries, manifest)


@pytest.fixture
def query_data(tmp_path, monkeypatch):
    folder = tmp_path / "filtered" / "d03"
    folder.mkdir(parents=True)
    manifest = {
        "dataset": "princeton-nlp/LitSearch",
        "dataset_revision": comparison.REVISION,
        "retained_corpusids": [1, 2],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "query.arrow").write_bytes(b"fake arrow")
    sets = ["a"] * 151 + ["b"] * 150 + ["c"] * 149 + ["d"] * 147
    rows = [
        {"query": "private query text", "query_set": name, "corpusids": [1, 2]} for name in sets
    ]
    monkeypatch.setattr(comparison, "read_table", lambda _: SimpleNamespace(to_pylist=lambda: rows))
    return tmp_path, rows, manifest


def test_queries_reuse_d22_split_before_limiting(query_data):
    data_dir, rows, _ = query_data
    queries, provenance = comparison.load_queries(data_dir)
    expected = split_queries([row["query_set"] for row in rows], seed=42)
    assert len(queries) == 597
    assert [row["split"] for row in queries] == expected
    assert expected.count("dev") == 300 and expected.count("test") == 297
    assert queries[0]["id"] == "q000" and queries[-1]["id"] == "q596"
    assert len(provenance["query_sha256"]) == 64


def test_bad_query_inputs_cannot_produce_a_successful_benchmark(query_data):
    data_dir, rows, _ = query_data
    rows[0]["corpusids"] = []
    with pytest.raises(ValueError, match="row 0"):
        comparison.load_queries(data_dir)
    rows.pop()
    with pytest.raises(ValueError, match="597"):
        comparison.load_queries(data_dir)


def test_cli_writes_scores_and_report_without_private_text(query_data, tmp_path, monkeypatch):
    data_dir, _, _ = query_data
    output, report = tmp_path / "scores.json", tmp_path / "report.md"
    backend = corpus_backend([1, 2])
    backend.dense = backend.bm25 = lambda *args, **kwargs: []
    settings = load_settings({"USE_MOCKS": "1", "DATA_DIR": str(data_dir)})
    monkeypatch.setattr(comparison, "load_settings", lambda: settings)

    def build_backend(_, *, statement_timeout_s):
        assert statement_timeout_s == 60
        return backend

    monkeypatch.setattr(comparison, "default_backend", build_backend)
    monkeypatch.setattr(
        "sys.argv", ["compare_retrieval.py", "--output", str(output), "--report", str(report)]
    )
    assert comparison.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert not result["smoke_run"] and result["queries"] == 597
    assert result["split_counts"] == {"dev": 300, "test": 297, "all": 597}
    assert result["metric_cutoffs"]["mrr"] == 100
    assert result["settings"]["title_boost"] == 0.5
    assert "pdb.boost(0.5)" in BM25_SQL
    assert result["settings"]["statement_timeout_s"] == 60
    assert result["bootstrap"]["draws"] == 10000 and result["bootstrap"]["seed"] == 42
    assert all(set(row) == {"id", "query_set", "split", "metrics"} for row in result["per_query"])
    assert '"relevant"' not in output.read_text(encoding="utf-8")
    assert '"rankings"' not in output.read_text(encoding="utf-8")
    assert "private query text" not in output.read_text(encoding="utf-8")
    assert "MRR@100" in report.read_text(encoding="utf-8")


def test_smoke_run_keeps_original_split_and_uses_separate_outputs(
    query_data, tmp_path, monkeypatch
):
    data_dir, _, _ = query_data
    backend = corpus_backend([1, 2])
    backend.dense = backend.bm25 = lambda *args, **kwargs: []
    monkeypatch.setattr(comparison, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        comparison,
        "load_settings",
        lambda: load_settings({"USE_MOCKS": "1", "DATA_DIR": str(data_dir)}),
    )
    monkeypatch.setattr(comparison, "default_backend", lambda _, **kwargs: backend)
    monkeypatch.setattr("sys.argv", ["compare_retrieval.py", "--limit", "1"])
    queries, _ = comparison.load_queries(data_dir)
    assert comparison.main() == 0
    result = json.loads((tmp_path / "eval/private/d18_retrieval_smoke.json").read_text())
    assert result["smoke_run"] and result["queries"] == 1
    assert result["per_query"][0]["split"] == queries[0]["split"]
    assert result["bootstrap"] is None


def test_bootstrap_resamples_paired_differences_within_each_split():
    rows = []
    for part in ("dev", "test"):
        difference = 0.1 if part == "dev" else -0.1
        for value in (0.1, 0.4, 0.7):
            rows.append(
                {
                    "split": part,
                    "metrics": {
                        "dense": {metric: value for metric in comparison.COMPARISON_METRICS},
                        "hybrid": {
                            metric: value + difference for metric in comparison.COMPARISON_METRICS
                        },
                    },
                }
            )
    result = {"smoke_run": False, "per_query": rows}
    comparison.add_bootstrap(result)
    for part, metrics in result["bootstrap"]["intervals"].items():
        difference = 0.1 if part == "dev" else -0.1
        for estimate in metrics.values():
            assert estimate == pytest.approx(
                {"delta": difference, "ci_low": difference, "ci_high": difference}
            )


@pytest.fixture
def saved_result(query_data):
    data_dir, _, _ = query_data
    queries, _ = comparison.load_queries(data_dir)
    backend = SimpleNamespace(dense=lambda *args, **kwargs: [], bm25=lambda *args, **kwargs: [])
    saved = comparison.evaluate(queries, backend)
    saved.update(
        smoke_run=False,
        split_seed=42,
        documents=2,
        database={"embed_model": "test-model"},
        settings={"statement_timeout_s": 60},
    )
    return saved


def test_score_only_uses_saved_metrics_without_database_or_model(
    saved_result, tmp_path, monkeypatch
):
    saved = saved_result
    source = tmp_path / "saved.json"
    source.write_text(json.dumps(saved), encoding="utf-8")
    output, report = tmp_path / "rescored.json", tmp_path / "rescored.md"

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline statistics must not load services or dataset text")

    monkeypatch.setattr(comparison, "load_settings", forbidden)
    monkeypatch.setattr(comparison, "default_backend", forbidden)
    monkeypatch.setattr(comparison, "load_queries", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_retrieval.py",
            "--score-only",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    assert comparison.main() == 0
    updated = json.loads(output.read_text(encoding="utf-8"))
    assert updated["per_query"] == saved["per_query"] and updated["scores"] == saved["scores"]
    assert len(updated["bootstrap"]["intervals"]) == 2
    assert all(
        estimate["ci_low"] <= 0 <= estimate["ci_high"]
        for metrics in updated["bootstrap"]["intervals"].values()
        for estimate in metrics.values()
    )
    text = report.read_text(encoding="utf-8")
    assert "All six intervals include zero" in text
    assert "does not establish equivalence" in text


@pytest.mark.parametrize(
    "problem",
    [
        "truncated",
        "duplicate_id",
        "wrong_split",
        "wrong_counts",
        "wrong_average",
        "nan",
        "string_metric",
    ],
)
def test_score_only_rejects_inconsistent_results_before_writing(
    problem, saved_result, tmp_path, monkeypatch
):
    saved = copy.deepcopy(saved_result)
    if problem == "truncated":
        saved["per_query"] = [
            next(row for row in saved["per_query"] if row["split"] == part)
            for part in ("dev", "test")
        ]
    elif problem == "duplicate_id":
        saved["per_query"][1]["id"] = saved["per_query"][0]["id"]
    elif problem == "wrong_split":
        saved["per_query"][0]["split"] = (
            "test" if saved["per_query"][0]["split"] == "dev" else "dev"
        )
        saved.update(comparison.summarize(saved["per_query"]))
    elif problem == "wrong_counts":
        saved["split_counts"]["dev"] -= 1
    elif problem == "wrong_average":
        saved["scores"]["dense"]["all"]["recall@20"] = 0.5
    else:
        saved["per_query"][0]["metrics"]["dense"]["mrr"] = (
            float("nan") if problem == "nan" else "0.5"
        )
    source, report = tmp_path / "saved.json", tmp_path / "report.md"
    content = json.dumps(saved)
    source.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["compare_retrieval.py", "--score-only", str(source), "--report", str(report)]
    )
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 1
    assert source.read_text(encoding="utf-8") == content
    assert not report.exists()


def test_score_only_accepts_smoke_results_with_original_split(saved_result, tmp_path, monkeypatch):
    saved = copy.deepcopy(saved_result)
    saved.update(comparison.summarize(saved["per_query"][:5]))
    saved["smoke_run"] = True
    source = tmp_path / "smoke.json"
    source.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["compare_retrieval.py", "--score-only", str(source)])
    assert comparison.main() == 0
    updated = json.loads(source.read_text(encoding="utf-8"))
    assert updated["queries"] == 5 and updated["bootstrap"] is None
    assert updated["per_query"] == saved["per_query"]
    assert "Status: smoke run" in source.with_suffix(".md").read_text(encoding="utf-8")
