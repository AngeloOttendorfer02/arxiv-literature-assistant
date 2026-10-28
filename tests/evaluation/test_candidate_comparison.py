import asyncio
import copy
import json

import pytest

from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.evaluation.runner import main, rescore_report, run_case

CASES = [
    {"id": "s1", "query": "retrieval papers", "scenario": "S1"},
    {"id": "s2", "query": "Which papers cite BERT?", "scenario": "S2"},
    {"id": "s3", "query": "Who is the most cited author for retrieval?", "scenario": "S3"},
]
SETTINGS = load_settings({"USE_MOCKS": "1", "LOG_DIR": "", "LLM_MODEL": "a"})


def test_summary_keeps_failed_run_latency_missing_candidates_and_unknown_quality():
    first = asyncio.run(run_case(CASES[0], mock_components(0), SETTINGS))
    first["elapsed_ms"] = 100
    failed = copy.deepcopy(first)
    failed.update(id="s1-failed", query="another retrieval query", elapsed_ms=1000)
    failed["events"] = [{"event": "error", "data": {"code": "timeout", "message": "timeout"}}]
    failed["sentence_supported"] = []
    bundle = {"cases": [*CASES, {"id": "s1-failed", "query": failed["query"], "scenario": "S1"}]}
    report = rescore_report(bundle, {"candidate_models": ["a", "b"], "runs": [first, failed]})
    summary = report["summary"]
    row = next(r for r in summary if r["model"] == "a" and r["scenario"] == "S1")
    assert row["passed"] == 1 and row["failed"] == 1 and row["expected_cases"] == 2
    assert row["elapsed_ms"] == {"n": 2, "mean": 550, "median": 550, "p95": 1000}
    assert row["semantic_citation_precision"] is None
    assert all(
        r["runs"] == 0 and r["elapsed_ms"]["mean"] is None for r in summary if r["model"] == "b"
    )
    assert len(report["missing_cases"]) == 6


@pytest.mark.parametrize("fail_fast", [True, False])
def test_misrouting_fails_cli_and_fail_fast_records_missing_cases(tmp_path, monkeypatch, fail_fast):
    cases = tmp_path / "cases.json"
    # First query completes as S3, but the expected scenario deliberately differs.
    cases.write_text(json.dumps({"cases": [{**CASES[2], "scenario": "S1"}, CASES[1]]}))
    output = tmp_path / "runs.json"
    monkeypatch.setattr("scilitrag.evaluation.runner.load_settings", lambda: SETTINGS)
    args = [str(cases), "--allow-mocks", "--models", "a", "b", "--output", str(output)]
    assert main(args + (["--fail-fast"] if fail_fast else [])) == 1
    report = json.loads(output.read_text())
    assert len(report["runs"]) == (1 if fail_fast else 4)
    assert report["runs"][0]["metrics"]["completed"]
    assert not report["runs"][0]["metrics"]["scenario_exact"]
    assert len(report["missing_cases"]) == (3 if fail_fast else 0)


def test_runner_preserves_existing_output_before_service_startup(tmp_path):
    output = tmp_path / "runs.json"
    output.write_text("keep")
    with pytest.raises(SystemExit):
        main(["unused.json", "--output", str(output)])
    assert output.read_text() == "keep"


def test_duplicate_candidate_case_runs_are_not_double_counted():
    record = asyncio.run(run_case(CASES[0], mock_components(0), SETTINGS))
    with pytest.raises(ValueError, match="once"):
        rescore_report({"cases": [CASES[0]]}, {"runs": [record, copy.deepcopy(record)]})
