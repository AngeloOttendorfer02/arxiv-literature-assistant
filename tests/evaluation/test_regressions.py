"""Regression checks for invalid input, frozen references and non-answer outcomes."""

import asyncio
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.evaluation.datasets import ReferenceSet
from scilitrag.evaluation.metrics import score_run
from scilitrag.evaluation.reference import build_reference_set
from scilitrag.evaluation.runner import dataset_revision, main, rescore_report, run_case
from scilitrag.rag.errors import RagUserError

CASE = {"id": "s1", "query": "retrieval papers", "scenario": "S1"}
SETTINGS = load_settings({"USE_MOCKS": "1", "LOG_DIR": ""})


def test_refusal_is_not_counted_as_a_completed_evaluation():
    case = {**CASE, "query": "Delete all papers"}
    record = asyncio.run(run_case(case, mock_components(0), SETTINGS))
    assert record["metrics"]["refused"]
    assert not record["metrics"]["completed"]


def test_empty_answer_is_not_counted_as_a_completed_evaluation():
    async def empty(*args):
        if False:
            yield "unreachable"

    components = replace(mock_components(0), generate=empty)
    record = asyncio.run(run_case(CASE, components, SETTINGS))
    assert record["events"][-1]["event"] == "done"
    assert not record["metrics"]["completed"]


@pytest.mark.parametrize("actual", [None, "different-revision"])
def test_author_counts_are_not_compared_without_matching_dataset_revision(actual):
    case = {"scenario": "S3", "seed_ids": [2], "expected_author_counts": {"10": 2}}
    record = {
        "plan": {"scenario": "S3"},
        "dataset_revision": actual,
        "evidence": [{"type": "paper", "label": 1, "corpus_id": 2, "text": "source"}],
        "events": [
            {
                "event": "table",
                "data": {"columns": ["author_id", "citation_count"], "rows": [[10, 2]]},
            },
            {"event": "done", "data": {"refused": False}},
        ],
    }
    scores = score_run(case, record, expected_revision="reference-revision")
    assert scores["author_scope_exact"]
    assert scores["author_count_exact"] is None
    record["dataset_revision"] = "reference-revision"
    assert score_run(case, record, expected_revision="reference-revision")["author_count_exact"]
    record["dataset_changed"] = True
    assert (
        score_run(case, record, expected_revision="reference-revision")["author_count_exact"]
        is None
    )


def test_missing_years_are_not_counted_as_correct_metadata():
    record = {
        "events": [],
        "evidence": [{"type": "paper", "label": 1, "corpus_id": 2, "text": "source", "year": None}],
    }
    assert score_run({"expected_years": {"2": None}}, record)["source_year_exact"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"id": " "},
        {"query": " "},
        {"scenario": "S4"},
        {"seed_ids": [1, 1]},
        {"seed_ids": [0]},
        {"seed_ids": [True]},
        {"expected_author_counts": {"1": 2}},
        {"unexpected": "field"},
        {"expected_years": {"paper": 2020}},
    ],
)
def test_invalid_cases_are_rejected(changes):
    with pytest.raises(ValidationError):
        ReferenceSet.model_validate({"cases": [{**CASE, **changes}]})


def test_duplicate_case_ids_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        ReferenceSet.model_validate({"cases": [CASE, CASE]})


@pytest.mark.parametrize("report", [{"runs": []}, {"runs": [{"id": "unknown", "events": []}]}])
def test_invalid_saved_reports_do_not_succeed_vacuously(report):
    with pytest.raises(ValueError):
        rescore_report({"cases": [CASE]}, report)


def test_mock_dataset_revision_never_opens_database(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mock evaluation attempted database access")

    monkeypatch.setattr("scilitrag.rag.search.database.DatabaseReader", forbidden)
    assert dataset_revision(SETTINGS) is None


def test_invalid_cli_input_fails_before_component_or_service_access(tmp_path, monkeypatch):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"cases": [CASE, CASE]}))

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid cases reached component startup")

    monkeypatch.setattr("scilitrag.evaluation.runner.get_components", forbidden)
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        main([str(path), "--allow-mocks", "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_offline_rescore_never_loads_runtime_settings(tmp_path, monkeypatch):
    bundle = tmp_path / "cases.json"
    bundle.write_text(json.dumps({"cases": [CASE]}))
    run = asyncio.run(run_case(CASE, mock_components(0), SETTINGS))
    report = tmp_path / "run.json"
    report.write_text(json.dumps({"runs": [run]}))

    def forbidden():
        raise AssertionError("Offline scoring loaded services or configuration")

    monkeypatch.setattr("scilitrag.evaluation.runner.load_settings", forbidden)
    output = tmp_path / "reviewed.json"
    assert main([str(bundle), "--score-only", str(report), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["runs"][0]["metrics"]["completed"]


def test_unknown_export_papers_are_rejected_before_reference_queries():
    def reader(sql, params=None):
        assert sql == "SELECT corpus_id FROM papers WHERE corpus_id=ANY(%(ids)s)"
        return []

    with pytest.raises(RagUserError, match="unknown paper IDs"):
        build_reference_set(reader, [1], [], "retrieval")


def test_export_requires_frozen_revision():
    def reader(sql, params=None):
        if sql.startswith("SELECT corpus_id"):
            return [{"corpus_id": 1}]
        return [{"dataset_revision": None}]

    with pytest.raises(RagUserError, match="frozen dataset revision"):
        build_reference_set(reader, [1], [], "retrieval")


def test_reference_export_deduplicates_targets_and_counts_seed_pairs_once():
    calls = []

    def reader(sql, params=None):
        calls.append((sql, params))
        if sql.startswith("SELECT corpus_id"):
            return [{"corpus_id": 1}, {"corpus_id": 2}]
        if sql.startswith("SELECT dataset_revision"):
            return [{"dataset_revision": "fixture-v1"}]
        if sql.startswith("SELECT p.corpus_id"):
            return [{"corpus_id": 2, "year": 2021}]
        if sql.startswith("SELECT c.chunk_id"):
            return [
                {
                    "corpus_id": 2,
                    "chunk_id": "2-0",
                    "section": "Experiments",
                    "text": "We use BERT as a baseline.",
                    "target_title": "BERT",
                }
            ]
        assert params["seeds"] == [2]
        return [{"citing_id": 2, "cited_id": 1, "author_id": 10, "name": "Alice"}]

    bundle = build_reference_set(reader, [1, 1], [2, 2], "retrieval")
    assert len(bundle["cases"]) == 2
    assert bundle["cases"][1]["expected_author_counts"] == {"10": 1}
    assert bundle["baseline_annotations"][0]["baseline_use"] is None
    assert sum(sql.startswith("SELECT p.corpus_id") for sql, _ in calls) == 1


def test_saved_report_requires_json_object():
    with pytest.raises(TypeError, match="JSON object"):
        rescore_report({"cases": [CASE]}, [])


def test_reference_export_rejects_revision_changes():
    metadata_calls = []

    def reader(sql, params=None):
        if sql.startswith("SELECT corpus_id"):
            return [{"corpus_id": 1}]
        if sql.startswith("SELECT dataset_revision"):
            metadata_calls.append(True)
            return [{"dataset_revision": "v1" if len(metadata_calls) == 1 else "v2"}]
        return []

    with pytest.raises(RagUserError, match="changed during reference export"):
        build_reference_set(reader, [1], [], "retrieval")


def test_duplicate_evidence_labels_are_rejected_when_scoring_saved_runs():
    item = {"type": "paper", "label": 1, "corpus_id": 2, "text": "source"}
    with pytest.raises(ValueError, match="unique positive"):
        score_run(CASE, {"events": [], "evidence": [item, item]})


def test_structurally_invalid_citation_cannot_receive_semantic_credit():
    record = {
        "events": [{"event": "sentence", "data": {"text": "A claim [99]."}}],
        "evidence": [],
        "sentence_supported": [True],
    }
    assert score_run(CASE, record)["semantic_citation_precision"] == 0


def test_export_rejects_boolean_ids_before_deduplication():
    def forbidden(*args):
        raise AssertionError("Invalid IDs reached the database")

    with pytest.raises(ValueError, match="positive database IDs"):
        build_reference_set(forbidden, [1, True], [], "retrieval")


def test_runner_invalidates_batch_when_dataset_revision_changes(tmp_path, monkeypatch):
    from scilitrag.evaluation.runner import CORE

    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"dataset_revision": "v1", "cases": [CASE]}))
    revisions = iter(["v1", "v2"])
    selected_settings = replace(SETTINGS, real_components=frozenset(CORE))
    monkeypatch.setattr("scilitrag.evaluation.runner.load_settings", lambda: selected_settings)
    monkeypatch.setattr("scilitrag.evaluation.runner.dataset_revision", lambda _: next(revisions))
    monkeypatch.setattr(
        "scilitrag.evaluation.runner.get_components", lambda *_args, **_kwargs: mock_components(0)
    )
    output = tmp_path / "changed.json"
    assert main([str(path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["runs"][0]["dataset_changed"]
    assert report["runs"][0]["metrics"]["dataset_revision_exact"] is False
    assert report["development_only"] and report["sentence_validator_mocked"]
