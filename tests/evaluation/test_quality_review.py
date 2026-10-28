import asyncio
import copy
import csv
import json
from dataclasses import replace

import pytest

from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.contracts import EvidenceCheckResult
from scilitrag.evaluation.metrics import score_run
from scilitrag.evaluation.review import FIELDS, export_reviews, import_reviews
from scilitrag.evaluation.runner import main, rescore_report, run_case
from scilitrag.timings import record_timing

CASE = {"id": "s1", "query": "retrieval papers", "scenario": "S1"}
SETTINGS = load_settings({"USE_MOCKS": "1", "LOG_DIR": "", "LLM_MODEL": "candidate"})


def report():
    record = asyncio.run(run_case(CASE, mock_components(0), SETTINGS))
    return {"candidate_models": ["candidate"], "runs": [record]}


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def test_citation_precision_is_per_reference_and_fact_reviews_are_separate():
    run = {
        "evidence": [
            {"type": "paper", "label": i, "corpus_id": i, "text": "source"} for i in (1, 2)
        ],
        "events": [{"event": "sentence", "data": {"text": "A claim in 2021 [1][2]."}}],
        "sentence_supported": [True],
        "citation_supported": [{"1": True, "2": False}],
        "sentence_year_exact": [False],
        "sentence_count_exact": [None],
    }
    metrics = score_run(CASE, run)
    assert metrics["semantic_citation_precision"] == 1
    assert metrics["citation_precision"] == 0.5 and metrics["citation_review_coverage"] == 1
    assert metrics["answer_year_exact"] == 0 and metrics["answer_year_reviewed"] == 1
    assert metrics["answer_count_exact"] is None and metrics["answer_count_reviewed"] == 0


def test_missing_expected_years_do_not_disappear_from_denominator():
    run = {
        "evidence": [{"type": "paper", "label": 1, "corpus_id": 2, "text": "source"}],
        "events": [],
    }
    metrics = score_run({"expected_years": {"2": 2021, "3": 2022, "4": None}}, run)
    assert metrics["source_year_exact"] == 0 and metrics["source_year_compared"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("citation_supported", [{"99": True}]),
        ("sentence_year_exact", ["yes"]),
        ("sentence_count_exact", [True, False]),
        ("sentence_supported", ""),
    ],
)
def test_invalid_or_mismatched_review_labels_are_rejected(field, value):
    run = {
        "evidence": [{"type": "paper", "label": 1, "corpus_id": 1, "text": "source"}],
        "events": [{"event": "sentence", "data": {"text": "Claim [1]."}}],
        field: value,
    }
    with pytest.raises(ValueError):
        score_run(CASE, run)


def test_blank_csv_roundtrip_leaves_all_quality_unknown(tmp_path):
    original = report()
    path = tmp_path / "review.csv"
    export_reviews(original, path)
    reviewed = rescore_report({"cases": [CASE]}, import_reviews(original, path))
    metrics = reviewed["runs"][0]["metrics"]
    assert metrics["semantic_citation_precision"] is None and metrics["citation_precision"] is None
    assert metrics["answer_year_exact"] is None and metrics["answer_count_exact"] is None
    assert reviewed["summary"][0]["semantic_citation_precision"] is None
    assert "citation_supported" not in original["runs"][0]  # no in-place mutation


def test_review_import_updates_only_labels_and_rescores_without_services(tmp_path, monkeypatch):
    original = report()
    path = tmp_path / "review.csv"
    export_reviews(original, path)
    rows = read_rows(path)
    for row in rows:
        row["supported"] = "true" if row["kind"] == "sentence" else "false"
        if row["kind"] == "sentence":
            row["year_exact"] = "false"
        row["note"] = "review note"
    write_rows(path, list(reversed(rows)))  # worksheet sorting is allowed
    cases, source, output = (
        tmp_path / name for name in ("cases.json", "source.json", "reviewed.json")
    )
    cases.write_text(json.dumps({"cases": [CASE]}))
    source.write_text(json.dumps(original))

    def forbidden():
        raise AssertionError("Offline review loaded service configuration")

    monkeypatch.setattr("scilitrag.evaluation.runner.load_settings", forbidden)
    assert (
        main(
            [
                str(cases),
                "--score-only",
                str(source),
                "--import-review",
                str(path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    scores = json.loads(output.read_text())["runs"][0]["metrics"]
    assert scores["citation_precision"] == 0 and scores["semantic_citation_precision"] == 1
    assert scores["answer_year_exact"] == 0
    assert json.loads(source.read_text()) == original


@pytest.mark.parametrize(
    "change", ["sentence", "source_text", "duplicate", "missing", "label", "fact_on_citation"]
)
def test_stale_or_modified_worksheet_is_rejected(tmp_path, change):
    original = report()
    path = tmp_path / "review.csv"
    export_reviews(original, path)
    rows = read_rows(path)
    if change in ("sentence", "source_text"):
        rows[0][change] += " changed"
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "label":
        rows[0]["supported"] = "yes"
    else:
        next(r for r in rows if r["kind"] == "citation")["year_exact"] = "true"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        import_reviews(original, path)


def test_changed_run_cannot_reuse_old_reviews(tmp_path):
    original = report()
    path = tmp_path / "review.csv"
    export_reviews(original, path)
    changed = copy.deepcopy(original)
    changed["runs"][0]["evidence"][0]["text"] += " changed"
    with pytest.raises(ValueError, match="source report"):
        import_reviews(changed, path)
    with pytest.raises(FileExistsError):
        export_reviews(original, path)


def test_powershell_utf8_bom_files_can_be_rescored(tmp_path, monkeypatch):
    cases, source, output = (tmp_path / name for name in ("cases.json", "runs.json", "scored.json"))
    cases.write_text(json.dumps({"cases": [CASE]}), encoding="utf-8-sig")
    source.write_text(json.dumps(report()), encoding="utf-8-sig")

    def forbidden():
        raise AssertionError("Offline scoring loaded runtime settings")

    monkeypatch.setattr("scilitrag.evaluation.runner.load_settings", forbidden)
    assert main([str(cases), "--score-only", str(source), "--output", str(output)]) == 0


def test_run_capture_uses_screened_sources_and_component_timings():
    components = mock_components(0)

    def retrieve(plan):
        record_timing("paper_dense_ms", 12.5)
        return components.retrieve(plan)

    def screen(evidence):
        return EvidenceCheckResult(evidence=evidence[1:], removed=[evidence[0].label], latency_ms=1)

    run = asyncio.run(
        run_case(CASE, replace(components, retrieve=retrieve, check_evidence=screen), SETTINGS)
    )
    assert run["query"] == CASE["query"] and run["expected_scenario"] == "S1"
    assert run["elapsed_ms"] > 0 and run["component_ms"] == {"paper_dense_ms": 12.5}
    assert run["removed_evidence_labels"] == [1]
    assert all(e["label"] != 1 for e in run["evidence"])
