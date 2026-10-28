import importlib.util
from pathlib import Path

import pytest

from scilitrag.evaluation import validator_eval as ve


def case(cid, expected, entail=0.9, contra=0.0, number_ok=True, kind=None, split="tuning"):
    return {
        "id": cid,
        "split": split,
        "expected": expected,
        "manipulation": kind or ("faithful_paraphrase" if expected == "ok" else "negation"),
        "number_ok": number_ok,
        "nli": {"entailment": entail, "contradiction": contra, "neutral": 1 - entail - contra},
    }


def test_predict_status_follows_the_order_of_check_sentence():
    assert ve.predict_status(case("a", "ok", number_ok=False), 0.5, 0.5) == "fact_mismatch"
    assert ve.predict_status(case("a", "ok", entail=0.6), 0.5, 0.5) == "ok"
    assert ve.predict_status(case("a", "ok", entail=0.2, contra=0.7), 0.5, 0.5) == "fact_mismatch"
    assert ve.predict_status(case("a", "ok", entail=0.2, contra=0.1), 0.5, 0.5) == "unsupported"
    without_text = {**case("a", "ok"), "nli": None}
    assert ve.predict_status(without_text, 0.5, 0.5) == "ok"


def sample():
    return [
        case("f1", "ok", entail=0.95),
        case("f2", "ok", entail=0.3),
        case("m1", "removed", entail=0.1, contra=0.8),
        case("m2", "removed", entail=0.2, number_ok=False, kind="number"),
        case("m3", "removed", entail=0.7),
    ]


def test_score_cases_counts_recall_precision_and_false_removals():
    scored = ve.score_cases(sample(), 0.5, 0.5)
    assert scored["recall"]["hits"] == 2 and scored["recall"]["total"] == 3
    assert scored["precision"]["hits"] == 2 and scored["precision"]["total"] == 3
    assert scored["false_removal"]["hits"] == 1 and scored["false_removal"]["total"] == 2
    assert scored["removed_by_number_check"]["hits"] == 1
    assert scored["removed_by_kind"]["number"]["hits"] == 1
    assert scored["thresholds"] == {"entailment": 0.5, "contradiction": 0.5}


def test_raising_the_entailment_threshold_removes_more_of_everything():
    low = ve.score_cases(sample(), 0.2, 0.5)
    high = ve.score_cases(sample(), 0.9, 0.5)
    assert high["recall"]["hits"] >= low["recall"]["hits"]
    assert high["false_removal"]["hits"] >= low["false_removal"]["hits"]


def test_selection_keeps_faithful_sentences_within_the_cap_and_maximises_recall():
    faithful = [case(f"f{i}", "ok", entail=0.6 + 0.02 * i) for i in range(20)]
    manipulated = [case(f"m{i}", "removed", entail=0.05 * i) for i in range(10)]
    records = faithful + manipulated
    threshold = ve.select_entailment_threshold(records, 0.5, frr_cap=0.05)
    scored = ve.score_cases(records, threshold, 0.5)
    assert scored["false_removal"]["rate"] <= 0.05
    best = max(
        ve.score_cases(records, t, 0.5)["recall"]["rate"]
        for t in ve.ENTAILMENT_GRID
        if ve.score_cases(records, t, 0.5)["false_removal"]["rate"] <= 0.05
    )
    assert scored["recall"]["rate"] == best


def test_selection_fails_without_any_threshold_inside_the_cap():
    with pytest.raises(ValueError):
        ve.select_entailment_threshold([case("f", "ok", number_ok=False)], 0.5, frr_cap=0.0)


def test_report_renders_from_recorded_scores():
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_validator.py"
    spec = importlib.util.spec_from_file_location("evaluate_validator", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    records = []
    for split in ("tuning", "test"):
        records += [{**case(f"{split}-f{i}", "ok", entail=0.9, split=split)} for i in range(6)] + [
            {**case(f"{split}-m{i}", "removed", entail=0.1, split=split)} for i in range(6)
        ]
    raw = {
        "records": records,
        "meta": {
            "nli_model": "demo",
            "replay_mismatches": [],
            "defaults": {"entailment": 0.5, "contradiction": 0.5},
            "latency_ms": {"median": 1.0, "p95": 2.0},
        },
    }
    result = script.analyze(raw)
    text = script.report(raw, result)
    assert "# Sentence validator evaluation" in text and "recall" in text
    assert result["tuning_chosen"]["false_removal"]["rate"] <= 0.05
