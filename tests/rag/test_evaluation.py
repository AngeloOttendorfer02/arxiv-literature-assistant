import asyncio

import pytest
from build_wp2_golden import build_goldens
from evaluate_wp2 import run_case

from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.rag.evaluation import score_baselines, score_run


def test_mock_run_is_structurally_valid_but_has_no_semantic_score():
    case = {"id": "s1", "scenario": "S1", "query": "retrieval papers"}
    record = asyncio.run(run_case(case, mock_components(0), load_settings({"USE_MOCKS": "1"})))
    assert record["metrics"]["completed"]
    assert record["metrics"]["structural_citation_validity"] == 1
    assert record["metrics"]["semantic_citation_precision"] is None
    record["sentence_supported"] = [True] * len(record["sentence_supported"])
    assert score_run(case, record)["semantic_citation_precision"] == 1
    record["sentence_supported"] = ["yes"] * len(record["sentence_supported"])
    with pytest.raises(ValueError, match="true, false or null"):
        score_run(case, record)


def test_author_counts_are_not_compared_across_different_seed_scopes():
    record = {"events": [], "plan": {"scenario": "S3"}, "evidence": []}
    case = {"scenario": "S3", "seed_ids": [2], "expected_author_counts": {"10": 2}}
    scores = score_run(case, record)
    assert not scores["author_scope_exact"]
    assert scores["author_count_exact"] is None


def test_golden_export_leaves_human_baseline_labels_empty(fixture_reader):
    bundle = build_goldens(fixture_reader, [1], [2, 5], "retrieval")
    assert bundle["cases"][0]["expected_citing_ids"] == [5, 3, 2]
    assert bundle["cases"][1]["expected_author_counts"] == {"20": 4, "10": 2, "30": 2}
    assert len(bundle["baseline_annotations"]) == 2
    assert all(item["baseline_use"] is None for item in bundle["baseline_annotations"])


def test_baseline_metrics_require_human_review():
    annotations = [
        {"text": "We use BERT as a baseline.", "target_title": "BERT", "baseline_use": None},
        {"text": "BERT is discussed.", "target_title": "BERT", "baseline_use": None},
    ]
    assert score_baselines(annotations)["precision"] is None
    annotations[0]["baseline_use"] = True
    annotations[1]["baseline_use"] = False
    scores = score_baselines(annotations)
    assert scores["precision"] == scores["recall"] == 1
    assert scores["reviewed"] == 2
