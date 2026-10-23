from scilitrag.evaluation.metrics import score_baselines, score_run


def test_author_counts_are_not_compared_across_different_seed_scopes():
    record = {"events": [], "plan": {"scenario": "S3"}, "evidence": []}
    case = {"scenario": "S3", "seed_ids": [2], "expected_author_counts": {"10": 2}}
    scores = score_run(case, record)
    assert not scores["author_scope_exact"]
    assert scores["author_count_exact"] is None


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
