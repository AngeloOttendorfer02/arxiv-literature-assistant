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


def test_baseline_scoring_shares_reviewed_aliases_and_excludes_outperforms():
    common = {
        "target_id": 215737187,
        "target_title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "section": "Experiments",
    }
    annotations = [
        {**common, "text": "We use DPR as its pretrained baseline.", "baseline_use": True},
        {
            **common,
            "text": "Karpukhin et al. (2020) serves as our baseline.",
            "baseline_use": True,
        },
        {
            **common,
            "text": "DPR was used as a baseline in our experiments.",
            "baseline_use": True,
        },
        {
            **common,
            "text": "DPR was used as a baseline in earlier work.",
            "baseline_use": False,
        },
        {
            **common,
            "text": "In our experiments, DPR outperforms BM25.",
            "baseline_use": True,
        },
        {
            **common,
            "target_id": 999,
            "text": "We use DPR as a baseline.",
            "baseline_use": False,
        },
        {**common, "text": "We use DPR as a baseline.", "baseline_use": None},
    ]
    scores = score_baselines(annotations)
    assert scores["true_positive"] == 3 and scores["true_negative"] == 2
    # A relevant performance-only comparison remains a deliberate miss until evaluated.
    assert scores["false_positive"] == 0 and scores["false_negative"] == 1
    assert scores["reviewed"] == 6 and scores["total"] == 7
