from scilitrag.evaluation.reference import build_reference_set


def test_golden_export_leaves_human_baseline_labels_empty(fixture_reader):
    bundle = build_reference_set(fixture_reader, [1], [2, 5], "retrieval")
    assert bundle["cases"][0]["expected_citing_ids"] == [5, 3, 2]
    assert bundle["cases"][1]["expected_author_counts"] == {"20": 4, "10": 2, "30": 2}
    assert len(bundle["baseline_annotations"]) == 2
    assert all(item["baseline_use"] is None for item in bundle["baseline_annotations"])
