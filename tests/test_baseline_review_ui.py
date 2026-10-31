import copy

import pytest

pytest.importorskip("streamlit")

from scilitrag.rag.evidence.provenance import digest
from scilitrag.ui.baseline_review import reviewed_reference


def test_review_preserves_sources_and_uncertainty():
    passage = {
        "chunk_id": "c1", "corpus_id": 3, "target_id": 2,
        "target_title": "Target", "section": "Results", "text": "A passage.",
    }
    passage["source_hash"] = digest(passage)
    source = {
        "cases": [{"id": "s2", "scenario": "S2", "query": "Who cites Target?"}],
        "baseline_annotations": [passage],
    }
    before = copy.deepcopy(source)
    result = reviewed_reference(source, [("Yes", "Comparison in results")])
    assert source == before
    assert result["baseline_annotations"][0]["baseline_use"] is True
    assert result["baseline_annotations"][0]["source_hash"] == passage["source_hash"]
    assert reviewed_reference(source, [("Uncertain", "")])["baseline_annotations"][0][
        "baseline_use"
    ] is None
    with pytest.raises(ValueError):
        reviewed_reference(source, [])
