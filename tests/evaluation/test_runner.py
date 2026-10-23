import asyncio

import pytest

from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.evaluation.metrics import score_run
from scilitrag.evaluation.runner import run_case


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
