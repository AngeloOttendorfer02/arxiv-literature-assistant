import math

import pytest
from compare_embeddings import (
    document_text,
    paired_bootstrap,
    query_metrics,
    select_docs,
    split_queries,
)


def test_query_metrics_single_target():
    scores = query_metrics([9, 8, 1, 7], [1])
    assert scores["recall@5"] == 1.0
    assert scores["mrr"] == pytest.approx(1 / 3)
    assert scores["ndcg@10"] == pytest.approx(1 / math.log2(4))


def test_query_metrics_multiple_targets_and_miss():
    scores = query_metrics([1, 5, 2], [1, 2, 3])
    assert scores["recall@20"] == pytest.approx(2 / 3)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert scores["ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)
    assert query_metrics([4, 5], [1])["mrr"] == 0.0


def test_split_is_stratified_and_deterministic():
    sets = ["a"] * 5 + ["b"] * 4
    split = split_queries(sets)
    assert split == split_queries(sets)
    assert [split[i] for i in range(5)].count("dev") == 3
    assert [split[i] for i in range(5, 9)].count("dev") == 2


def test_paired_bootstrap_constant_difference():
    result = paired_bootstrap([0.5, 0.7, 0.9], [0.4, 0.6, 0.8], draws=200)
    assert result["delta"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)


def test_select_docs_keeps_targets():
    ids = list(range(100))
    picked = select_docs(ids, {3, 97}, 10)
    assert len(picked) == 10 and 3 in picked and 97 in picked
    assert select_docs(ids, {3}, 0) == ids


def test_document_text_handles_empty_title():
    assert document_text("", " abstract ") == "abstract"
    assert document_text("Title", "") == "Title"
    assert document_text("Title", "Abs") == "Title\nAbs"
