"""D-20 chunk positions, span hits, BM25 and the decision rule, with a whitespace tokenizer."""

import re

import numpy as np
import pytest
from chunk_experiment import (
    PRIMARY,
    Bm25,
    decide,
    located_chunks,
    papers_in_order,
    span_coverage,
    top,
)


def words(text):
    return [match.span() for match in re.finditer(r"\S+", text)]


def section(count, prefix="w"):
    return " ".join(f"{prefix}{index}" for index in range(count))


@pytest.mark.parametrize("overlap", [0, 10, 20])
def test_located_chunks_know_their_section_and_character_span(overlap):
    sections = [("1 A", section(300, "a")), ("2 B", section(600, "b"))]
    chunks = located_chunks(5, sections, words, 256, overlap)
    for chunk in chunks:
        text = sections[chunk["section_index"]][1]
        assert text[chunk["start"] : chunk["end"]] == chunk["text"]
        assert chunk["tokens"] == len(chunk["text"].split())
        assert chunk["section"] == sections[chunk["section_index"]][0]
    assert [c["section_index"] for c in chunks][:2] == [0, 0]
    assert max(c["tokens"] for c in chunks) == 256


def test_span_coverage_counts_only_the_same_section():
    chunk = {"corpus_id": 1, "section_index": 0, "start": 100, "end": 200}
    item = {"corpus_id": 1, "section_index": 0, "start": 150, "end": 250}
    assert span_coverage(chunk, item) == 0.5
    assert span_coverage(chunk, {**item, "start": 120, "end": 180}) == 1.0
    assert span_coverage(chunk, {**item, "start": 200, "end": 260}) == 0.0
    assert span_coverage(chunk, {**item, "section_index": 1}) == 0.0
    assert span_coverage(chunk, {**item, "corpus_id": 2}) == 0.0


def test_span_cut_by_a_chunk_border_still_hits_the_larger_part():
    text = section(600)
    chunks = located_chunks(1, [("1 A", text)], words, 256, 0)
    border = chunks[0]["end"]
    item = {"corpus_id": 1, "section_index": 0, "start": border - 30, "end": border + 10}
    assert [span_coverage(c, item) >= 0.5 for c in chunks] == [True, False, False]


def test_bm25_prefers_rare_matching_terms_and_ignores_case():
    bm25 = Bm25(["dense retrieval model", "BM25 sparse retrieval", "graph neural network"])
    scores = bm25.scores("Sparse RETRIEVAL")
    assert scores.argmax() == 1
    assert scores[0] > 0 and scores[2] == 0
    assert bm25.scores("absent")[:].sum() == 0


def test_top_breaks_ties_by_index_and_filters():
    scores = np.array([0.5, 0.9, 0.5, 0.0, 0.7], dtype=np.float32)
    assert top(scores, 3) == [1, 4, 0]
    assert top(scores, 5, among=[0, 2, 3]) == [0, 2, 3]
    assert top(scores, 5, among=[0, 3], positive=True) == [0]


def test_papers_in_order_keeps_the_first_chunk_of_each_paper():
    chunks = [{"corpus_id": 7}, {"corpus_id": 3}, {"corpus_id": 7}]
    assert papers_in_order([{"i": 2}, {"i": 0}, {"i": 1}], chunks) == [7, 3]


def config(size, truncated, primary, ci_high):
    return {
        "size": size,
        "truncated_share": truncated,
        "primary": dict.fromkeys(PRIMARY, primary),
        "vs_best": {m: {"ci_high": ci_high} for m in PRIMARY},
    }


def test_decision_prefers_fewer_truncations_then_smaller_size_among_equivalents():
    configs = {
        "512/10": config(512, 0.30, 0.60, 0.0),
        "384/10": config(384, 0.00, 0.58, 0.01),
        "256/10": config(256, 0.00, 0.50, 0.02),
        "256/0": config(256, 0.00, 0.40, -0.01),
    }
    assert decide(configs) == ("256/10", ["512/10", "384/10", "256/10"])
    configs["256/10"]["truncated_share"] = 0.10
    assert decide(configs)[0] == "384/10"
