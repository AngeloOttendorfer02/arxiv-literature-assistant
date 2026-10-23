"""D-09 citation pair checks with synthetic lists, never real LitSearch."""

import json

from citations import bibliography_counts, bibliography_ids, build_citations, direction_counts


def test_pairs_keep_direction_and_drop_self_outside_and_duplicates():
    graph = {1: [2, 2, 1, 99], 2: [1], 3: []}
    pairs, counts = build_citations(graph, {1, 2, 3})
    assert pairs == [(1, 2), (2, 1)]
    assert counts == {
        "entries": 5,
        "duplicate": 1,
        "self_citation": 1,
        "outside_corpus": 1,
        "citations": 2,
    }


def test_pairs_require_both_papers_inside_corpus():
    graph = {
        1: [2],  # Citing paper is outside.
        2: [3],  # Both papers are inside.
        3: [1],  # Cited paper is outside.
    }
    pairs, counts = build_citations(graph, {2, 3})
    assert pairs == [(2, 3)]
    assert counts == {"entries": 3, "outside_corpus": 2, "citations": 1}


def test_empty_graph_has_no_pairs():
    assert build_citations({}, set()) == ([], {"citations": 0})


def test_direction_counts_compare_years():
    years = {1: 2020, 2: 2018, 3: 2020, 4: 2022, 5: None}
    pairs = [(1, 2), (1, 3), (1, 4), (1, 5)]
    assert direction_counts(pairs, years) == {
        "citing_newer": 1,
        "same_year": 1,
        "citing_older": 1,
        "year_missing": 1,
    }


def test_bibliography_confirms_the_citing_paper():
    entries = [
        {"attributes": {"matched_paper_id": 2}},
        {"attributes": {"matched_paper_id": None}},
        {"attributes": {}},
    ]
    row = {"corpusid": 1, "content": {"annotations": {"bibentry": json.dumps(entries)}}}
    bibliographies = {1: bibliography_ids(row), 2: set()}
    assert bibliographies[1] == {2}
    assert bibliography_counts([(1, 2), (2, 1)], bibliographies) == {
        "in_bibliography": 1,
        "not_in_bibliography": 1,
    }


def test_paper_without_bibliography_has_no_ids():
    assert bibliography_ids({"corpusid": 1, "content": {"text": "Body"}}) == set()
