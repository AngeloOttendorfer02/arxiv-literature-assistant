"""D-23 statistics helpers with synthetic rows, never real LitSearch."""

import json

import pytest
from data_statistics import (
    bibliography_row,
    bibliography_summary,
    check_ids,
    drop_middle_initials,
    extra_authors,
    fold_accents,
    link_counts,
    metadata_counts,
)


def s2orc_row(corpus_id, matched, unmatched=0, encoded=True):
    entries = [{"attributes": {"matched_paper_id": cid}} for cid in matched]
    entries += [{"attributes": {}}] * unmatched
    bibentry = json.dumps(entries) if encoded else entries
    return {"corpusid": corpus_id, "content": {"annotations": {"bibentry": bibentry}}}


def test_bibliography_counts_entries_matches_and_corpus_hits_without_self_citations():
    ids = {1, 2, 3}
    rows = [
        bibliography_row(s2orc_row(1, [2, 2, 3, 1, 99], unmatched=2), ids),
        bibliography_row(s2orc_row(2, [3], encoded=False), ids),
        bibliography_row({"corpusid": 3, "content": {"annotations": {}}}, ids),
    ]
    assert rows == [(7, 5, 3, 2), (1, 1, 1, 1), (0, 0, 0, 0)]
    assert bibliography_summary(rows) == {
        "papers": 3,
        "entries": 8,
        "matched_entries": 6,
        "in_corpus_entries": 4,
        "in_corpus_distinct_pairs": 3,
        "papers_with_repeated_entries": 1,
        "papers_without_bibliography": 1,
        "median_entries_per_paper": 1,
    }


@pytest.mark.parametrize("found", [[1, 1, 3], [1, 2], [1, 2, 3, 4], [1, 2, 3, 3]])
def test_corpus_ids_must_match_the_manifest_exactly_once(found):
    with pytest.raises(ValueError, match="exactly once"):
        check_ids(found, {1, 2, 3}, "corpus_s2orc")
    check_ids([3, 1, 2], {1, 2, 3}, "corpus_s2orc")


def test_link_counts_separate_citing_cited_isolated_and_mutual_papers():
    pairs = {(1, 2), (2, 1), (3, 1)}
    assert link_counts(pairs, {1, 2, 3, 4}) == {
        "papers_citing_no_corpus_paper": 1,
        "papers_not_cited": 2,
        "papers_without_links": 1,
        "mutual_pairs": 1,
    }


def test_metadata_counts_treat_blank_and_null_text_as_missing():
    rows = [
        {"title": "A", "abstract": "B", "year": 2020},
        {"title": " ", "abstract": None, "year": None},
        {"title": None, "abstract": "B", "year": 2021},
    ]
    assert metadata_counts(rows) == {
        "without_title": 2,
        "without_abstract": 1,
        "without_title_and_abstract": 1,
        "without_year": 1,
        "by_year": {2020: 1, 2021: 1},
    }


def test_open_author_variants_count_only_names_that_would_merge():
    assert fold_accents("adam liška") == "adam liska"
    assert drop_middle_initials("aaron w li") == "aaron li"
    assert drop_middle_initials("a li") == "a li"
    keys = ["adam liska", "adam liška", "aaron li", "aaron w li", "aaron w x li", "bo li"]
    assert extra_authors(keys, fold_accents) == 1
    assert extra_authors(keys, drop_middle_initials) == 2
    assert extra_authors(keys, lambda key: key) == 0
