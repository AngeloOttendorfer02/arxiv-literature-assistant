"""D-12 full-text selection checks with synthetic rows, never real LitSearch."""

import json

import pytest
import select_fulltext
from select_fulltext import choose_fulltext, query_targets, write_selection


def paper(cid, *, title="Synthetic paper", year=2020, citations=()):
    return {
        "corpusid": cid,
        "title": title,
        "abstract": "",
        "year": year,
        "citations": list(citations),
    }


ROWS = [
    paper(1),
    paper(2, title="Evaluating retrieval-augmented generation", year=2023),
    paper(3, title="RAG for question answering", year=2021),
    paper(4, title="Ragged arrays", year=2023),
    paper(5, citations=[9]),
    paper(6, year=2022, citations=[9, 1]),
    paper(9, title="Dense retriever"),
    *(paper(cid) for cid in range(10, 30)),
]


def test_groups_follow_priority_and_fill_randomly():
    selected, groups = choose_fulltext(ROWS, {1, 6}, count=10, seed=1, s2_papers=(9,))
    assert groups["query_targets"] == [1, 6]
    assert groups["rag_topic"] == [2]
    assert groups["s2_demo_papers"] == [9]
    assert groups["s2_citing"] == [5]
    assert len(groups["random"]) == 5 and not set(groups["random"]) & {1, 2, 5, 6, 9}
    assert len(selected) == 10 == len(set(selected))
    assert selected == choose_fulltext(ROWS, {1, 6}, count=10, seed=1, s2_papers=(9,))[0]
    assert selected != choose_fulltext(ROWS, {1, 6}, count=10, seed=2, s2_papers=(9,))[0]


def test_budget_must_fit_fixed_groups():
    with pytest.raises(ValueError, match="more than"):
        choose_fulltext(ROWS, {1, 6}, count=4, s2_papers=(9,))


def test_targets_and_demo_papers_must_exist():
    with pytest.raises(ValueError, match="query target"):
        choose_fulltext(ROWS, {999}, count=10, s2_papers=(9,))
    with pytest.raises(ValueError, match="S2 demo paper"):
        choose_fulltext(ROWS, {1}, count=10, s2_papers=(999,))


def test_query_and_target_counts_must_match_the_manifest():
    queries = [{"corpusids": [1, 6]}, {"corpusids": [6]}]
    provenance = {"counts": {"queries": 2, "target_papers": 2}}
    assert query_targets(queries, provenance) == {1, 6}
    with pytest.raises(ValueError, match="Query count"):
        query_targets(queries[:1], provenance)
    with pytest.raises(ValueError, match="Target paper count"):
        query_targets(queries, {"counts": {"queries": 2, "target_papers": 3}})
    with pytest.raises(ValueError, match="nonempty list"):
        query_targets([{"corpusids": []}, {"corpusids": [6]}], provenance)


def test_failed_write_leaves_no_selection_and_allows_a_retry(tmp_path, monkeypatch):
    destination = tmp_path / "fulltext" / "v1"

    def fail(*args, **kwargs):
        raise TypeError("not serializable")

    monkeypatch.setattr(select_fulltext.json, "dumps", fail)
    with pytest.raises(TypeError):
        write_selection(destination, {"corpusids": [1]})
    assert not destination.exists()
    assert list((tmp_path / "fulltext").iterdir()) == []
    monkeypatch.undo()
    path = write_selection(destination, {"corpusids": [1]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"corpusids": [1]}
    with pytest.raises(FileExistsError):
        write_selection(destination, {"corpusids": [2]})
