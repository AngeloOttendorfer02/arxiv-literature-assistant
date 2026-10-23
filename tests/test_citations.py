import pytest

from scilitrag.contracts import Evidence
from scilitrag.rag.evidence.citations import render_citations
from scilitrag.rag.evidence.labels import citation_labels


def test_labels_are_unique_and_ordered():
    assert citation_labels("Result [2][1][2].") == [2, 1]
    assert citation_labels("No label.") == []
    with pytest.raises(ValueError):
        citation_labels("Invalid [0].")


def test_metadata_comes_from_evidence():
    e = Evidence(
        label=1,
        type="chunk",
        text="stored passage",
        title="Stored title",
        corpus_id=17,
        chunk_id="17-0",
        year=2022,
        author_ids=[3],
    )
    rendered = render_citations("Claim [1][1].", {1: e})
    assert rendered.text == "Claim [1][1]."
    [citation] = rendered.citations
    assert (citation.title, citation.year, citation.chunk_id) == ("Stored title", 2022, "17-0")
    assert citation.text == "stored passage" and citation.author_ids == [3]
    citation.author_ids.append(4)
    assert e.author_ids == [3]


def test_unknown_labels_do_not_leak_between_requests():
    e = Evidence(label=1, type="paper", text="source")
    render_citations("Known [1].", {1: e})
    with pytest.raises(ValueError, match="Unknown"):
        render_citations("Unknown [1].", {})
    with pytest.raises(ValueError, match="mapping"):
        render_citations("Broken [2].", {2: e})


def test_sql_citation_keeps_query_id():
    e = Evidence(label=2, type="sql", query_id="query-7", text="counts")
    assert render_citations("Count [2].", {2: e}).citations[0].query_id == "query-7"
