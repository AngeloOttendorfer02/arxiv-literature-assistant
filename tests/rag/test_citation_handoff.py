import asyncio
from dataclasses import replace

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.contracts import Citation, SqlFacts
from scilitrag.rag.evidence.citations import render_citations
from scilitrag.ui.citation_details import citation_details


def test_unknown_labels_are_removed_before_semantic_validation():
    mocks = mock_components(0)
    checked = []

    def semantic(sentence, mapping):
        checked.append(sentence)
        return mocks.check_sentence(sentence, mapping)

    async def generate(plan, evidence):
        yield "Unknown source [999]. Valid source [1]. "

    c = replace(
        mocks, generate=generate, check_sentence=semantic, render_citations=render_citations
    )

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    sentences = [e for name, e in events if name == "sentence"]
    assert len(sentences) == 1 and sentences[0].citations[0].source_hash
    assert checked == ["Valid source [1]."]
    assert events[-1][1].sentences_removed == 1


def test_drawer_preserves_sql_scope_and_author_provenance():
    table = SqlFacts(
        query_id="q",
        sql="SELECT fixed",
        parameters={"topic": "retrieval"},
        scope="Indexed corpus",
        dataset_revision="r1",
        columns=["author", "count"],
        rows=[["A", 2]],
    )
    citation = Citation(label=1, query_id="q", author_ids=[7], source_hash="abc")
    details = citation_details(citation, [table])
    assert details["kind"] == "sql" and details["parameters"] == {"topic": "retrieval"}
    assert details["author_ids"] == [7] and details["dataset_revision"] == "r1"
