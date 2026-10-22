import asyncio
from dataclasses import replace

import pytest

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.contracts import Plan
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.retrieval import retrieve_with

PLAN = Plan(scenario="S1", topic="retrieval", k=2)


class Backend:
    def hybrid(self, plan):
        return [
            {
                "corpus_id": 1,
                "title": "A",
                "year": 2021,
                "abstract": "Method A",
                "has_indexed_chunks": True,
            },
            {
                "corpus_id": 2,
                "title": "B",
                "year": 2022,
                "abstract": "Survey B",
                "has_indexed_chunks": False,
            },
        ]

    def chunks(self, plan, ids):
        assert ids == [1]
        return [
            {"chunk_id": "1-0", "corpus_id": 1, "section": "Method", "text": "Method evidence."}
        ]


def test_two_stage_sources_have_stable_identity_and_coverage():
    evidence = retrieve_with(PLAN, Backend())
    assert [e.label for e in evidence] == [1, 2, 3]
    assert evidence[1].chunk_id == "1-0" and evidence[2].section == "Abstract only"
    assert "unavailable" in evidence[2].text


def test_available_fulltext_without_a_passage_is_still_abstract_only():
    class MissingPassages(Backend):
        def chunks(self, *args):
            return []

    evidence = retrieve_with(PLAN, MissingPassages())
    assert evidence[0].section == "Abstract only"
    assert "Indexed passages: available" in evidence[0].text
    assert "passages: none" in evidence[0].text


def test_empty_results_stop_before_generation():
    class Empty(Backend):
        def hybrid(self, plan):
            return []

    mocks = mock_components(0)

    def forbidden(*args):
        raise AssertionError("generation was called")

    c = replace(
        mocks, plan=lambda _: PLAN, retrieve=lambda p: retrieve_with(p, Empty()), generate=forbidden
    )

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    assert events[-1][0] == "error" and events[-1][1].code == "bad_request"
    assert "No matching papers" in events[-1][1].message


def test_cross_paper_support_is_rejected():
    class WrongScope(Backend):
        def chunks(self, *args):
            return [{"corpus_id": 99, "chunk_id": "99-0", "text": "Wrong"}]

    with pytest.raises(RagUserError):
        retrieve_with(PLAN, WrongScope())
