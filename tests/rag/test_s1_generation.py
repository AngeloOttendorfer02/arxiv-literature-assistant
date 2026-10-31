import asyncio
import json
from dataclasses import replace

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.evidence.citations import render_citations
from scilitrag.rag.generation import generate_with


def test_grounded_generation_flows_through_sentence_validation():
    plan = Plan(scenario="S1", query="Compare retrieval methods", topic="retrieval", k=1)
    evidence = [
        Evidence(
            label=1,
            type="paper",
            corpus_id=7,
            title="A",
            section="Abstract only",
            text="Retrieved full-text passages: none. Abstract: A retrieval method.",
        )
    ]

    class Client:
        async def stream(self, messages):
            assert json.loads(messages[1]["content"])["question"] == plan.query
            yield "The abstract describes a retrieval method [1]. "

    async def generate(p, sources):
        async for token in generate_with(p, sources, Client()):
            yield token

    c = replace(
        mock_components(0),
        plan=lambda _: plan,
        retrieve=lambda _: evidence,
        generate=generate,
        render_citations=render_citations,
    )

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    sentences = [e for name, e in events if name == "sentence"]
    assert len(sentences) == 2 and all(s.citations[0].corpus_id == 7 for s in sentences)
    assert all(s.citations[0].source_hash for s in sentences)
    assert events[-1][0] == "done"


def test_no_evidence_never_calls_the_model():
    class Forbidden:
        def stream(self, *args):
            raise AssertionError("called")

    async def collect():
        return [token async for token in generate_with(Plan(scenario="S1", k=1), [], Forbidden())]

    assert asyncio.run(collect()) == []


def test_closing_generation_closes_the_provider_stream():
    state = []

    class Client:
        async def stream(self, messages):
            try:
                yield "first "
                yield "second"
            finally:
                state.append("closed")

    evidence = [Evidence(label=1, type="paper", corpus_id=1, text="source")]

    async def scenario():
        stream = generate_with(Plan(scenario="S1", k=1), evidence, Client())
        assert await anext(stream) == "first "
        await stream.aclose()

    asyncio.run(scenario())
    assert state == ["closed"]


def test_plain_s3_ranking_uses_sql_without_calling_language_model():
    plan = Plan(
        scenario="S3", query="Rank up to 3 most cited authors for dense passage retrieval.",
        topic="dense passage retrieval", k=3,
    )
    evidence = [Evidence(label=10, type="sql", query_id="ranking", text="ranking data")]

    class Forbidden:
        def stream(self, *args):
            raise AssertionError("plain ranking must not request a paper summary")

    async def collect():
        return [token async for token in generate_with(plan, evidence, Forbidden())]

    result = asyncio.run(collect())
    assert len(result) == 1 and "[10]" in result[0]


def test_s3_ranking_with_a_summary_still_calls_language_model():
    plan = Plan(
        scenario="S3",
        query="Rank authors for dense passage retrieval and explain their work.",
        topic="dense passage retrieval", k=3,
    )
    evidence = [
        Evidence(label=1, type="paper", corpus_id=1, text="source"),
        Evidence(label=2, type="sql", query_id="ranking", text="ranking data"),
    ]

    class Client:
        async def stream(self, messages):
            yield "A paper discusses retrieval [1]. "

    async def collect():
        return [token async for token in generate_with(plan, evidence, Client())]

    result = asyncio.run(collect())
    assert len(result) == 2 and "retrieval" in result[-1]
