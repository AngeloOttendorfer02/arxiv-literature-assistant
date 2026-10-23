import asyncio
from dataclasses import replace

import pytest

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.errors import LLMClientError, RagUserError
from scilitrag.rag.retrieval import retrieve_with


@pytest.mark.parametrize(
    "scenario,message",
    [
        ("S2", "No indexed papers cite this target within the requested filters."),
        ("S3", "No author citation credits were found for the selected topic papers."),
    ],
)
def test_scenario_failures_are_terminal_safe_errors(scenario, message):
    def fail(_plan):
        raise RagUserError(message)

    def never_generate(*args):
        raise AssertionError("Generation must not start after retrieval/facts failure")

    changes = {"retrieve": fail} if scenario == "S2" else {"sql_facts": fail}
    c = replace(
        mock_components(0),
        plan=lambda _: Plan(scenario=scenario, k=1),
        generate=never_generate,
        **changes,
    )

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    assert events[-1][0] == "error"
    assert events[-1][1].code == "bad_request"
    assert events[-1][1].message == message
    assert not any(name in {"sentence", "done"} for name, _ in events)


def test_model_timeout_during_real_generation_has_clear_terminal_event():
    from scilitrag.rag.generation import generate_with

    class Client:
        async def stream(self, _messages):
            raise LLMClientError("The model service timed out. Try again.", kind="timeout")
            yield "unreachable"

    evidence = [Evidence(label=1, type="paper", corpus_id=1, text="A retrieval paper.")]
    plan = Plan(scenario="S1", topic="retrieval", k=1)

    async def generate(p, sources):
        async for token in generate_with(p, sources, Client()):
            yield token

    c = replace(
        mock_components(0), plan=lambda _: plan, retrieve=lambda _: evidence, generate=generate
    )

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    assert events[-1][0] == "error" and events[-1][1].code == "timeout"
    assert "timed out" in events[-1][1].message


def test_empty_s3_stops_before_sql_or_model():
    class Empty:
        def hybrid(self, plan):
            return []

    with pytest.raises(RagUserError, match="No matching papers"):
        retrieve_with(Plan(scenario="S3", topic="retrieval", k=1), Empty())
