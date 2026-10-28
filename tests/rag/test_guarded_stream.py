"""Real generation/client/validator/rendering with controlled HTTP and NLI inference.

These tests exercise the component handoff without downloading model weights. They
do not measure a model's semantic quality or replace live acceptance runs.
"""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.contracts import Evidence, Plan
from scilitrag.guardrails.validation import NliModel, build_component
from scilitrag.rag.evidence.citations import render_citations
from scilitrag.rag.generation import generate_with
from scilitrag.rag.llm.client import ChatClient


@pytest.mark.parametrize("one_character", [False, True])
def test_only_checked_supported_sentences_reach_the_client(monkeypatch, one_character):
    text = (
        "Supported claim [1]. Neutral claim [1]. Wrong value 99 [1]. "
        "Unknown source [99]. Supported tail. [1]"
    )
    deltas = list(text) if one_character else [text[:20], text[20:]]
    body = (
        "".join(
            "data: " + json.dumps({"choices": [{"delta": {"content": delta}}]}) + "\n\n"
            for delta in deltas
        )
        + "data: [DONE]\n\n"
    )
    settings = load_settings(
        {"USE_MOCKS": "1", "LLM_BASE_URL": "http://model.test/v1", "LLM_MODEL": "candidate"}
    )
    requests, nli_calls, released = [], [], []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    def inference(self, premise, hypothesis):
        nli_calls.append(hypothesis)
        return {
            "entailment": 0.1 if "Neutral" in hypothesis else 0.9,
            "neutral": 0.8 if "Neutral" in hypothesis else 0.05,
            "contradiction": 0.05,
        }

    monkeypatch.setattr(NliModel, "probabilities", inference)
    client = ChatClient(settings, transport=httpx.MockTransport(respond))
    validator = build_component(settings)
    evidence = [
        Evidence(
            label=1,
            type="chunk",
            corpus_id=7,
            chunk_id="7-0",
            text="Supported claim. Supported tail.",
        )
    ]

    def mask(sentence):
        released.append(("mask", sentence))
        return sentence

    def render(sentence, mapping):
        released.append(("render", sentence))
        return render_citations(sentence, mapping)

    c = replace(
        mock_components(0),
        plan=lambda _: Plan(scenario="S1", topic="retrieval", k=1),
        retrieve=lambda _: evidence,
        generate=lambda p, e: generate_with(p, e, client, settings),
        check_sentence=validator,
        mask_pii=mask,
        render_citations=render,
    )

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    accepted = [e for name, e in events if name == "sentence"]
    assert [e.text for e in accepted] == ["Supported claim [1].", "Supported tail. [1]"]
    assert all(e.check.status == "ok" and e.citations[0].chunk_id == "7-0" for e in accepted)
    assert all(e.citations[0].source_hash for e in accepted)
    assert [stage for stage, _ in released] == ["mask", "render", "mask", "render"]
    assert len(nli_calls) == 3  # unknown label and wrong number stop before NLI
    removed = [e.result for n, e in events if n == "status" and e.result.startswith("removed:")]
    assert removed == ["removed: unsupported", "removed: fact_mismatch", "removed: unknown_label"]
    assert events[-1][0] == "done"
    assert events[-1][1].sentences_checked == 5 and events[-1][1].sentences_removed == 3
    assert requests[0]["stream"] and requests[0]["model"] == "candidate"


def test_incomplete_tail_is_not_sent_to_the_validator():
    checked = []

    async def generate(*_):
        yield "Unfinished claim [1]"

    def forbidden(*args):
        checked.append(args)
        raise AssertionError("Fragment reached semantic validation")

    c = replace(mock_components(0), generate=generate, check_sentence=forbidden)

    async def collect():
        return [event async for event in run_query("retrieval", c)]

    events = asyncio.run(collect())
    assert not checked and not any(name == "sentence" for name, _ in events)
    assert events[-1][0] == "error" and "incomplete sentence" in events[-1][1].message
