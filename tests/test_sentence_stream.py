import asyncio
from dataclasses import replace

import pytest

from scilitrag.api.pipeline import run_query, split_sentences
from scilitrag.components import mock_components
from scilitrag.rag.errors import LLMClientError
from scilitrag.rag.sentence_stream import SentenceBuffer


def test_buffer_releases_complete_sentences_and_flushes_a_complete_tail():
    buffer = SentenceBuffer(split_sentences, max_chars=100)
    assert buffer.feed("First result ") == []
    assert buffer.feed("[1]. Second result [2].") == ["First result [1]."]
    assert buffer.finish() == ["Second result [2]."]


@pytest.mark.parametrize("token", [None, 123])
def test_nontext_deltas_are_controlled_errors(token):
    with pytest.raises(LLMClientError):
        SentenceBuffer(split_sentences).feed(token)


def test_incomplete_and_oversized_output_are_rejected():
    buffer = SentenceBuffer(split_sentences, max_chars=10)
    with pytest.raises(LLMClientError, match="too much"):
        buffer.feed("x" * 11)
    buffer = SentenceBuffer(split_sentences)
    buffer.feed("Unfinished claim [1]")
    with pytest.raises(LLMClientError, match="incomplete"):
        buffer.finish()


def test_api_reports_buffer_failure_without_releasing_a_claim():
    async def generate(plan, evidence):
        yield "x" * 100

    c = replace(mock_components(0), generate=generate)

    async def collect():
        return [event async for event in run_query("retrieval", c, max_buffer_chars=32)]

    events = asyncio.run(collect())
    assert not any(name == "sentence" for name, _ in events)
    assert events[-1][0] == "error" and "unfinished" in events[-1][1].message
