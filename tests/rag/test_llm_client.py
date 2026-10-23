import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components
from scilitrag.config import load_settings
from scilitrag.rag.llm.client import ChatClient, LLMClientError

SETTINGS = load_settings({"USE_MOCKS": "1", "LLM_BASE_URL": "http://model/v1", "LLM_MODEL": "test"})


def test_completion_uses_config_and_schema():
    def handler(request):
        body = json.loads(request.content)
        assert str(request.url) == "http://model/v1/chat/completions"
        assert body["model"] == "test" and body["temperature"] == 0
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = ChatClient(SETTINGS, transport=httpx.MockTransport(handler))
    assert client.complete([], schema={"type": "object"}) == "{}"


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_are_controlled(status):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, text="secret body"))
    with pytest.raises(LLMClientError) as err:
        ChatClient(SETTINGS, transport=transport).complete([])
    assert err.value.kind == "unavailable"
    assert "secret" not in str(err.value)


def test_timeout_and_configuration():
    def timeout(request):
        raise httpx.ReadTimeout("secret", request=request)

    with pytest.raises(LLMClientError) as err:
        ChatClient(SETTINGS, transport=httpx.MockTransport(timeout)).complete([])
    assert err.value.kind == "timeout"
    with pytest.raises(LLMClientError):
        ChatClient(replace(SETTINGS, llm_model=None)).complete([])


def test_stream_parses_frames_and_ignores_comments():
    content = (
        ': keepalive\n\ndata: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=content))

    async def collect():
        return [s async for s in ChatClient(SETTINGS, transport=transport).stream([])]

    assert asyncio.run(collect()) == ["Hello ", "world"]


@pytest.mark.parametrize("content", ["data: nope\n\n", 'data: {"choices":[]}\n\n'])
def test_stream_rejects_invalid_or_incomplete_responses(content):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=content))

    async def collect():
        return [s async for s in ChatClient(SETTINGS, transport=transport).stream([])]

    with pytest.raises(LLMClientError):
        asyncio.run(collect())


def test_pipeline_reports_timeout():
    async def generate(plan, evidence):
        raise LLMClientError("The model request timed out.", kind="timeout")
        yield "unreachable"

    async def collect():
        c = replace(mock_components(0), generate=generate)
        return [item async for item in run_query("dense retrieval", c)]

    events = asyncio.run(collect())
    assert events[-1][0] == "error" and events[-1][1].code == "timeout"
