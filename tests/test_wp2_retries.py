import asyncio
from dataclasses import replace

import httpx
import pytest

from scilitrag.config import load_settings
from scilitrag.rag.llm_client import ChatClient, LLMClientError
from scilitrag.rag.retry import RetryBudget

SETTINGS = load_settings(
    {
        "USE_MOCKS": "1",
        "LLM_BASE_URL": "http://model/v1",
        "LLM_MODEL": "test",
        "LLM_RETRY_BACKOFF_S": "0.001",
    }
)


def test_transient_completion_recovers():
    calls = []

    def handler(request):
        calls.append(request)
        return (
            httpx.Response(503, headers={"retry-after": "0"})
            if len(calls) == 1
            else httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )

    assert ChatClient(SETTINGS, transport=httpx.MockTransport(handler)).complete([]) == "ok"
    assert len(calls) == 2


def test_authentication_failures_are_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401)

    with pytest.raises(LLMClientError):
        ChatClient(SETTINGS, transport=httpx.MockTransport(handler)).complete([])
    assert len(calls) == 1


def test_partial_stream_is_never_replayed():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"first"}}]}\n\n')

    async def collect():
        result = []
        with pytest.raises(LLMClientError, match="ended"):
            async for token in ChatClient(SETTINGS, transport=httpx.MockTransport(handler)).stream(
                []
            ):
                result.append(token)
        return result

    assert asyncio.run(collect()) == ["first"] and len(calls) == 1


def test_backoff_cannot_exceed_remaining_budget():
    budget = RetryBudget.start(replace(SETTINGS, request_timeout_s=0.00001))
    request = httpx.Request("POST", "http://model")
    assert budget.delay(0, httpx.ConnectError("offline", request=request)) is None
