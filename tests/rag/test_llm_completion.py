import json
from dataclasses import replace

import httpx
import pytest

from scilitrag.config import ConfigError, load_settings
from scilitrag.rag.llm.client import ChatClient, LLMClientError

SETTINGS = load_settings(
    {"USE_MOCKS": "1", "LLM_BASE_URL": "http://model/v1", "LLM_MODEL": "candidate"}
)


def test_configured_completion_payload():
    def handler(request):
        data = json.loads(request.content)
        assert str(request.url) == "http://model/v1/chat/completions"
        assert data["model"] == "candidate" and data["max_tokens"] == 1024
        assert data["temperature"] == 0 and data["stream"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    assert ChatClient(SETTINGS, transport=httpx.MockTransport(handler)).complete([]) == "answer"


@pytest.mark.parametrize(
    "payload", [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}]
)
def test_invalid_provider_payloads(payload):
    client = ChatClient(
        SETTINGS, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    )
    with pytest.raises(LLMClientError, match="invalid response"):
        client.complete([])


def test_provider_errors_do_not_expose_response_bodies():
    client = ChatClient(
        SETTINGS, transport=httpx.MockTransport(lambda _: httpx.Response(401, text="secret"))
    )
    with pytest.raises(LLMClientError) as error:
        client.complete([])
    assert error.value.kind == "unavailable" and "secret" not in str(error.value)


@pytest.mark.parametrize(
    "env",
    [{"LLM_MAX_RETRIES": "7"}, {"LLM_MAX_OUTPUT_TOKENS": "0"}, {"LLM_RETRY_BACKOFF_S": "nan"}],
)
def test_request_policy_configuration_is_validated(env):
    with pytest.raises(ConfigError):
        load_settings({"USE_MOCKS": "1", **env})


def test_missing_model_and_endpoint_are_configuration_errors():
    for settings in [replace(SETTINGS, llm_model=None), replace(SETTINGS, llm_base_url=None)]:
        with pytest.raises(LLMClientError) as error:
            ChatClient(settings).complete([])
        assert error.value.kind == "configuration"
