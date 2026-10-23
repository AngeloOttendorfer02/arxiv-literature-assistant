"""Acceptance-tool orchestration and actual loopback timeout checks."""

import asyncio
from dataclasses import replace

import check_llm
import pytest

from scilitrag.config import load_settings
from scilitrag.rag.errors import LLMClientError

SETTINGS = load_settings(
    {"USE_MOCKS": "1", "LLM_BASE_URL": "http://model/v1", "LLM_MODEL": "configured-model"}
)


def test_acceptance_uses_configured_client_even_with_mocks_enabled(monkeypatch):
    calls = []

    class Client:
        def __init__(self, settings):
            calls.append(settings)

        def complete(self, messages):
            assert messages == check_llm.MESSAGES
            return "Retrieval finds documents."

        async def stream(self, messages):
            assert messages == check_llm.MESSAGES
            yield "Retrieval "
            yield "finds documents."

    monkeypatch.setattr(check_llm, "ChatClient", Client)
    monkeypatch.setattr(check_llm, "timeout_result", lambda settings: {"tested": True})
    report = check_llm.check(SETTINGS)
    assert calls == [SETTINGS]
    assert report["model"] == "configured-model"
    assert report["streaming"]["text_deltas"] == 2
    assert report["completion"]["characters"] > 0
    assert "not provider tokens" in report["throughput"]


def test_real_http_timeouts_are_translated_in_both_modes():
    report = check_llm.timeout_result(replace(SETTINGS, llm_max_retries=3))
    assert "controlled local" in report["endpoint"]
    for mode in ("completion", "stream"):
        assert report["checks"][mode]["error_kind"] == "timeout"


def test_stream_is_closed_when_consumption_fails():
    closed = []

    class Client:
        async def stream(self, messages):
            try:
                yield "Partial answer"
                raise LLMClientError("Provider stopped.")
            finally:
                closed.append(True)

    with pytest.raises(LLMClientError):
        asyncio.run(check_llm.stream_result(Client()))
    assert closed == [True]


def test_failed_acceptance_does_not_create_a_success_record(tmp_path, monkeypatch, capsys):
    output = tmp_path / "acceptance.json"

    def fail(settings):
        raise LLMClientError("private provider details", kind="timeout")

    monkeypatch.setattr(check_llm, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(check_llm, "check", fail)
    monkeypatch.setattr("sys.argv", ["check_llm.py", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        check_llm.main()
    assert error.value.code == 1
    assert not output.exists()
    assert "private provider details" not in capsys.readouterr().err
