"""OpenAI-compatible completions with bounded retries and cancellable SSE.

Both request modes share one retry deadline. Streaming retries are permitted
only before the first text delta reaches the caller.
"""

import json
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from urllib.parse import urlsplit

import httpx

from scilitrag.config import Settings
from scilitrag.rag.errors import LLMClientError
from scilitrag.rag.llm.retry import RetryBudget

MAX_COMPLETION_BYTES = 2_000_000
MAX_STREAM_FRAME_CHARS = 131_072


class ChatClient:
    """Use configured model settings without exposing private provider errors."""

    def __init__(self, settings: Settings, *, transport=None) -> None:
        """Accept an optional HTTP transport for deterministic service tests."""
        self.settings = settings
        self.transport = transport

    def _url(self) -> str:
        base = (self.settings.llm_base_url or "").rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LLMClientError("Configure LLM_BASE_URL.", kind="configuration")
        return f"{base}/chat/completions"

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.settings.llm_model:
            raise LLMClientError("Configure LLM_MODEL.", kind="configuration")
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "stream": stream,
            "temperature": 0,
            "max_tokens": self.settings.llm_max_output_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        return payload

    def _complete_once(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> str:
        url = self._url()
        payload = self._payload(messages, stream=False, schema=schema)
        timeout = timeout or self.settings.request_timeout_s
        deadline = time.monotonic() + timeout
        body = bytearray()
        try:
            with (
                httpx.Client(timeout=timeout, transport=self.transport) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if time.monotonic() >= deadline:
                        raise LLMClientError("The model request timed out.", kind="timeout")
                    body.extend(chunk)
                    if len(body) > MAX_COMPLETION_BYTES:
                        raise LLMClientError(
                            "The model response exceeds the configured service limit."
                        )
            content = json.loads(body)["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty completion")
            return content
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMClientError("The model returned an invalid response.") from None

    async def _stream_once(
        self, messages: list[dict[str, str]], *, timeout: float | None = None
    ) -> AsyncIterator[str]:
        timeout = timeout or self.settings.request_timeout_s
        deadline = time.monotonic() + timeout
        url = self._url()
        payload = self._payload(messages, stream=True)
        async with (
            httpx.AsyncClient(timeout=timeout, transport=self.transport) as client,
            client.stream("POST", url, json=payload) as response,
        ):
            response.raise_for_status()
            data: list[str] = []
            async for line in response.aiter_lines():
                if time.monotonic() >= deadline:
                    raise LLMClientError("The model request timed out.", kind="timeout")
                if len(line) + sum(map(len, data)) > MAX_STREAM_FRAME_CHARS:
                    raise LLMClientError("The model stream frame exceeds the service limit.")
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    # Blank lines terminate SSE events; multiple data fields
                    # form one payload. Keep-alive comments carry no content.
                    value = "\n".join(data)
                    data.clear()
                    if value == "[DONE]":
                        return
                    token = self._delta(value)
                    if token:
                        yield token
            if data:
                # Some providers omit the final blank line, but [DONE] is
                # still required to distinguish success from a broken stream.
                value = "\n".join(data)
                if value == "[DONE]":
                    return
                token = self._delta(value)
                if token:
                    yield token
            raise LLMClientError("The model stream ended before completion.")

    @staticmethod
    def _delta(value: str) -> str:
        try:
            packet = json.loads(value)
            if packet.get("error"):
                raise ValueError("stream error")
            choices = packet["choices"]
            if not choices:  # optional usage-only packet
                return ""
            if choices[0].get("finish_reason") == "length":
                raise LLMClientError(
                    "The model reached its generation limit before completing the answer.",
                    kind="truncated",
                )
            token = choices[0]["delta"].get("content")
            if token is not None and not isinstance(token, str):
                raise ValueError("invalid token")
            return token or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMClientError("The model returned an invalid stream event.") from None

    @staticmethod
    def _translate(error: Exception) -> LLMClientError:
        if isinstance(error, LLMClientError):
            return error
        if isinstance(error, httpx.TimeoutException):
            return LLMClientError("The model request timed out.", kind="timeout")
        return LLMClientError("The model service is unavailable.", kind="unavailable")

    def complete(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None
    ) -> str:
        """Return one completion within the shared request/retry time budget."""
        budget = RetryBudget.start(self.settings)
        for attempt in range(budget.max_retries + 1):
            try:
                if budget.remaining() <= 0:
                    raise LLMClientError("The model request timed out.", kind="timeout")
                return self._complete_once(messages, schema=schema, timeout=budget.remaining())
            except (httpx.HTTPError, LLMClientError) as error:
                delay = budget.delay(attempt, error)
                if delay is None:
                    raise self._translate(error) from None
                budget.wait(delay)
        raise AssertionError("unreachable")

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Yield text deltas and close the HTTP stream when the caller cancels."""
        budget = RetryBudget.start(self.settings)
        for attempt in range(budget.max_retries + 1):
            emitted = False
            try:
                if budget.remaining() <= 0:
                    raise LLMClientError("The model request timed out.", kind="timeout")
                async with aclosing(
                    self._stream_once(messages, timeout=budget.remaining())
                ) as stream:
                    async for token in stream:
                        emitted = True
                        yield token
                return
            except (httpx.HTTPError, LLMClientError) as error:
                # Replaying a partially delivered answer would duplicate text
                # and could change the evidence associated with later claims.
                delay = None if emitted else budget.delay(attempt, error)
                if delay is None:
                    raise self._translate(error) from None
                await budget.wait_async(delay)
        raise AssertionError("unreachable")
