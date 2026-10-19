from collections.abc import Mapping
from urllib.parse import urlsplit

import httpx

from scilitrag.config import Settings
from scilitrag.rag.errors import LLMClientError


class ChatClient:
    def __init__(self, settings: Settings, *, transport=None) -> None:
        self.settings = settings
        self.transport = transport

    def _url(self) -> str:
        base = (self.settings.llm_base_url or "").rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LLMClientError("Configure LLM_BASE_URL.", kind="configuration")
        return f"{base}/chat/completions"

    def _payload(self, messages, *, stream: bool, schema=None) -> dict:
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

    def complete(self, messages: list[Mapping[str, str]], *, schema=None) -> str:
        url = self._url()
        payload = self._payload(messages, stream=False, schema=schema)
        try:
            with httpx.Client(
                timeout=self.settings.request_timeout_s, transport=self.transport
            ) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty completion")
                return content
        except httpx.TimeoutException:
            raise LLMClientError("The model request timed out.", kind="timeout") from None
        except httpx.HTTPError:
            raise LLMClientError("The model service is unavailable.", kind="unavailable") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMClientError("The model returned an invalid response.") from None
