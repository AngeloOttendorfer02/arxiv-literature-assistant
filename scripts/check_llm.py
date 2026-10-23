"""Verify real configured completions/streaming and controlled HTTP timeouts.

Run from the project root. No database or retrieval component is used.
The timeout probe uses a deliberately stalled local HTTP endpoint, not Ollama.
"""

import argparse
import asyncio
import json
import threading
import time
from contextlib import aclosing, contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from scilitrag.config import ConfigError, load_settings
from scilitrag.rag.errors import LLMClientError
from scilitrag.rag.llm.client import ChatClient

MESSAGES = [
    {"role": "system", "content": "Give a short answer. Do not include analysis or explanations."},
    {"role": "user", "content": "In one short sentence, what is information retrieval?"},
]


@contextmanager
def stalled_endpoint():
    """Accept HTTP requests without returning headers before the client deadline."""
    release = threading.Event()
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.set()
            release.wait(5)
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


async def stream_result(client):
    """Consume the complete SSE response and measure text-delta delivery."""
    started = time.perf_counter()
    first_delta_s = None
    fragments = []
    async with aclosing(client.stream(MESSAGES)) as stream:
        async for fragment in stream:
            if first_delta_s is None:
                first_delta_s = time.perf_counter() - started
            fragments.append(fragment)
    if not "".join(fragments).strip():
        raise LLMClientError("The model stream did not return answer text.")
    return {
        "status": "passed",
        "elapsed_s": round(time.perf_counter() - started, 4),
        "first_text_delta_s": round(first_delta_s, 4),
        "text_deltas": len(fragments),
        "characters": sum(map(len, fragments)),
    }


def timeout_result(settings):
    """Exercise real socket read timeouts in both modes against a local stall."""
    results = {}
    with stalled_endpoint() as (base_url, received):
        probe = replace(settings, llm_base_url=base_url, request_timeout_s=0.25, llm_max_retries=0)
        for mode in ("completion", "stream"):
            received.clear()
            started = time.perf_counter()
            try:
                if mode == "completion":
                    ChatClient(probe, transport=httpx.HTTPTransport()).complete(MESSAGES)
                else:
                    asyncio.run(
                        stream_result(ChatClient(probe, transport=httpx.AsyncHTTPTransport()))
                    )
            except LLMClientError as error:
                if error.kind != "timeout" or not received.is_set():
                    raise LLMClientError(
                        f"The controlled {mode} timeout probe failed.", kind="verification"
                    ) from None
                results[mode] = {
                    "status": "passed",
                    "error_kind": error.kind,
                    "elapsed_s": round(time.perf_counter() - started, 4),
                }
            else:
                raise LLMClientError(
                    f"The controlled {mode} timeout probe did not time out.", kind="verification"
                )
    return {"endpoint": "controlled local stalled HTTP server", "checks": results}


def check(settings):
    """Always contact the configured model, regardless of mock component selection."""
    client = ChatClient(settings)
    started = time.perf_counter()
    answer = client.complete(MESSAGES)
    if not answer.strip():
        raise LLMClientError("The model completion did not return answer text.")
    completion = {
        "status": "passed",
        "elapsed_s": round(time.perf_counter() - started, 4),
        "characters": len(answer),
    }
    streaming = asyncio.run(stream_result(client))
    timeouts = timeout_result(settings)
    return {
        "status": "passed",
        "model": settings.llm_model,
        "request_timeout_s": settings.request_timeout_s,
        "max_output_tokens": settings.llm_max_output_tokens,
        "completion": completion,
        "streaming": streaming,
        "timeout_probe": timeouts,
        "throughput": "Not measured: text deltas are not provider tokens.",
        "hardware": "Record Ollama device placement separately; not inferred by this script.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional local JSON acceptance record.")
    args = parser.parse_args()
    try:
        report = check(load_settings())
    except (ConfigError, LLMClientError) as error:
        kind = getattr(error, "kind", "configuration")
        # Do not print configuration values, URLs or private provider errors.
        parser.exit(1, f"LLM acceptance failed ({kind}); no success report was written.\n")
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
