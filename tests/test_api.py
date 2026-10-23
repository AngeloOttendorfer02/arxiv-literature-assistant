import asyncio
import json
import logging
import socket
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from scilitrag.api.app import create_app
from scilitrag.api.pipeline import run_query
from scilitrag.api.telemetry import RequestIdFilter
from scilitrag.components import mock_components
from scilitrag.config import ConfigError, load_settings
from scilitrag.contracts import QueryRequest
from scilitrag.mocks.rag import Generate
from scilitrag.timings import record_timing
from scilitrag.ui.sse import iter_events

QUERY = "dense retrieval papers by alice@example.org"


class CountingGenerate(Generate):
    def __init__(self, token_delay: float) -> None:
        super().__init__(token_delay)
        self.produced = 0

    async def __call__(self, plan, evidence):
        async for token in super().__call__(plan, evidence):
            self.produced += 1
            yield token


def make_app(tmp_path, token_delay=0.0, **env):
    settings = load_settings({"USE_MOCKS": "1", "LOG_DIR": str(tmp_path), **env})
    generate = CountingGenerate(token_delay)
    components = replace(mock_components(0), generate=generate)
    return create_app(components, settings), generate


def log_records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_request_id_header_matches_log_without_query_text(tmp_path):
    app, _ = make_app(tmp_path)
    response = TestClient(app).post("/query", json={"query": QUERY})
    [record] = log_records(tmp_path)
    assert response.headers["x-request-id"] == record["request_id"]
    assert record["status"] == "ok"
    assert record["tokens"] > 0
    assert "alice" not in json.dumps(record) and "dense" not in json.dumps(record)


def test_component_logs_carry_the_request_id(tmp_path):
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append((record.getMessage(), record.request_id))

    handler = Capture()
    handler.addFilter(RequestIdFilter())
    logger = logging.getLogger("scilitrag.test_component")
    logger.addHandler(handler)
    mocks = mock_components(0)

    def check_input(query):  # runs in a worker thread
        logger.warning("check_input")
        return mocks.check_input(query)

    async def generate(plan, evidence):  # runs in the pipeline task
        logger.warning("generate")
        async for token in mocks.generate(plan, evidence):
            yield token

    components = replace(mocks, check_input=check_input, generate=generate)
    settings = load_settings({"USE_MOCKS": "1", "LOG_DIR": str(tmp_path)})
    try:
        response = TestClient(create_app(components, settings)).post(
            "/query", json={"query": QUERY}
        )
    finally:
        logger.removeHandler(handler)
    request_id = response.headers["x-request-id"]
    assert seen == [("check_input", request_id), ("generate", request_id)]


def test_log_has_component_timings_and_guardrail_scores(tmp_path):
    mocks = mock_components(0)

    def retrieve(plan):  # runs in a worker thread
        record_timing("paper_dense_ms", 2.0)
        record_timing("paper_dense_ms", 3.0)
        return mocks.retrieve(plan)

    settings = load_settings({"USE_MOCKS": "1", "LOG_DIR": str(tmp_path)})
    client = TestClient(create_app(replace(mocks, retrieve=retrieve), settings))
    for _ in range(2):
        client.post("/query", json={"query": QUERY})
    for record in log_records(tmp_path):
        assert record["component_ms"] == {"paper_dense_ms": 5.0}
        scores = {e["stage"]: e.get("score") for e in record["guardrail_events"]}
        assert scores == {"prompt_guard": 0.0, "topic_gate": 0.0, "pii": 1.0, "generate": None}


def test_record_timing_outside_a_request_is_ignored():
    record_timing("paper_dense_ms", 1.0)


def test_timeout_sends_timeout_event_and_logs_it(tmp_path):
    app, _ = make_app(tmp_path, token_delay=0.2, REQUEST_TIMEOUT_S="0.5")
    with TestClient(app).stream("POST", "/query", json={"query": QUERY}) as response:
        events = list(iter_events(response.iter_lines()))
    name, event = events[-1]
    assert name == "error" and event.code == "timeout"
    assert log_records(tmp_path)[0]["status"] == "timeout"


def test_closing_the_stream_stops_generation():
    generate = CountingGenerate(token_delay=0.05)
    components = replace(mock_components(0), generate=generate)

    async def scenario():
        events = run_query(QUERY, components, timeout_s=30)
        async for name, _ in events:
            if name == "sentence":
                break
        await events.aclose()
        at_close = generate.produced
        await asyncio.sleep(0.5)
        return at_close, generate.produced

    at_close, later = asyncio.run(scenario())
    assert later == at_close


def test_disconnect_is_logged_as_cancelled(tmp_path):
    app, _ = make_app(tmp_path, token_delay=0.05)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/query")
    request = Request({"type": "http", "state": {"request_id": "test-id"}})

    async def scenario():
        stream = endpoint(QueryRequest(query=QUERY), request)
        await anext(stream)
        await stream.aclose()

    asyncio.run(scenario())
    [record] = log_records(tmp_path)
    assert (record["request_id"], record["status"]) == ("test-id", "cancelled")


@pytest.mark.parametrize(
    ("database_url", "expected"),
    [
        (f"postgresql://app:app@127.0.0.1:{free_port()}/x", "unreachable"),
        ("postgresql://app:app@127.0.0.1:notaport/x", "invalid url"),
        ("postgresql:///x", "invalid url"),
    ],
)
def test_health_reports_unhealthy_database(tmp_path, database_url, expected):
    app, _ = make_app(tmp_path, REAL_COMPONENTS="retrieve", DATABASE_URL=database_url)
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["checks"] == {"database": expected, "llm": "skipped"}


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "abc"])
def test_request_timeout_must_be_positive_and_finite(value):
    with pytest.raises(ConfigError):
        load_settings({"USE_MOCKS": "1", "REQUEST_TIMEOUT_S": value})
