import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from load_test import load_queries, load_test, pick, run_one, summarize

QUERIES = [{"id": f"q{i}", "query": f"question {i}"} for i in range(20)]


def sse(*events):
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def done(**fields):
    return {
        "stage_latency_ms": {"retrieve": 30.0, "generate": 900.0},
        "ttft_engine_ms": 200.0,
        "ttft_user_ms": 400.0,
        "total_ms": 1000.0,
        "sentences_checked": 2,
        "sentences_removed": 1,
        "refused": False,
        "tokens": 50,
        "tokens_per_s": 40.0,
    } | fields


ANSWER = sse(
    ("status", {"stage": "retrieve", "result": "5 evidence", "latency_ms": 30.0}),
    (
        "sentence",
        {"text": "A [1].", "citations": [], "check": {"status": "ok", "latency_ms": 70.0}},
    ),
    ("status", {"stage": "generate", "result": "removed: unsupported", "latency_ms": 60.0}),
    ("status", {"stage": "generate", "result": "2 checked, 1 removed", "latency_ms": 900.0}),
    ("done", done()),
)


def fake_api(body_for, delay=0.0, active=None):
    async def handler(request):
        query = json.loads(request.content)["query"]
        if active is not None:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(delay)
        if active is not None:
            active["now"] -= 1
        return httpx.Response(200, text=body_for(query), headers={"X-Request-ID": f"r-{query}"})

    return httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(handler))


def test_run_one_parses_the_stream():
    async def scenario():
        async with fake_api(lambda q: ANSWER) as client:
            return await run_one(client, "x")

    record = asyncio.run(scenario())
    assert record["status"] == "ok" and record["request_id"] == "r-x"
    assert record["check_ms"] == 130.0
    assert record["tokens"] == 50 and record["ttft_user_ms"] == 400.0
    assert record["stage_latency_ms"]["retrieve"] == 30.0
    assert record["first_sentence_ms"] is not None
    assert "text" not in record and "answer" not in record


@pytest.mark.parametrize(
    ("body", "status", "error"),
    [
        (sse(("error", {"code": "timeout", "message": "slow"})), "timeout", "timeout"),
        (sse(("error", {"code": "internal", "message": "x"})), "error", "internal"),
        (sse(("done", done(refused=True, stage_latency_ms={}))), "refused", None),
        (
            sse(("status", {"stage": "plan", "result": "S1", "latency_ms": 1.0})),
            "error",
            "stream ended without done",
        ),
    ],
)
def test_run_one_classifies_the_outcome(body, status, error):
    async def scenario():
        async with fake_api(lambda q: body) as client:
            return await run_one(client, "x")

    record = asyncio.run(scenario())
    assert record["status"] == status
    assert record.get("error") == error
    if status == "timeout":
        assert record["message"] == "slow"


def test_run_one_records_transport_errors():
    def handler(request):
        raise httpx.ConnectError("refused")

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="http://api", transport=transport) as client:
            return await run_one(client, "x")

    record = asyncio.run(scenario())
    assert record["status"] == "error" and record["error"].startswith("ConnectError")


def test_levels_keep_concurrency_and_use_each_question_once():
    active = {"now": 0, "max": 0}
    peaks, records, seen = {}, [], []

    def body_for(query):
        seen.append(query)
        return ANSWER

    def write(record):
        records.append(record)
        if record["kind"] == "level":
            peaks[record["level"]] = active["max"]
            active["max"] = 0

    warmup, sets = pick(QUERIES, [1, 2, 4], requests=4, warmup=2, seed=1789)

    async def scenario():
        async with fake_api(body_for, delay=0.01, active=active) as client:
            await load_test(client, [1, 2, 4], warmup, sets, write)

    asyncio.run(scenario())
    assert peaks == {1: 1, 2: 2, 4: 4}
    assert len(seen) == len(set(seen)) == 14
    assert sum(r["kind"] == "warmup" for r in records) == 2

    rows = summarize(records)
    assert [row["level"] for row in rows] == [1, 2, 4]
    row = rows[0]
    assert row["requests"] == row["ok"] == 4 and row["errors"] == 0
    assert row["total_p50_ms"] > 0 and row["requests_per_min"] > 0
    assert row["ttft_user_p50_ms"] == 400.0 and row["check_p50_ms"] == 130.0
    assert row["retrieve_p50_ms"] == 30.0 and row["generate_p50_ms"] == 900.0


def test_summary_counts_failures_and_skips_unfinished_levels():
    def request(level, status, **extra):
        return {"kind": "request", "run": "a", "level": level, "id": "q", "status": status} | extra

    ok = {"total_client_ms": 1000.0, "tokens": 60, "stage_latency_ms": {"generate": 800.0}}
    records = [
        request(1, "ok", **ok),
        request(1, "timeout"),
        request(1, "refused", tokens=0, stage_latency_ms={}),
        {"kind": "level", "run": "a", "level": 1, "wall_s": 30.0},
        request(2, "ok", **ok),
    ]
    one, two = summarize(records)
    assert (one["ok"], one["timeouts"], one["refused"], one["errors"]) == (1, 1, 1, 0)
    assert one["requests_per_min"] == 6.0 and one["tokens_per_s_sum"] == 2.0
    assert two["requests_per_min"] is None and two["total_p50_ms"] == 1000.0
    assert list(one) == list(two)


def test_pick_is_seeded_and_disjoint():
    warmup, sets = pick(QUERIES, [1, 2], requests=5, warmup=2, seed=1789)
    assert (warmup, sets) == pick(QUERIES, [1, 2], requests=5, warmup=2, seed=1789)
    ids = [q["id"] for q in warmup + sets[1] + sets[2]]
    assert len(ids) == len(set(ids)) == 12
    with pytest.raises(ValueError):
        pick(QUERIES, [1, 2, 4, 8], requests=24, warmup=2, seed=1789)


@pytest.mark.parametrize(
    "content",
    [
        '{"id": "a"}',
        '{"id": 1, "query": "x"}',
        '{"id": "a", "query": "x"}\n{"id": "a", "query": "y"}',
    ],
)
def test_load_queries_rejects_bad_files(tmp_path, content):
    path = tmp_path / "queries.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(path)
