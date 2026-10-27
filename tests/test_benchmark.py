import json

import pytest

httpx = pytest.importorskip("httpx")

from benchmark import benchmark, bust_cache, load_prompts, percentile, summarize

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def fake_ollama(seen):
    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": "m", "size": 3e9, "size_vram": 2e9}]}
            )
        body = json.loads(request.content)
        seen.append(body)
        chunks = [
            {"message": {"content": "Answer "}, "done": False},
            {"message": {"content": "[1]."}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
                "load_duration": 1_000_000,
                "prompt_eval_duration": 4_000_000,
                "prompt_eval_count": 50,
                "eval_count": 20,
                "eval_duration": 500_000_000,
            },
        ]
        return httpx.Response(200, text="\n".join(json.dumps(c) for c in chunks))

    return httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))


def test_benchmark_records_engine_times_memory_and_unique_prompts():
    seen, records = [], []
    prompts = [{"id": "a", "messages": MESSAGES}, {"id": "b", "messages": MESSAGES}]
    benchmark(
        fake_ollama(seen),
        ["m"],
        prompts,
        repeats=2,
        warmup=1,
        options={"temperature": 0},
        write=records.append,
    )
    requests = [r for r in records if r["kind"] == "request"]
    assert len(requests) == 4
    first = requests[0]
    assert first["ttft_engine_ms"] == 5.0
    assert first["tokens_per_s"] == 40.0
    assert first["answer"] == "Answer [1]."
    assert {
        "kind": "memory",
        "model": "m",
        "size_gb": 3.0,
        "vram_gb": 2.0,
        "cpu_gb": 1.0,
    } in records
    system_prompts = [body["messages"][0]["content"] for body in seen]
    assert len(set(system_prompts)) == len(seen) == 5
    assert all(body["think"] is False and body["stream"] for body in seen)

    (row,) = summarize(records)
    assert row["requests"] == 4 and row["errors"] == 0
    assert row["ttft_engine_p50_ms"] == 5.0 and row["vram_gb"] == 2.0


def test_failed_request_is_recorded_and_the_run_continues():
    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, text=json.dumps({"error": "model not found"}))

    records = []
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    prompts = [{"id": "a", "messages": MESSAGES}]
    benchmark(client, ["m"], prompts, repeats=2, warmup=0, options={}, write=records.append)
    requests = [r for r in records if r["kind"] == "request"]
    assert [r["error"] for r in requests] == ["model not found"] * 2
    assert summarize(records)[0]["errors"] == 2


def test_bust_cache_prefixes_the_first_message_without_mutating_it():
    busted = bust_cache(MESSAGES, "x1")
    assert busted[0]["content"] == "[x1] sys"
    assert busted[1] is MESSAGES[1]
    assert MESSAGES[0]["content"] == "sys"


@pytest.mark.parametrize(
    "content",
    [
        '{"id": "a"}',
        '{"id": 1, "messages": [1]}',
        '{"id": "a", "messages": [1]}\n{"id": "a", "messages": [1]}',
    ],
)
def test_load_prompts_rejects_bad_files(tmp_path, content):
    path = tmp_path / "prompts.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(path)


def test_example_prompts_load():
    from pathlib import Path

    prompts = load_prompts(Path(__file__).parents[1] / "bench/prompts.example.jsonl")
    assert len(prompts) == 30


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
