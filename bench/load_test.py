"""Load test of the whole app at 1, 2, 4 and 8 concurrent requests (A-13, D043).

Sends LitSearch questions to POST /query of a running API and reads the SSE stream, so
DB, NLI and LLM all take part. Records go to --output as JSON lines, one per request, with
numbers and ids only, no answer text. Each question is used once, so Ollama's prompt cache
cannot shorten the time to the first token.
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx
from benchmark import percentile, print_summary

from scilitrag.contracts import SENTENCE_REMOVED

DONE_FIELDS = (
    "total_ms",
    "ttft_engine_ms",
    "ttft_user_ms",
    "tokens",
    "tokens_per_s",
    "sentences_checked",
    "sentences_removed",
    "refused",
    "stage_latency_ms",
)


def load_queries(path: Path) -> list[dict]:
    queries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not all(isinstance(item.get(key), str) for key in ("id", "query")):
            raise ValueError(f"{path}:{number}: expected an object with id and query")
        queries.append({"id": item["id"], "query": item["query"]})
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError(f"{path}: query ids must be unique")
    return queries


def pick(queries: list[dict], levels: list[int], requests: int, warmup: int, seed: int):
    """Disjoint questions for the warm-up and for each level."""
    needed = warmup + len(levels) * requests
    if needed > len(queries):
        raise ValueError(f"need {needed} queries, file has {len(queries)}")
    chosen = random.Random(seed).sample(queries, needed)
    sets = {
        n: chosen[warmup + i * requests : warmup + (i + 1) * requests] for i, n in enumerate(levels)
    }
    return chosen[:warmup], sets


async def sse_events(lines):
    name, data = None, []
    async for line in lines:
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and name:
            yield name, json.loads("\n".join(data))
            name, data = None, []
    if name:
        yield name, json.loads("\n".join(data))


async def run_one(client: httpx.AsyncClient, query: str) -> dict:
    start = time.perf_counter()
    record = {"status": "error", "check_ms": 0.0, "first_sentence_ms": None}
    try:
        async with client.stream("POST", "/query", json={"query": query}) as response:
            record["request_id"] = response.headers.get("x-request-id")
            if response.status_code != 200:
                record["error"] = f"http {response.status_code}"
                return record
            async for name, event in sse_events(response.aiter_lines()):
                if name == "sentence":
                    if record["first_sentence_ms"] is None:
                        record["first_sentence_ms"] = (time.perf_counter() - start) * 1000
                    record["check_ms"] += event["check"]["latency_ms"]
                elif name == "status" and event["result"].startswith(SENTENCE_REMOVED):
                    record["check_ms"] += event["latency_ms"]
                elif name == "done":
                    record |= {k: event.get(k) for k in DONE_FIELDS}
                    record["status"] = "refused" if event.get("refused") else "ok"
                elif name == "error":
                    record["status"] = "timeout" if event["code"] == "timeout" else "error"
                    record["error"] = event["code"]
                    # Fixed text set by the app, it does not repeat the query.
                    record["message"] = event.get("message")
            if record["status"] == "error" and "error" not in record:
                record["error"] = "stream ended without done"
    except httpx.HTTPError as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["total_client_ms"] = (time.perf_counter() - start) * 1000
    return record


async def run_level(client, level: int, queries: list[dict], run: str, write) -> None:
    slots = asyncio.Semaphore(level)

    async def one(item):
        async with slots:
            record = await run_one(client, item["query"])
        write({"kind": "request", "run": run, "level": level, "id": item["id"], **record})
        print(f"level {level} {item['id']}", record["status"], file=sys.stderr)

    start = time.perf_counter()
    await asyncio.gather(*(one(item) for item in queries))
    write({"kind": "level", "run": run, "level": level, "wall_s": time.perf_counter() - start})


async def load_test(client, levels, warmup_queries, level_queries, write) -> None:
    run = uuid.uuid4().hex[:8]
    for item in warmup_queries:
        record = await run_one(client, item["query"])
        write({"kind": "warmup", "run": run, "id": item["id"], "status": record["status"]})
    for level in levels:
        await run_level(client, level, level_queries[level], run, write)


def _p(records, key, q):
    return percentile([r[key] for r in records if r.get(key) is not None], q)


def summarize(records: list[dict]) -> list[dict]:
    requests = [r for r in records if r["kind"] == "request"]
    walls = {(r["run"], r["level"]): r["wall_s"] for r in records if r["kind"] == "level"}
    stages = sorted({s for r in requests for s in r.get("stage_latency_ms") or {}})
    rows = []
    for level in sorted({r["level"] for r in requests}):
        own = [r for r in requests if r["level"] == level]
        ok = [r for r in own if r["status"] == "ok"]
        # Only finished levels have a wall time; throughput needs it.
        wall = sum(w for (run, n), w in walls.items() if n == level)
        done = [r for r in own if (r["run"], level) in walls]
        row = {
            "level": level,
            "requests": len(own),
            "ok": len(ok),
            "refused": sum(r["status"] == "refused" for r in own),
            "timeouts": sum(r["status"] == "timeout" for r in own),
            "errors": sum(r["status"] == "error" for r in own),
            "total_p50_ms": _p(ok, "total_client_ms", 50),
            "total_p95_ms": _p(ok, "total_client_ms", 95),
            "ttft_user_p50_ms": _p(ok, "ttft_user_ms", 50),
            "ttft_user_p95_ms": _p(ok, "ttft_user_ms", 95),
            "ttft_engine_p50_ms": _p(ok, "ttft_engine_ms", 50),
            "ok_per_min": sum(r["status"] == "ok" for r in done) / wall * 60 if wall else None,
            "tokens_per_s_sum": sum(r.get("tokens") or 0 for r in done) / wall if wall else None,
            "tokens_per_s_p50": _p(ok, "tokens_per_s", 50),
            "check_p50_ms": _p(ok, "check_ms", 50),
        }
        for stage in stages:
            values = [r["stage_latency_ms"][stage] for r in ok if stage in r["stage_latency_ms"]]
            row[f"{stage}_p50_ms"] = statistics.median(values) if values else None
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", type=Path, help="JSON lines with id and query")
    parser.add_argument("--output", type=Path, required=True, help="JSON lines, appended")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--levels", default="1,2,4,8", help="comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=24, help="requests per level")
    parser.add_argument("--warmup", type=int, default=2, help="requests sent before measuring")
    parser.add_argument("--seed", type=int, default=1789)
    parser.add_argument("--summary-only", action="store_true", help="summarize --output only")
    args = parser.parse_args()

    if not args.summary_only:
        levels = [int(n) for n in args.levels.split(",") if n.strip()]
        warmup, sets = pick(
            load_queries(args.queries), levels, args.requests, args.warmup, args.seed
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)

        async def go():
            # The API ends a request itself after REQUEST_TIMEOUT_S; this only guards a hang.
            timeout = httpx.Timeout(900, connect=10)
            limits = httpx.Limits(max_connections=max(levels))
            with args.output.open("a", encoding="utf-8") as out:

                def write(record):
                    out.write(json.dumps(record) + "\n")
                    out.flush()

                async with httpx.AsyncClient(
                    base_url=args.api_url, timeout=timeout, limits=limits
                ) as client:
                    await load_test(client, levels, warmup, sets, write)

        asyncio.run(go())
    lines = args.output.read_text(encoding="utf-8").splitlines()
    print_summary(summarize([json.loads(line) for line in lines if line.strip()]))


if __name__ == "__main__":
    main()
