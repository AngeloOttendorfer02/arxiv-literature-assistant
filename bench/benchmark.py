"""Engine benchmark per Ollama model tag: VRAM, TTFT and tokens/s (A-10).

Talks to Ollama's native API directly, separate from the app (E11). Reads LLM_BASE_URL
(with or without /v1) and LLM_API_KEY, sent as bearer token for the Modal proxy.
Records go to --output as JSON lines, one per request, including the answer text for
quality scoring. Real RAG prompts contain LitSearch text: keep their output in eval/private/.
"""

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import find_dotenv, load_dotenv


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if not isinstance(item.get("id"), str) or not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{number}: expected an object with id and messages")
        prompts.append(item)
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError(f"{path}: prompt ids must be unique")
    return prompts


def bust_cache(messages: list[dict], tag: str) -> list[dict]:
    """Ollama reuses the KV cache of a shared prompt prefix; a unique first token avoids it."""
    first, *rest = messages
    return [{**first, "content": f"[{tag}] {first['content']}"}, *rest]


def run_one(client: httpx.Client, model: str, messages: list[dict], options: dict) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": options,
    }
    start = time.perf_counter()
    first = None
    parts = []
    final = None
    with client.stream("POST", "/api/chat", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            content = chunk.get("message", {}).get("content", "")
            if content and first is None:
                first = time.perf_counter()
            parts.append(content)
            if chunk.get("done"):
                final = chunk
    end = time.perf_counter()
    if final is None:
        raise RuntimeError("stream ended without a done chunk")
    eval_s = final.get("eval_duration", 0) / 1e9
    return {
        # Engine: model load plus prompt processing, measured by Ollama, no network.
        "ttft_engine_ms": (final.get("load_duration", 0) + final.get("prompt_eval_duration", 0))
        / 1e6,
        "ttft_client_ms": None if first is None else (first - start) * 1000,
        "total_client_ms": (end - start) * 1000,
        "load_ms": final.get("load_duration", 0) / 1e6,
        "prompt_tokens": final.get("prompt_eval_count", 0),
        "output_tokens": final.get("eval_count", 0),
        "tokens_per_s": final.get("eval_count", 0) / eval_s if eval_s else None,
        "done_reason": final.get("done_reason"),
        "answer": "".join(parts),
    }


def memory(client: httpx.Client, model: str) -> dict:
    for entry in client.get("/api/ps").raise_for_status().json().get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            size, vram = entry["size"], entry["size_vram"]
            return {"size_gb": size / 1e9, "vram_gb": vram / 1e9, "cpu_gb": (size - vram) / 1e9}
    return {}


def percentile(values: list[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def _values(records: list[dict], key: str) -> list[float]:
    return [r[key] for r in records if r.get(key) is not None]


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for model in dict.fromkeys(r["model"] for r in records):
        own = [r for r in records if r["model"] == model and r["kind"] == "request"]
        ok = [r for r in own if "error" not in r]
        mem = next((r for r in records if r["model"] == model and r["kind"] == "memory"), {})
        ttft_engine = _values(ok, "ttft_engine_ms")
        ttft_client = _values(ok, "ttft_client_ms")
        rows.append(
            {
                "model": model,
                "requests": len(own),
                "errors": len(own) - len(ok),
                "vram_gb": mem.get("vram_gb"),
                "cpu_gb": mem.get("cpu_gb"),
                "ttft_engine_p50_ms": percentile(ttft_engine, 50),
                "ttft_engine_p95_ms": percentile(ttft_engine, 95),
                "ttft_client_p50_ms": percentile(ttft_client, 50),
                "ttft_client_p95_ms": percentile(ttft_client, 95),
                "tokens_per_s_p50": percentile(_values(ok, "tokens_per_s"), 50),
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    columns = list(rows[0]) if rows else []
    print("\t".join(columns))
    for row in rows:
        print(
            "\t".join(
                "" if v is None else f"{v:.1f}" if isinstance(v, float) else str(v)
                for v in row.values()
            )
        )


def benchmark(client, models, prompts, *, repeats, warmup, options, write) -> None:
    run = uuid.uuid4().hex[:8]
    for model in models:
        for i, prompt in enumerate(prompts[:warmup]):
            # Loads the model; the first load can take minutes.
            result = run_one(client, model, bust_cache(prompt["messages"], f"{run}-w{i}"), options)
            write({"kind": "warmup", "model": model, "load_ms": result["load_ms"]})
        write({"kind": "memory", "model": model, **memory(client, model)})
        for repeat in range(repeats):
            for prompt in prompts:
                record = {"kind": "request", "model": model, "repeat": repeat, "id": prompt["id"]}
                tag = f"{run}-{repeat}-{prompt['id']}"
                try:
                    record |= run_one(client, model, bust_cache(prompt["messages"], tag), options)
                except (httpx.HTTPError, RuntimeError) as error:
                    record["error"] = str(error)
                write(record)
                print(
                    f"{model} repeat {repeat} {prompt['id']}",
                    record.get("error", "ok"),
                    file=sys.stderr,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompts", type=Path, help="JSON lines with id and messages")
    parser.add_argument("--models", help="comma-separated Ollama tags, e.g. qwen3.8:27b-q4_K_M")
    parser.add_argument("--output", type=Path, required=True, help="JSON lines, appended")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2, help="prompts sent before measuring")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=8192, help="same context for every tag")
    parser.add_argument("--summary-only", action="store_true", help="summarize --output only")
    args = parser.parse_args()

    if not args.summary_only:
        if not args.models:
            parser.error("--models is required unless --summary-only is set")
        load_dotenv(find_dotenv(usecwd=True))
        base = os.environ["LLM_BASE_URL"].rstrip("/").removesuffix("/v1")
        key = os.environ.get("LLM_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        options = {
            "temperature": 0,
            "seed": 0,
            "num_predict": args.max_tokens,
            "num_ctx": args.num_ctx,
        }
        prompts = load_prompts(args.prompts)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        timeout = httpx.Timeout(900, connect=600)  # Modal cold start plus model load
        with (
            httpx.Client(base_url=base, headers=headers, timeout=timeout) as client,
            args.output.open("a", encoding="utf-8") as out,
        ):

            def write(record):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

            models = [m.strip() for m in args.models.split(",") if m.strip()]
            benchmark(
                client,
                models,
                prompts,
                repeats=args.repeats,
                warmup=args.warmup,
                options=options,
                write=write,
            )
    lines = args.output.read_text(encoding="utf-8").splitlines()
    print_summary(summarize([json.loads(line) for line in lines if line.strip()]))


if __name__ == "__main__":
    main()
