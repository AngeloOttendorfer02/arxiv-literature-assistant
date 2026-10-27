"""Check the Modal Ollama endpoint: auth, OpenAI streaming, native eval stats, VRAM, cold start.

Usage: python client_check.py [--cold] [--model TAG]
Reads LLM_BASE_URL (ending in /v1) and LLM_API_KEY from the environment or ./.env.
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cold", action="store_true", help="time the first request after scale-down"
    )
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.5:2b-mtp-q4_K_M"))
    args = parser.parse_args()

    load_env(Path(__file__).with_name(".env"))
    base = os.environ["LLM_BASE_URL"].rstrip("/").removesuffix("/v1")
    auth = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    client = httpx.Client(base_url=base, timeout=httpx.Timeout(600, connect=600))

    if args.cold:
        t0 = time.perf_counter()
        r = client.get("/healthz")
        print(f"cold start: container up after {time.perf_counter() - t0:.1f} s ({r.status_code})")

    r = client.get("/v1/models")
    print(f"no token: {r.status_code} (expect 401)")
    assert r.status_code == 401

    r = client.get("/v1/models", headers=auth)
    r.raise_for_status()
    print("models:", [m["id"] for m in r.json()["data"]])

    body = {
        "model": args.model,
        "stream": True,
        "messages": [{"role": "user", "content": "Name three uses of BM25 in one sentence each."}],
    }
    t0 = time.perf_counter()
    ttft = None
    text = []
    with client.stream("POST", "/v1/chat/completions", json=body, headers=auth) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and ttft is None:
                ttft = time.perf_counter() - t0
            text.append(delta)
    total = time.perf_counter() - t0
    print(
        f"openai stream: ttft {ttft:.2f} s, total {total:.2f} s (includes model load if first call)"
    )
    print("answer:", "".join(text)[:300].replace("\n", " "))

    r = client.post(
        "/api/generate",
        headers=auth,
        json={
            "model": args.model,
            "prompt": "Explain reciprocal rank fusion briefly.",
            "stream": False,
        },
    )
    r.raise_for_status()
    stats = r.json()
    tps = stats["eval_count"] / (stats["eval_duration"] / 1e9)
    print(
        f"native: eval_count {stats['eval_count']}, {tps:.1f} tokens/s, load {stats['load_duration'] / 1e9:.2f} s"
    )

    for m in client.get("/api/ps", headers=auth).json()["models"]:
        offload = (
            "none"
            if m["size_vram"] >= m["size"]
            else f"{(m['size'] - m['size_vram']) / 1e9:.2f} GB on CPU"
        )
        print(
            f"loaded {m['name']}: size {m['size'] / 1e9:.2f} GB, vram {m['size_vram'] / 1e9:.2f} GB, offload {offload}"
        )


if __name__ == "__main__":
    main()
