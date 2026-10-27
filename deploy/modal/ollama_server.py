"""Ollama inference endpoint on Modal for SciLitRAG (task A-09).

Ollama listens on 127.0.0.1 inside the container only; a FastAPI proxy in front of it
enforces a bearer token and forwards both the OpenAI-compatible /v1 API and the native /api.
"""

import hmac
import os
import subprocess
import time

import modal

OLLAMA_VERSION = "0.40.5"
OLLAMA_URL = "http://127.0.0.1:11434"
MODELS_DIR = "/models"
# D038: one GPU type for all precisions; bf16 of qwen3.8:27b needs 56 GB.
GPU = "A100-80GB"
# 1 for the precision benchmark (A-12), so VRAM covers one context; deploy with 8 for the
# load test (A-13, D043): OLLAMA_NUM_PARALLEL=8 modal deploy ollama_server.py
NUM_PARALLEL = os.environ.get("OLLAMA_NUM_PARALLEL", "1")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "pciutils", "zstd")
    .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh")
    .pip_install("fastapi==0.118.0", "httpx==0.28.1")
    .env(
        {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_MODELS": MODELS_DIR,
            "OLLAMA_KEEP_ALIVE": "-1",
            # one model in VRAM at a time so precision benchmarks do not share memory
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": NUM_PARALLEL,
        }
    )
)

models_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)
auth_secret = modal.Secret.from_name("ollama-auth")

app = modal.App("scilitrag-ollama", image=image)


def _start_ollama() -> subprocess.Popen:
    import httpx

    proc = subprocess.Popen(["ollama", "serve"])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{OLLAMA_URL}/api/version", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError("ollama serve did not become ready within 60 s")


def _default_no_think(path: str, body: bytes) -> bytes:
    """Disable model thinking unless the client set it explicitly."""
    import json

    try:
        payload = json.loads(body)
    except ValueError:
        return body
    if not isinstance(payload, dict):
        return body
    if path in ("api/chat", "api/generate"):
        payload.setdefault("think", False)
    elif path == "v1/chat/completions":
        if "reasoning_effort" not in payload and "reasoning" not in payload:
            payload["reasoning_effort"] = "none"
    elif path == "v1/responses":
        payload.setdefault("reasoning", {"effort": "none"})
    else:
        return body
    return json.dumps(payload).encode()


@app.cls(
    gpu=GPU,
    volumes={MODELS_DIR: models_volume},
    secrets=[auth_secret],
    scaledown_window=300,
    max_containers=1,
    # per request; caps a runaway generation instead of billing the GPU for up to an hour
    timeout=600,
)
@modal.concurrent(max_inputs=16)
class Ollama:
    @modal.enter()
    def start(self):
        self.started_at = time.time()
        self.proc = _start_ollama()

    @modal.exit()
    def stop(self):
        self.proc.terminate()

    @modal.asgi_app()
    def serve(self):
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.background import BackgroundTask

        api_key = os.environ["LLM_API_KEY"].encode()
        client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=httpx.Timeout(None, connect=5))
        hop_by_hop = {"host", "authorization", "content-length", "connection", "transfer-encoding"}
        web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @web.get("/healthz")
        async def healthz():
            return {"status": "ok", "gpu": GPU, "container_started_at": self.started_at}

        @web.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
        async def proxy(path: str, request: Request):
            given = request.headers.get("authorization", "").removeprefix("Bearer ").encode()
            if not hmac.compare_digest(given, api_key):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            body = await request.body()
            if request.method == "POST" and body:
                body = _default_no_think(path, body)
            upstream = client.build_request(
                request.method,
                f"/{path}",
                params=request.query_params,
                headers={k: v for k, v in request.headers.items() if k.lower() not in hop_by_hop},
                content=body,
            )
            resp = await client.send(upstream, stream=True)
            return StreamingResponse(
                resp.aiter_raw(),
                status_code=resp.status_code,
                headers={k: v for k, v in resp.headers.items() if k.lower() not in hop_by_hop},
                background=BackgroundTask(resp.aclose),
            )

        return web


@app.function(volumes={MODELS_DIR: models_volume}, timeout=3600)
def pull(models: str):
    """Pull comma-separated Ollama tags into the volume (CPU only, no GPU cost)."""
    proc = _start_ollama()
    try:
        for tag in (m.strip() for m in models.split(",") if m.strip()):
            print(f"pulling {tag}")
            subprocess.run(["ollama", "pull", tag], check=True)
        subprocess.run(["ollama", "list"], check=True)
    finally:
        proc.terminate()
    models_volume.commit()


@app.function(gpu=GPU, timeout=120)
def gpu_info():
    subprocess.run(["nvidia-smi"], check=True)
