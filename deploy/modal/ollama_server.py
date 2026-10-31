"""Ollama inference endpoint on Modal for SciLitRAG (task A-09).

Ollama listens on 127.0.0.1 inside the container only; a FastAPI proxy in front of it
enforces a bearer token and forwards both the OpenAI-compatible /v1 API and the native /api.
POST /nli runs the app's NLI check on the same GPU (D072), POST /guard Prompt Guard on the
retrieved sources (D079).
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
# Same checkpoint as NLI_MODEL in the app (D057); the app sends its id and a mismatch fails.
NLI_MODEL = os.environ.get("NLI_MODEL", "dleemiller/ModernCE-large-nli")
# Prompt Guard for the source check (D079). Gated on Hugging Face, so the weights are copied
# into the volume once (README) instead of giving the server a token.
GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
GUARD_DIR = f"{MODELS_DIR}/prompt-guard-2-86M"
GUARD_BATCH = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "pciutils", "zstd")
    .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh")
    .pip_install("fastapi==0.118.0", "httpx==0.28.1")
    # Same versions as uv.lock; the PyPI torch wheel for Linux includes CUDA.
    .pip_install("torch==2.14.1", "transformers==5.20.2")
    .env(
        {
            "OLLAMA_HOST": "127.0.0.1:11434",
            "OLLAMA_MODELS": MODELS_DIR,
            "OLLAMA_KEEP_ALIVE": "-1",
            # one model in VRAM at a time so precision benchmarks do not share memory
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": NUM_PARALLEL,
            "NLI_MODEL": NLI_MODEL,
            # NLI weights are cached in the volume next to the Ollama models.
            "HF_HOME": f"{MODELS_DIR}/hf",
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
        import threading

        self.started_at = time.time()
        self.proc = _start_ollama()
        self.nli = None
        self.nli_lock = threading.Lock()
        self.guard = None
        self.guard_lock = threading.Lock()

    def nli_scores(self, pairs: list[list[str]]) -> list[list[float]]:
        """Class probabilities in checkpoint order; the model loads on the first call."""
        with self.nli_lock:
            # Imported under the lock: concurrent first imports of transformers can fail.
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self.nli is None:
                tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
                model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
                self.nli = tokenizer, model.eval().to("cuda")
                models_volume.commit()
            tokenizer, model = self.nli
            premises, hypotheses = zip(*pairs, strict=True)
            inputs = tokenizer(
                list(premises), list(hypotheses), padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                return model(**inputs).logits.softmax(-1).tolist()

    def guard_scores(self, windows: list[list[int]]) -> list[float]:
        """Malicious probability per window; the app cuts the windows with its tokenizer."""
        with self.guard_lock:
            import torch
            from transformers import AutoModelForSequenceClassification

            if self.guard is None:
                model = AutoModelForSequenceClassification.from_pretrained(GUARD_DIR)
                self.guard = model.eval().to("cuda")
            model = self.guard
            pad = model.config.pad_token_id or 0
            scores = []
            for i in range(0, len(windows), GUARD_BATCH):
                batch = windows[i : i + GUARD_BATCH]
                width = max(len(w) for w in batch)
                ids = torch.tensor([w + [pad] * (width - len(w)) for w in batch], device="cuda")
                mask = torch.tensor(
                    [[1] * len(w) + [0] * (width - len(w)) for w in batch], device="cuda"
                )
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=mask).logits
                # Index 1 is "malicious", as in the app.
                scores.extend(logits.softmax(-1)[:, 1].tolist())
            return scores

    @modal.exit()
    def stop(self):
        self.proc.terminate()

    @modal.asgi_app()
    def serve(self):
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.background import BackgroundTask
        from starlette.concurrency import run_in_threadpool

        api_key = os.environ["LLM_API_KEY"].encode()
        client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=httpx.Timeout(None, connect=5))
        hop_by_hop = {"host", "authorization", "content-length", "connection", "transfer-encoding"}
        web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @web.get("/healthz")
        async def healthz():
            return {"status": "ok", "gpu": GPU, "container_started_at": self.started_at}

        def authorized(request: Request) -> bool:
            given = request.headers.get("authorization", "").removeprefix("Bearer ").encode()
            return hmac.compare_digest(given, api_key)

        @web.post("/nli")
        async def nli(request: Request):
            if not authorized(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            payload = await request.json()
            if payload.get("model") != NLI_MODEL:
                return JSONResponse({"error": f"server runs {NLI_MODEL}"}, status_code=400)
            pairs = payload.get("pairs") or []
            if not all(isinstance(p, list) and len(p) == 2 for p in pairs):
                return JSONResponse(
                    {"error": "pairs must be [premise, hypothesis]"}, status_code=400
                )
            probs = await run_in_threadpool(self.nli_scores, pairs) if pairs else []
            return {"model": NLI_MODEL, "probabilities": probs}

        @web.post("/guard")
        async def guard(request: Request):
            if not authorized(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            payload = await request.json()
            if payload.get("model") != GUARD_MODEL:
                return JSONResponse({"error": f"server runs {GUARD_MODEL}"}, status_code=400)
            windows = payload.get("windows") or []
            if not all(
                isinstance(w, list) and 0 < len(w) <= 512 and all(isinstance(t, int) for t in w)
                for w in windows
            ):
                return JSONResponse(
                    {"error": "windows must be lists of 1 to 512 token ids"}, status_code=400
                )
            scores = await run_in_threadpool(self.guard_scores, windows) if windows else []
            return {"model": GUARD_MODEL, "malicious": scores}

        @web.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
        async def proxy(path: str, request: Request):
            if not authorized(request):
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
