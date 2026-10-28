# Ollama endpoint on Modal (A-09)

Ollama on a Modal A100 80 GB (D038) for the precision benchmark and the load test. The model weights live in the Volume `ollama-models`, and a bearer-token proxy sits in front of Ollama. Ollama itself only listens on 127.0.0.1 inside the container.

The Modal CLI is not part of the uv lock; `uvx` runs a pinned version in its own environment. Run the commands from the project root.

## Setup (one time)

```
uvx modal@1.6.2 token set --token-id <id> --token-secret <secret> --profile <workspace> --activate
uvx modal@1.6.2 secret create ollama-auth LLM_API_KEY=<random token>
uvx modal@1.6.2 run deploy/modal/ollama_server.py::pull --models "qwen3.8:27b-bf16,qwen3.8:27b-q8_0,qwen3.8:27b-q4_K_M"
uvx modal@1.6.2 deploy deploy/modal/ollama_server.py
```

`pull` runs on CPU only and stores the D037 tags in the Volume (about 104 GB, 11 to 22 min per tag).

## Connection settings

Put these in `deploy/modal/.env` (gitignored) for `client_check.py`, and in the project `.env` when the benchmark should use Modal:

```
LLM_BASE_URL=https://<workspace>--scilitrag-ollama-ollama-serve.us-east.modal.run/v1
LLM_MODEL=qwen3.8:27b-q4_K_M
LLM_API_KEY=<token>
```

The app's LLM client and `bench/benchmark.py` send `LLM_API_KEY` as bearer token. The native Ollama API (`/api/chat`, `/api/ps`) is available under the same host. `GET /healthz` needs no token.

## NLI on the GPU

`POST /nli` runs the app's sentence check (`NLI_MODEL`, ModernCE-large-nli by default, D057) on the same A100, so it does not take 1 to 2 s per source on the laptop CPU (D072). The model loads on the first call and its weights are cached in the Volume. To use it, add to the project `.env`:

```
NLI_URL=https://<workspace>--scilitrag-ollama-ollama-serve.us-east.modal.run/nli
```

The app sends `LLM_API_KEY` as token and its `NLI_MODEL`; the server rejects a different model. Without `NLI_URL` the check runs on the CPU as before.

## Demo with docker compose

Set these in the project `.env`, then `docker compose up -d --build`:

```
API_LLM_BASE_URL=https://<workspace>--scilitrag-ollama-ollama-serve.us-east.modal.run/v1
API_LLM_API_KEY=<token>
API_NLI_URL=https://<workspace>--scilitrag-ollama-ollama-serve.us-east.modal.run/nli
LLM_MODEL=qwen3.8:27b-q4_K_M
```

With a token, `/health` reports the LLM as "not probed" instead of calling it, so the compose healthcheck does not keep the GPU container running. The first request after an idle period includes the cold start and model load (about 1 min in the A-09 check). Stop the app after the demo.

## Check

`uv run --group rag python deploy/modal/client_check.py [--cold] [--model TAG]` checks that a request without token gets 401, OpenAI streaming with TTFT, a native call with tokens/s, and VRAM use and CPU offload from `/api/ps`. The first call includes loading the model.

## Benchmark and load test

- Only one model is loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`), so the precisions are measured one after another.
- `OLLAMA_NUM_PARALLEL` is 1 by default for the precision benchmark (A-12).
- The load test (A-13) uses only `qwen3.8:27b-q4_K_M` with 8 slots (D043): `OLLAMA_NUM_PARALLEL=8 uvx modal@1.6.2 deploy deploy/modal/ollama_server.py`.

## Costs and limits

- An A100 80 GB costs about 2.50 USD/h (0.000694 USD/s, modal.com/pricing, read 2026-10-27), and only while a container is running.
- Volume storage is free up to 1 TiB per month.
- The container scales down after 5 min idle; `max_containers=1`.
- Stop the app when done: `uvx modal@1.6.2 app stop scilitrag-ollama`.
- Set a spending limit in the Modal dashboard.
