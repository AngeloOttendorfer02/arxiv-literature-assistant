# Decisions

One entry per decision, newest first. Changes to a decision get a new entry that references the old one.

## Template

### DNNN Title

- Context:
- Options:
- Decision:
- Reason:
- Date:

---

### D014 Database image

- Context: E5 fixes PostgreSQL with pgvector and pg_search (ParadeDB) in Docker; A-01 needs a concrete image.
- Options: `paradedb/paradedb:0.26.2-pg18`, `paradedb/paradedb:0.26.2-pg17`, plain Postgres with self-built extensions.
- Decision: `paradedb/paradedb:0.26.2-pg18`, pinned to the exact tag. Port bound to 127.0.0.1, data in a named volume.
- Reason: Latest release (2026-10-16). Checked in the container: PostgreSQL 18.6, pg_search 0.26.2 and pgvector 0.8.6 preinstalled, pg_trgm 1.6 available. No own Dockerfile needed.
- Date: 2026-10-17

### D013 Python 3.13 (replaces D001)

- Context: The team had agreed on Python 3.13; the setup used 3.12.
- Options: stay on 3.12, switch to 3.13.
- Decision: Python 3.13, switched before the work packages start.
- Reason: Team decision. All planned dependencies support 3.13 (checked on PyPI 2026-10-17: torch 2.14, spaCy 3.8 with thinc 8.3 ship 3.13 wheels; presidio-analyzer, transformers, sentence-transformers, psycopg and streamlit allow 3.13).
- Date: 2026-10-17

### D012 Development LLM without GPU

- Context: Team members without a GPU need a local model for development; the final model is chosen later (O6).
- Options: qwen3.5:2b-q4_K_M, qwen3.5:2b-mtp-q4_K_M.
- Decision: `qwen3.5:2b-mtp-q4_K_M` via Ollama as `LLM_MODEL` for CPU-only development.
- Reason: Measured on CPU (i7-14700K, `num_gpu=0`, 3 prompts with 2000 output tokens each): 34.2 vs 31.7 tokens/s median, same output on a short RAG prompt.
- Date: 2026-10-17

### D011 Embedding model candidates

- Context: E8 proposes bge-base-en-v1.5. The choice fixes the vector dimension in the schema, so it must be final before the embeddings are built (D-14, day 6).
- Options: bge-base-en-v1.5, granite-embedding-english-r2, cadet-embed-base-v1, all-MiniLM-L6-v2, larger models (stella_en_400M_v5, Qwen3-Embedding-0.6B).
- Decision: bge-base-en-v1.5 stays the default. Before D-14 it is compared with `ibm-granite/granite-embedding-english-r2` and `manveertamber/cadet-embed-base-v1` on the LitSearch queries (Recall@k, NDCG@10). All three have 768 dimensions, so a switch does not change the schema.
- Reason: MTEB results (embeddings-benchmark/results, read 2026-10-17, NDCG@10): base-size models differ by 1 to 3 points. granite-r2 beats bge-base on SCIDOCS (25.0 vs 21.7) and SciFact (75.8 vs 74.3); cadet is the only small model with a LitSearchRetrieval score (47.9), bge-base has none. all-MiniLM-L6-v2 is clearly weaker (SciFact 64.5, 256 tokens). Larger models are too heavy for CPU and pgvector.
- Date: 2026-10-17

### D010 Linting and formatting

- Context: Consistent style across four people.
- Options: ruff, black plus flake8.
- Decision: ruff for lint and format, default rules, line length 100.
- Reason: One fast tool, no configuration overhead.
- Date: 2026-10-17

### D009 Smoke test

- Context: The team works on Windows and macOS; `main` must always run.
- Options: Makefile target, Python script.
- Decision: `scripts/smoke.py`, runs the app in-process with FastAPI's TestClient. The same cases run in pytest. The TestClient uses httpx2 (`dev` group), since Starlette deprecates plain httpx for it.
- Reason: Works on every OS without `make` and without a running server.
- Date: 2026-10-17

### D008 Configuration

- Context: All settings come from environment variables (`DATABASE_URL`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, `USE_MOCKS`, `REAL_COMPONENTS`).
- Options: plain `os.environ`, pydantic-settings, python-dotenv plus `os.environ`.
- Decision: Small `config.py` with `os.environ` and defaults; `.env` is loaded with python-dotenv. Missing `DATABASE_URL`, `LLM_BASE_URL` or `LLM_MODEL` are an error only with `USE_MOCKS=0`. `USE_MOCKS` defaults to 0, so a misconfigured demo never runs on mocks silently.
- Reason: Minimal code; python-dotenv is already used by the data download script.
- Date: 2026-10-17

### D007 Server-sent events

- Context: `POST /query` streams events to the UI.
- Options: FastAPI native SSE (`fastapi.sse`), sse-starlette, plain `StreamingResponse`.
- Decision: `fastapi.sse.EventSourceResponse` (FastAPI >= 0.135).
- Reason: No extra dependency, Pydantic models as event data, keep-alive pings built in.
- Date: 2026-10-17

### D006 Sync and async interfaces

- Context: The API is async, most components are CPU-bound or use blocking clients.
- Options: all async, all sync, only generation async.
- Decision: `generate` returns an `AsyncIterator[str]`. All other interfaces are synchronous; the API calls them in the thread pool.
- Reason: Token streaming needs async; the other components stay simple to write and test.
- Date: 2026-10-17

### D005 Contracts

- Context: Work packages develop in parallel against shared interfaces.
- Options: dataclasses, Pydantic models, TypedDicts; abstract base classes or `typing.Protocol`.
- Decision: Pydantic v2 models for data, `typing.Protocol` for interfaces, `CONTRACT_VERSION = "0.1"`. Besides the interfaces in plan section 4.2: `embed` (shared by data and guardrails), `mask_pii` (answers and logs), per-stage results in `CheckInputResult.stages`.
- Reason: Validation and JSON serialization for the SSE events for free; protocols need no inheritance, so mocks and real modules stay independent.
- Date: 2026-10-17

### D004 Separation of API and UI

- Context: FastAPI and Streamlit run as separate services (docker-compose `api` and `ui`).
- Options: one package with an import rule, a separate top-level UI project.
- Decision: One package. `scilitrag.ui` talks to the API over HTTP only and imports nothing but `scilitrag.contracts`. Own dependency group `ui`.
- Reason: Event models are shared without duplication; the rule keeps the services deployable separately.
- Date: 2026-10-17

### D003 Package name and layout

- Context: The plan lists `src/api`, `src/contracts` etc. as top-level folders.
- Options: flat `src/*` packages, one package `src/scilitrag/*`.
- Decision: One package `scilitrag` under `src/`. `eval/`, `bench/`, `scripts/`, `tests/`, `docs/` stay at the top level.
- Reason: Unambiguous imports (`from scilitrag.contracts import Plan`); generic names like `api`, `data`, `ui` would clash with installed libraries.
- Date: 2026-10-17

### D002 Package manager and dependency groups

- Context: Some modules need heavy model dependencies (torch), others do not.
- Options: pip with requirements files, uv with `pyproject.toml`, Poetry.
- Decision: uv with `pyproject.toml` and `uv.lock`. Dependency groups per work package (`data`, `rag`, `guardrails`, `ui`, `bench`) plus `dev`.
- Reason: Reproducible lockfile, fast installs, nobody has to install torch for work that does not need it.
- Date: 2026-10-17

### D001 Python version

- Context: The team needs one interpreter version for all modules and containers.
- Options: 3.11, 3.12, 3.13.
- Decision: Python 3.12. Replaced by D013.
- Reason: Team decision. Library support for the model dependencies is checked when the groups are filled.
- Date: 2026-10-17

---

## Rejected approaches

### R002 Makefile for the smoke test

- Approach: `make e2e` as suggested in the plan (G-08).
- Why rejected: `make` is not available by default on Windows. Replaced by D009.
- Date: 2026-10-17

### R001 requirements.txt

- Approach: dependencies in `requirements.txt` (initial repo state).
- Why rejected: no lockfile, no separation of heavy model dependencies. Replaced by D002.
- Date: 2026-10-17
