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
- Decision: Python 3.12.
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
