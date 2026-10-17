# SciLitRAG

A locally hosted literature assistant for ML and NLP papers. It answers questions with hybrid retrieval (dense vectors, BM25 and SQL over citation and author data), backs every statement with a deterministic citation, and blocks unsafe requests before any database or LLM call.

Developed by Angelo Ottendorfer, Benjamin Storck, Kristof Zörenyi and Nuray Visne.

## Dataset

The project uses [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) ([princeton-nlp/LitSearch](https://github.com/princeton-nlp/LitSearch)). The dataset is **not** part of this repository and must not be redistributed. Database dumps are shared privately within the team only.

TODO: data loading, fixture and dump restore.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.13 (uv installs it if missing).

```
uv sync
```

This installs the base dependencies and the `dev` group. Work package groups are installed on demand, for example:

```
uv sync --group data
```

Groups: `data`, `rag`, `guardrails`, `ui`, `bench`, `dev`.

Configuration is read from environment variables and from a `.env` file in the project root. Copy the template and adjust it:

```
cp .env.example .env
```

## Run the API with mocks

With `USE_MOCKS=1` (set in `.env.example`) the app runs without database, model or guardrails.

```
uv run uvicorn scilitrag.api.app:create_app --factory --reload
```

Check: http://127.0.0.1:8000/health

`POST /query` with body `{"query": "..."}` returns a stream of server-sent events: `status`, `sentence`, `table`, `done`, `error`.

### Switching single components to real

`USE_MOCKS=0` uses only real components. With `USE_MOCKS=1`, `REAL_COMPONENTS` lists the components that use the real implementation while all others stay mocked:

```
USE_MOCKS=1
REAL_COMPONENTS=plan,retrieve
```

Names: `check_input`, `mask_pii`, `embed`, `plan`, `retrieve`, `sql_facts`, `generate`, `check_sentence`, `render_citations`. Unknown names stop the app at startup.

## Database

PostgreSQL 18 with pgvector and pg_search (ParadeDB image) runs in Docker. Credentials and port come from `.env`; the port is bound to localhost only.

```
docker compose up -d db
docker compose ps
docker compose exec db psql -U postgres -d scilitrag -c "\dx"
```

`\dx` lists the extensions `vector`, `pg_search` and `pg_trgm`. Stop with `docker compose down`; `docker compose down -v` also deletes the data volume.

TODO: schema, roles, fixture and dump restore.

## Run with Docker

```
docker compose up -d --build
```

Starts `db` and `api` (http://127.0.0.1:8000/health). The API container uses the settings from `.env`, but reaches the database as `db` and Ollama on the host as `host.docker.internal`.

Ollama runs on the host, outside compose. Keep it bound to localhost (default `127.0.0.1:11434`); in the Ollama app, "Expose Ollama to the network" must be off. Ollama has no authentication.

## Tests

```
uv run pytest
uv run python scripts/smoke.py
uv run ruff check .
uv run ruff format --check .
```

The smoke test runs the scenarios S1 to S4 and attacks in-process, no server needed. It uses the mocks except for the components in `REAL_COMPONENTS`, e.g. `REAL_COMPONENTS=retrieve uv run python scripts/smoke.py` tests a real `retrieve` with everything else mocked.

## Layout

```
src/scilitrag/
  api/          FastAPI app and request pipeline
  ui/           Streamlit app (talks to the API over HTTP only)
  rag/          planner, retrieval, SQL facts, generation, citations
  guardrails/   input checks, PII masking, sentence validation
  data/         loading, cleaning, chunking, indexing
  contracts/    shared data types and interfaces
  mocks/        mock implementation of every interface
  config.py     environment configuration
  components.py selects mocks or real components
eval/           evaluation scripts
bench/          quantization benchmark
docker/         container configuration
scripts/        helper scripts (data download, smoke test)
tests/          pytest tests
docs/           decisions and diagrams
```

The UI imports nothing from the package except `scilitrag.contracts`.

## Decisions

See [docs/decisions.md](docs/decisions.md).
