# SciLitRAG

A locally hosted literature assistant for ML and NLP papers. It answers questions with hybrid retrieval (dense vectors, BM25 and SQL over citation and author data), backs every statement with a deterministic citation, and blocks unsafe requests before any database or LLM call.

Developed by Angelo Ottendorfer, Benjamin Storck, Kristof Zörenyi and Nuray Visne.

## Dataset

The project uses [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) ([princeton-nlp/LitSearch](https://github.com/princeton-nlp/LitSearch)). The dataset is **not** part of this repository and must not be redistributed. Database dumps are shared privately within the team only.

TODO: full-corpus loading and dump restore.

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

### Chat UI

With the API running (locally or in Docker):

```
uv run --group ui streamlit run src/scilitrag/ui/app.py
```

Opens http://localhost:8501. The UI reaches the API at `API_URL` (default `http://localhost:8000`).

### Switching single components to real

`USE_MOCKS=0` uses only real components. With `USE_MOCKS=1`, `REAL_COMPONENTS` lists the components that use the real implementation while all others stay mocked:

```
USE_MOCKS=1
REAL_COMPONENTS=plan,render_citations
```

Names: `check_input`, `mask_pii`, `embed`, `plan`, `retrieve`, `sql_facts`, `generate`, `check_sentence`, `render_citations`. Unknown names stop the app at startup.

### Retrieval components

| Area | Responsibility |
| --- | --- |
| Model access | Configured completions, cancellable streaming, bounded retries and strict JSON validation. |
| Citations | Request-local evidence snapshots, label checks, deterministic rendering, source fingerprints and ledgers. |
| Planning and entities | Scenario planning with S1 fallback, explicit year/result filters and reader-injected paper/author linking. |

Install the `rag` dependency group when working on these modules:

```sh
uv sync --locked --group rag
```

The real planner requires `LLM_BASE_URL` and `LLM_MODEL`. It validates the model's scenario plan,
reconciles explicit year and result-count constraints, and falls back to topic search when the
provider is unavailable or its response fails validation.

`EntityLinker` accepts a synchronous reader that binds named SQL parameters and returns dictionary
rows. The adapter must provide compatible paper, author and paper-author tables and support
`pg_trgm` for fuzzy title matching. Ingestion and lookup must use the same name-normalization rule.
Reader test doubles exercise matching and ambiguity handling independently of a live database.

Citation rendering resolves request-local labels to source snapshots. Structural citation checks
establish label validity and placement; semantic validation determines factual support.

The selectable real entry points are `plan` and `render_citations`. Retrieval, SQL facts and
generation expose placeholder entry points; use their mock components until implementations
are available.

## Database

PostgreSQL 18 with pgvector and pg_search (ParadeDB image) runs in Docker. Credentials and port come from `.env`; the port is bound to localhost only.

```
docker compose up -d db
docker compose ps
docker compose exec db psql -U postgres -d scilitrag -c "\dx"
```

`\dx` lists the extensions `vector`, `pg_search` and `pg_trgm`. Stop with `docker compose down`; `docker compose down -v` also deletes the data volume.

### Schema and database roles (D-04 to D-06)

Add the four role variables from `.env.example` to your local `.env`, **before**
`DATABASE_URL`. Update `DATABASE_URL` and add `INGEST_DATABASE_URL` as in the template.
The API uses the app role (SELECT only); imports use ingest (SELECT, INSERT,
UPDATE, DELETE and USAGE on the author ID sequence).

Use distinct role names, different from `POSTGRES_USER`: lowercase ASCII letters,
digits and underscores, starting with a letter or underscore, at most 63 characters.
Replace the password placeholders with different strong passwords using only
ASCII letters, digits and `._~-`. Keep `.env` private.

With the db container healthy, apply the files as the schema-owning superuser. These commands
use the default administrator `postgres` and database `scilitrag`; adjust them if
needed:

```sh
docker compose exec db psql -X -U postgres -d scilitrag -f /opt/scilitrag/db/schema.sql
docker compose exec db psql -X -U postgres -d scilitrag -f /opt/scilitrag/db/indexes.sql
docker compose exec db psql -X -U postgres -d scilitrag -f /opt/scilitrag/db/roles.sql
```

Skip `schema.sql` if the tables exist and `roles.sql` if the roles are already
configured. Both are initial setup scripts and stop on existing tables or roles.
SQL files are not run automatically. Existing volumes can be kept; do not delete
one to apply roles. Changing passwords in `.env` does not change existing DB roles.
Further permission details are in [D024](docs/decisions.md#d024-d-06-separate-import-and-application-database-roles).

Check each login separately through host `db`; container loopback uses trust
authentication and does not check passwords. Enter the role's password when
prompted and exit with `\q`. A wrong password must be rejected; the correct one
must connect. Adjust role names if yours differ:

```sh
docker compose exec db psql -X -h db -U app -d scilitrag -W -P pager=off
docker compose exec db psql -X -h db -U ingest -d scilitrag -W -P pager=off
```

In each session, run:

```sql
SELECT current_user;
SELECT count(*) FROM public.papers;
INSERT INTO public.papers (corpus_id) SELECT 0::bigint WHERE false;
UPDATE public.papers SET year = year WHERE false;
DELETE FROM public.papers WHERE false;
```

Both roles can read. App writes must fail with permission denied; ingest must
return `INSERT 0 0`, `UPDATE 0`, `DELETE 0`. `WHERE false` checks permissions without
changing rows.

After bulk imports, the administrator runs `ANALYZE papers, authors, paper_authors,
citations, chunks;`. Ingest has no MAINTAIN privilege.

### Local test data (D-07)

After D-03, select a reproducible fixture from your own filtered data:

```sh
uv run --locked --group data python scripts/select_fixture.py
uv run --locked --group data python scripts/load_fixture.py --check-only
uv run --locked --group data --group rag python scripts/load_fixture.py
```

The default is 300 papers around 20 queries. Files stay under
`DATA_DIR/fixtures/v0`; no LitSearch records are committed. The import uses
`INGEST_DATABASE_URL` and requires empty data tables with a fresh author ID
sequence. It loads paper metadata, internal citations and authors; chunks and
embeddings follow in later tasks. To reload, reset the tables first as
administrator; a failed or rolled-back import also needs this reset:

```sh
docker compose exec db psql -X -U postgres -d scilitrag -c "TRUNCATE papers, authors, paper_authors, citations, chunks RESTART IDENTITY;"
```

See [fixture selection and loading](tests/fixtures/README.md) for details.

TODO: dump restore.

## Run with Docker

```
docker compose up -d --build
```

Starts `db`, `api` (http://127.0.0.1:8000/health) and `ui` (chat at http://127.0.0.1:8501). Use `--build` after code changes, otherwise the old images are reused. The API container uses the settings from `.env`, but reaches the database as `db` and Ollama on the host as `host.docker.internal`; the UI container reaches the API as `api`. Host ports can be changed with `DB_PORT`, `API_PORT` and `UI_PORT` in `.env`.

## Request log

The API writes one JSON line per query to stdout and to `logs/requests.jsonl` (folder set by `LOG_DIR`): request id, status (`ok`, `refused`, `error`, `timeout`, `cancelled`), latencies per stage, TTFT, tokens per second, guardrail events. The query text is never logged, only its length. The id is also sent as `X-Request-ID` and shown under each answer in the UI.

```
docker compose logs -f api
```

On Linux, create `logs/` before the first `docker compose up` (`mkdir logs`); a folder created by Docker belongs to root, and the API then logs to stdout only.

`GET /health` reports whether the database and the LLM are reachable (`skipped` while the component is mocked).

Ollama runs on the host, outside compose. Keep it bound to localhost (default `127.0.0.1:11434`); in the Ollama app, "Expose Ollama to the network" must be off. Ollama has no authentication.

## Tests

```sh
uv run --locked --group data --group rag pytest
uv run --locked --group data --group rag python scripts/smoke.py
uv run --locked --group data --group rag ruff check .
uv run --locked --group data --group rag ruff format --check .
```

The UI tests also require the `ui` group and are skipped without it:

```sh
uv run --locked --group data --group rag --group ui pytest
```

The Arrow integration tests for profiling and quality filtering require the
`data` group. Missing or broken PyArrow installations fail these tests rather
than skipping them. CI installs the `data` and `rag` groups and runs the Arrow
integration tests with small synthetic datasets, without downloading LitSearch.

Set `USE_MOCKS=1` and clear `REAL_COMPONENTS` for offline checks. The role-play
jailbreak case is an expected failure while the keyword input-check mock is
selected. Passing these checks does not establish live database compatibility
or measured model quality.

The smoke test runs scenarios S1 to S4 and attacks in-process, without a server.
It uses mocks except for components selected through `REAL_COMPONENTS`.
For example, `REAL_COMPONENTS=plan uv run --locked --group data --group rag python scripts/smoke.py`
exercises the real planner with the other components mocked. A successful
fallback plan does not prove that the model service was reached.

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
