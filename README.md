# SciLitRAG

A locally hosted ML and NLP literature assistant with hybrid retrieval, citation-path exploration, author citation rankings and deterministic sentence-level citations. An asynchronous API streams checked sentences and SQL tables to a separate chat UI.

The default mock configuration runs offline. Real retrieval requires the shared embedding adapter and indexed corpus; real input and semantic guardrails still require integration. See [retrieval implementation and acceptance](docs/retrieval.md).

Developed by Angelo Ottendorfer, Benjamin Storck, Kristof Zörenyi and Nuray Visne.

## Dataset

The project uses [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) ([princeton-nlp/LitSearch](https://github.com/princeton-nlp/LitSearch)). The dataset is **not** part of this repository and must not be redistributed. Database dumps are shared privately within the team only.

The full database is shared as a dump; restoring or building it is described in
[Full database and dump (D-16)](#full-database-and-dump-d-16).

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

Names: `check_input`, `mask_pii`, `check_evidence`, `embed`, `plan`, `retrieve`, `sql_facts`, `generate`, `check_sentence`, `render_citations`. Unknown names stop the app at startup.

### Retrieval components

| Area | Responsibility |
| --- | --- |
| Model access | Configured completions, cancellable streaming, bounded retries and strict JSON validation. |
| Citations | Request-local evidence snapshots, label checks, deterministic rendering, source fingerprints and ledgers. |
| Planning and entities | Scenario planning with S1 fallback, explicit year/result filters and reader-injected paper/author linking. |
| Search and evidence | Filtered title/abstract search with pgvector/BM25, RRF k=60, scoped passages and explicit evidence coverage. |
| Citation paths | Incoming citation traversal with target-specific experimental baseline candidates. |
| Author rankings | SQL credits over the exact selected topic-paper set, with dataset revision, scope and examples. |
| Evaluation | Validated frozen references, pipeline model runs and separate structural/reference/human support metrics. |

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

The real entry points are `plan`, `retrieve`, `sql_facts`, `generate` and
`render_citations`. Real search requires frozen corpus metadata, deployed indexes
and the shared query embedder. Generation requires the configured model endpoint.
Select real retrieve and sql_facts together for S3; mock IDs are not real ranking seeds.
Input and semantic sentence validators must also be integrated before all-real evaluation.

Implementation details and acceptance requirements: [retrieval](docs/retrieval.md),
[evaluation](docs/evaluation.md) and [multi-hop execution](docs/execution_flow.md).

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
docker compose exec db psql -X -v ON_ERROR_STOP=1 -U postgres -d scilitrag -f /opt/scilitrag/db/retrieval_indexes.sql
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

### Retrieval readiness

BM25 indexes cover paper titles and abstracts plus chunk text (D-15). For existing
abstract-only indexes, first apply migration
`docker/db/migrations/002_bm25_title_abstract.sql` as the database owner, using
`psql -X -v ON_ERROR_STOP=1`. See [deployment and tests](docs/d15_bm25_index.md).

Run schema-only preflight with the configured application login:

```sh
uv run --locked --group rag python scripts/check_database.py
```

Use `--require-search` after the real data and embeddings are loaded and their
dataset/model metadata have been published. A missing-data response is a blocked
live check, not a failed unit test. Keep the frozen metadata tied to actual data.

`scripts/explain_retrieval.py` captures real query plans only when that handoff is
ready. Missing search prerequisites produce a controlled exit with no output file.
Optional integration tests use `RETRIEVAL_TEST_DATABASE_URL` for a disposable
ParadeDB database with an administrative test login; the read-only app login cannot
create the test schemas. Synthetic fixture rows are not the real project corpus.

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
sequence. It loads paper metadata, internal citations, authors and D-13 chunks;
`scripts/embed_database.py` (group `embed`) then adds the D-14 embeddings.
Before embedding or resetting an existing database, apply the additive fingerprint
migration as administrator (safe to re-run). New schemas already have the column:

```sh
docker compose exec db psql -X -U postgres -d scilitrag -f /opt/scilitrag/db/migrations/001_embedding_fingerprint.sql
```

To reload, reset the tables and embedding metadata as administrator; a failed or
rolled-back import also needs this reset:

```sh
docker compose exec db psql -X -U postgres -d scilitrag -c "TRUNCATE papers, authors, paper_authors, citations, chunks RESTART IDENTITY; UPDATE schema_metadata SET embed_model = NULL, embed_fingerprint = NULL WHERE id = 1;"
```

See [fixture selection and loading](tests/fixtures/README.md) for details.

### Full database and dump (D-16)

Usually you only restore the shared dump. Add the role variables to `.env` (see
above), start a fresh database volume and run:

```sh
docker compose up -d db
uv run --locked --group data python scripts/restore_database.py path/to/scilitrag_v2.dump
uv run --locked --group rag python scripts/check_database.py --require-search
```

The restore checks the `.sha256` file next to the dump, runs `pg_restore`, then
`roles.sql`, `indexes.sql` and `retrieval_indexes.sql` from this repository and
`ANALYZE` (about 80 s). It refuses a database that already has the project tables;
`docker compose down -v` deletes existing data first. The dump `scilitrag_v2`
(about 400 MiB, chunks of 256 tokens with 10 % overlap; D062) holds schema and data
of the six tables only, without indexes, roles or grants, so index changes need no
new dump. `scilitrag_v1` (512-token chunks) is superseded.

To build it yourself, run D-03 and D-12 first (`filter_data.py`, `select_fulltext.py`),
apply `schema.sql` and `roles.sql` to an empty database, then:

```sh
uv run --locked --group data --group rag python scripts/load_full.py --check-only
uv run --locked --group data --group rag python scripts/load_full.py
uv run --locked --group data --group rag --group embed python scripts/embed_database.py --batch-size 1024
uv run --locked --group data python scripts/dump_database.py
```

The import loads all 56,115 papers, 328,622 citations, 70,171 authors and 48,860
chunks of the 1,000 full-text papers (about 1 min). Embedding the 104,975 vectors
took about 12 min on an RTX 4070 Ti SUPER and takes many hours on CPU. The lock
installs CPU torch, so on a GPU run the embedding step in its own environment
(PowerShell: `$env:PYTHONPATH = "src"` first):

```sh
uv run --no-project --python 3.13 --index https://download.pytorch.org/whl/cu130 --with "torch==2.14.1+cu130" --with sentence-transformers==6.2.0 --with transformers==5.20.2 --with numpy==2.4.6 --with pyarrow==26.0.0 --with python-dotenv==1.2.4 --with pydantic==2.14.1 --with "psycopg[binary]==3.3.6" python scripts/embed_database.py --batch-size 1024
```

Create the dump before running `indexes.sql`; the restore builds the indexes.
Details in D054 and D062.

### Validate the database (D-17)

```sh
uv run --locked --group data --group rag python scripts/validate_database.py
```

Read-only checks with the app login: rows per table (at least 1,000 in at least 3
tables), validated single-column foreign keys with the expected target column
and schema and no orphans, no missing embeddings or blank texts and names
(including tabs and newlines), nonempty published metadata in the version 1
singleton row, and no e-mail addresses in stored text.
Papers without title and abstract or year are reported as INFO. `--json FILE`
also writes the results; a failed check gives exit code 1. Details in D055.

## Run with Docker

```
docker compose up -d --build
```

Starts `db`, `api` (http://127.0.0.1:8000/health) and `ui` (chat at http://127.0.0.1:8501). Use `--build` after code changes, otherwise the old images are reused. The API container uses the settings from `.env`, but reaches the database as `db` and Ollama on the host as `host.docker.internal`; the UI container reaches the API as `api`. Host ports can be changed with `DB_PORT`, `API_PORT` and `UI_PORT` in `.env`.

## Request log

The API writes one JSON line per query to stdout and to `logs/requests.jsonl` (folder set by `LOG_DIR`): request id, status (`ok`, `refused`, `error`, `timeout`, `cancelled`), latencies per stage, TTFT, tokens per second, guardrail events with score. `component_ms` holds steps inside a stage that components report with `scilitrag.timings.record_timing` (dense, BM25 and fusion time of the paper and chunk search). The query text is never logged, only its length. The id is also sent as `X-Request-ID` and shown under each answer in the UI.

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
  rag/          planner, retrieval, SQL facts, generation; llm/, evidence/, search/, scenarios/
  evaluation/   reference datasets, pipeline runner and quality metrics
  guardrails/   input checks, PII masking, sentence validation
  data/         embed component: granite embeddings for papers, chunks and queries (D050)
  contracts/    shared data types and interfaces
  mocks/        mock implementation of every interface
  config.py     environment configuration
  components.py selects mocks or real components
eval/retrieval/ synthetic public evaluation examples
eval/private/   ignored references, source passages, annotations and results
bench/          quantization benchmark
deploy/modal/   Ollama endpoint on Modal for benchmark and load test (A-09)
docker/         container configuration
scripts/        helper scripts (data download, smoke test)
tests/rag/      retrieval, generation and citation tests
tests/evaluation/ reference validation, runner and metric tests
tests/          shared/API/UI/data tests and fixtures
docs/           decisions and diagrams
```

The [database ER diagram](docs/er_diagram.md) documents the current tables,
relationships and search storage (D-24).

The UI imports nothing from the package except `scilitrag.contracts`.

## Evaluate retrieval

Compare the real database's dense, BM25 and hybrid paper search on all 597 LitSearch
queries, reusing D-22's stratified dev/test split and metrics:

```powershell
uv run --locked --group data --group rag --group embed python scripts/compare_retrieval.py
```

This needs the local D-03 files (`DATA_DIR`), the full search-ready database
(`DATABASE_URL`) and the pinned embedding model. It uses real search even with mocks
enabled and does not call the planner or LLM. See the [D-18 guide](docs/d18_retrieval_evaluation.md)
for smoke runs, outputs and the MRR@100 cutoff.
The [D-18 comparison tables](docs/d18_retrieval_comparison.md) contain the full 597-query run.

Run the synthetic cases with mocks before using real references or models:

```powershell
$env:USE_MOCKS = "1"
$env:REAL_COMPONENTS = ""
uv run --locked --group rag python scripts/evaluate_retrieval.py eval/retrieval/cases.example.json --allow-mocks --output eval/private/development.json
```

Export real references with `scripts/build_reference_set.py`, then compare installed model
candidates or re-score human-reviewed runs with `scripts/evaluate_retrieval.py`.
The [evaluation guide](docs/evaluation.md) explains frozen revisions, seed scopes, labels
and undefined metrics. A refused or empty response does not count as a completed evaluation.
Synthetic success does not establish real model quality.

## Decisions

See [docs/decisions.md](docs/decisions.md).
