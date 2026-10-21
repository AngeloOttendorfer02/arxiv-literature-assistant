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

### D025 D-07 reproducible local fixture selection

- Context: D-07 needs 200 to 500 papers around 20 LitSearch queries, without redistributing dataset records. The chosen approach is local selection by each teammate rather than a committed list of dataset IDs.
- Decision: scripts/select_fixture.py reads D-03 Arrow files offline at the pinned revision. Defaults: 300 papers, 20 queries (five per query set), seed 42. Keep all selected query targets; fill remaining slots by rotating between shuffled outgoing-neighbor, incoming-neighbor and distractor pools within the retained corpus. Save all three selected Arrow subsets and a manifest with provenance, IDs and file hashes under DATA_DIR/fixtures/v0. Never overwrite an existing fixture.
- Loading: scripts/load_fixture.py validates checksums and target/corpus coverage, then uses INGEST_DATABASE_URL and psycopg from the existing rag group. Mask e-mail addresses in titles and abstracts with the mask_pii mock's pattern and `<EMAIL>` placeholder (O13); full texts follow in D-11. COPY paper metadata and deduplicated citations whose two endpoints are selected, in one transaction; reject nonempty data tables. Record dataset_revision but leave embed_model NULL. Authors, chunks and embeddings remain later tasks. Queries stay in the local Arrow subset for evaluation.
- Reason: Each teammate can reproduce the same small corpus without committing LitSearch texts or adding a dependency. The original D-03 cache stays unchanged. Synthetic unit tests cover selection/export and import refusal without requiring corpus files or a running DB.
- Reproducibility: Both selection and offline validation print the manifest SHA-256 for comparison; the default fixture's reference is recorded in tests/fixtures/README.md. Write the fixture manifest as UTF-8 with LF endings and hash the parsed D-03 manifest as canonical JSON, so whitespace, key ordering and Windows/Linux line endings do not change the reference. Arrow checksums still cover exact file bytes. Query row indices are sufficient for this pinned fixture; stable content-based query IDs remain D-18 follow-up work.
- Validation: The default real-data fixture contains 300 papers, 20 queries, 21 target papers and 448 internal citations. The local loader reported a successful import in 6.71 seconds, meeting the under-60-second requirement. A read-only audit confirmed that DB paper metadata and citation pairs exactly match the validated fixture, with no e-mail addresses detected by the import regex. Synthetic tests cover reproducibility, source preservation, validation and import refusal; a regression test compares LF/CRLF source manifests with different JSON formatting. Hash-format changes do not require a DB reimport.
- Date: 2026-10-21

### D024 D-06 separate import and application database roles

- Context: D-06 requires a writable ingest login and a read-only API login. Compose previously used hardcoded app:app credentials, without creating the role.
- Decision: Apply docker/db/roles.sql explicitly with psql as the schema owner, after schema.sql. Create two new, distinct logins from environment variables. Grant the app SELECT on the six project tables; grant ingest SELECT, INSERT, UPDATE, DELETE and USAGE on the author ID sequence. Neither role owns objects or receives administrative privileges, role memberships, CREATE or TRUNCATE. Remove PUBLIC table/sequence grants and PUBLIC CREATE on schema public. Set default table/sequence grants for future objects created in public by the administrator running the script.
- Configuration: APP_DB_USER, APP_DB_PASSWORD, INGEST_DB_USER and INGEST_DB_PASSWORD in .env; placeholders and local connection URLs in .env.example. Compose builds the API URL from the app credentials and mounts setup SQL read-only outside initdb. Use URL-safe passwords so the same values work in SQL and connection URLs. INGEST_DATABASE_URL is reserved for the later loader.
- Reason: Keep import writes separate from query execution, support existing volumes without deleting data, and avoid new dependencies. Existing role names deliberately fail and roll back; password changes and migrations require explicit administrator actions.
- Credential handling: Disable pg_stat_statements.track_utility locally within the setup transaction so password-bearing CREATE ROLE statements are not collected. This does not remove older recorded statements. Login tests use host db; the container's loopback trust rule does not verify passwords. ANALYZE after imports remains an administrator action; ingest does not receive MAINTAIN.
- Validation: Initial setup applied locally with COMMIT; app SELECT succeeded and UPDATE was denied, while ingest INSERT/UPDATE/DELETE permission checks succeeded without changing rows. Local review confirmed the tracking setting with a rolled-back test role and identified loopback trust authentication. Successful password-authenticated app and ingest logins through host db are still pending.
- Date: 2026-10-21

### D023 D-05 query indexes matched to entity linking

- Context: D-05 lists year, both directions of citations and paper_authors, chunks by paper and a trigram title index. WP2 EntityLinker (src/scilitrag/rag/entities.py) also filters on lower(title), the title prefix before a colon and normalized_name.
- Options: only the D-05 indexes; also the expressions used by WP2.
- Decision: docker/db/indexes.sql, run after schema.sql. B-tree on papers(year), citations(cited_id, citing_id), paper_authors(corpus_id, author_id), both title expressions and authors(normalized_name), each with the ID used for ordering; GIN trigram on papers(title). The script creates pg_trgm itself, since the Docker init file only runs on a fresh volume.
- Reason: The primary keys cover the other directions, UNIQUE (corpus_id, seq_no) covers chunks by paper. Expression indexes only help while the WP2 queries use the same expressions, so query changes need a review with WP2. IF NOT EXISTS checks names only; a changed definition needs DROP INDEX first. Repeat ANALYZE after bulk imports.
- Validation: EXPLAIN plans after loading the fixture (D-07); still open.
- Date: 2026-10-20

### D022 D-04 initial relational schema and author identity

- Context: WP2 already queries papers, authors and paper_authors. LitSearch has no global author IDs; D021 keeps query targets with empty titles.
- Options: one author per normalized name enforced by UNIQUE (O2); non-unique names, author matching decided in D-08.
- Decision: docker/db/schema.sql, applied with psql. Plural table names as used by WP2, bigint IDs, author_id generated by the database. name and normalized_name required, not unique; author matching is decided in D-08. Title, abstract, year, first_name and last_name nullable. paper_authors and citations have composite primary keys with foreign keys. chunks: text primary key, seq_no >= 0 and unique per paper. Nullable vector(768) on papers and chunks, filled in D-14 (D011). One schema_metadata row with schema_version 1; dataset_revision and embed_model are set when the data are built.
- Reason: Supports the existing WP2 queries without fixing the author identity rule early. Foreign keys require both papers to exist, so D-09 must drop and count citations outside the corpus. No cascading deletes. Runs in one transaction and fails on existing tables; later changes need a migration and a new schema_version.
- Date: 2026-10-20

### D021 D-03 quality filters and protected query targets

- Context: D-02 found empty and oversized titles, extraction artifacts, missing annotations and missing years. Dropping all affected papers would also remove LitSearch evaluation targets.
- Decision: Reject non-target papers with a blank or one-character full text, a blank title, a title over 500 characters, no nonblank sectionheader, or neither an authorfirstname nor an authorlastname (checked on the annotated text spans). Keep every paper in query.corpusids and report its issues separately. Missing years become NULL. Titles and citation lists stay unchanged.
- Reason: Extends O4 with full-text and annotation checks. The author rule only removes papers without any name; normalization is D-08. Both corpus subsets keep the same IDs; the original cache stays unchanged.
- Validation: Counts in docs/d03_quality_filters.md and docs/d03_target_exceptions.md.
- Date: 2026-10-19

### D020 CPU-only PyTorch

- Context: Prompt Guard 2 and the NLI model need torch. On Linux, PyPI serves the CUDA build, which adds several GB to the API image and to every `uv sync`.
- Options: CPU build on every platform; separate CPU and GPU groups with `tool.uv.conflicts`; leave it open until WP3 adds torch.
- Decision: `torch` comes from the PyTorch CPU index on every platform (`tool.uv.sources`, explicit index) and is a direct dependency of the `guardrails` group. Any group that needs torch lists it directly, so the source applies. The API image installs `rag` and `guardrails`, without `dev`.
- Reason: Guardrails run on CPU as planned, the API image stays at about 1 GB, and there is one lock for everyone. The one-time embedding build (D-16) can run on a GPU outside this environment (Modal or a separate env); query embeddings with the same model on CPU stay compatible.
- Date: 2026-10-17

### D019 UI hardening and request tracing

- Context: A review of the UI and container PRs found that the UI crashes on events from a newer contract, its logic was untested, the SQL path of the citation drawer never ran, and only the request log line carried the request id.
- Decision: The UI shows a "contract version mismatch" error for unknown events or payloads instead of crashing; `app.py` runs `main()` only as `__main__`, so tests import it (`tests/test_ui.py`, skipped without the `ui` group). The `retrieve` mock returns a SQL evidence for S3 and the mock answer cites it, so the drawer shows the SQL query; the smoke test checks that S3 cites SQL. The request id is stored in a context variable and added to every application log line. A separate workflow builds both Docker images when Docker or source files change.
- Reason: The UI and API images can be rebuilt separately, so contract drift is a real case. Context variables follow the request into worker threads and the pipeline task without passing the id through every component.
- Date: 2026-10-17

### D018 Request limits, health checks and request log

- Context: A-03 needs clean handling of client disconnects, timeouts and missing services; A-08 needs structured per-request telemetry without PII for the benchmark and guardrail numbers in the report.
- Options: log file, stdout or both; per-stage or whole-request timeout; health check with or without DB driver.
- Decision: One time limit per query (`REQUEST_TIMEOUT_S`, default 120 s), then an `error` event with code `timeout`. The pipeline runs as its own task and the endpoint reads its events from a queue with a deadline, so the time limit never spans a `yield`. On client disconnect the endpoint closes the event stream, which cancels the task and closes the LLM stream; the request is logged as `cancelled`. `DoneEvent` carries `tokens` and `tokens_per_s`, measured once in the pipeline (contracts 0.3). One JSON line per query to stdout (live in `docker compose logs`) and, if writable, to `logs/requests.jsonl`; it holds ids, status, latencies, TTFT, tokens/s and guardrail events, never the query text. Request id via a plain ASGI middleware and `X-Request-ID`. `/health` checks database reachability by TCP and the LLM via `GET /models` in parallel (1 s each), only for components that run for real, and reports `degraded` with HTTP 200 otherwise.
- Reason: 120 s covers about 2000 tokens on CPU (about 30 tokens/s) plus retrieval and checks. Logging only the query length rules out PII in logs even if masking misses something. The TCP check needs no DB driver before WP1 picks one.
- Date: 2026-10-17

### D017 Live guardrail status and citation drawer

- Context: The UI must show live guardrail status and let users inspect each citation (A-06, A-07). Removed sentences were only counted in `done`.
- Options: report removed sentences in `done` only; a new event type; a `status` event per removed sentence.
- Decision: The API sends a `status` event with `stage="generate"` and result `SENTENCE_REMOVED` + the sentence status (e.g. `removed: unsupported`) for every sentence that fails `check_sentence`. The constant lives in the contracts (additive, no version change). The UI shows one badge per stage in the sidebar, updated per event, and a drawer with title, year, ids, section, supporting text and the SQL query for a clicked citation. The `check_sentence` mock now runs the structure stage (label present and known), and the mock answer contains one sentence without a source, so the removal path always runs.
- Reason: No new event type; the existing event list stays the single source for UI and logs.
- Date: 2026-10-17

### D016 Citation carries the supporting text (contracts 0.2)

- Context: The citation drawer (A-06) must show the supporting passage. The UI only receives SSE events, and `Citation` had no text or section.
- Options: add `section` and `text` to `Citation`; send the evidence list as its own event; show metadata only.
- Decision: `Citation` gets `section` and `text`, filled by `render_citations` from the evidence. `CONTRACT_VERSION` 0.2.
- Reason: Smallest change; each sentence event already carries everything the drawer needs, no new event type.
- Date: 2026-10-17

### D015 Smoke test grows with the system (extends D009)

- Context: The smoke test only exercised the mocks; real components were never run through the whole pipeline before integration.
- Options: keep it mock-only, run it against real components on demand, add a separate integration suite.
- Decision: `scripts/smoke.py` uses mocks except the components in `REAL_COMPONENTS` (environment or `.env`), so each owner runs the same cases with their real module. CI stays mock-only, since it has no database or model. Every PR that adds behaviour visible in the API adds or updates a smoke case. An expected failure only applies while `check_input` is mocked. Later: an option to run the cases against the running Docker stack (for E-10).
- Reason: One set of end-to-end cases from mocks to the final system; no second test suite to maintain.
- Date: 2026-10-17

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
