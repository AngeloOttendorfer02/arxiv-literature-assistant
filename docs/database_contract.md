# Retrieval database contract

The authoritative schema is `docker/db/schema.sql`. Retrieval reads its existing
relations and uses `schema_metadata` row `id=1`, schema version 1. Both embedding
columns are `vector(768)`; dimension is read from PostgreSQL's type catalog.
The application does not require an `embed_dim` metadata column.

The [ER diagram (D-24)](er_diagram.md) shows all tables, keys, relationship
cardinalities and the vector, sparse and relational storage choices.

| Relation | Retrieval fields |
| --- | --- |
| papers | corpus_id, title, abstract, year, embedding |
| authors | author_id, name, normalized_name |
| paper_authors | corpus_id, author_id |
| citations | citing_id, cited_id |
| chunks | chunk_id, corpus_id, section, seq_no, text, embedding |
| schema_metadata | id, schema_version, dataset_revision, embed_model |

Author names are not unique. Keep distinct author IDs and the existing ambiguity
handling. Ingestion must use the same normalization as `entities.normalize_name`.

## Readiness

`scripts/check_database.py` checks columns, extensions and vector declarations.
It accepts null dataset/model metadata while ingestion is incomplete.
`--require-search` also requires the frozen dataset revision and an exact match
between `EMBED_MODEL` and `schema_metadata.embed_model`. Metadata must describe
vectors actually loaded, with the same preprocessing and normalization as queries.
Matching a model name and dimension alone cannot prove preprocessing equivalence.

Ingestion also stores `schema_metadata.embed_fingerprint`, containing the model
revision and preprocessing settings. It remains set after publication, and the
embedding script verifies it before reusing either partial or completed vectors.
Existing databases add this column with
`docker/db/migrations/001_embedding_fingerprint.sql`; the additive ingestion field
does not change the version 1 retrieval interface above. Resetting embeddings also
clears this fingerprint. Vectors without matching provenance must be rebuilt.
The fixture loader also requires both `embed_model` and `embed_fingerprint` to be
NULL before importing, and rejects stale metadata before any COPY operation.

The production embedding component is created by
`scilitrag.data.embedding.build_component(settings)` and called as `embed(texts)`.
It returns one finite, nonzero, 768-dimensional vector per input. That module
is a shared data-layer dependency. There is no automatic substitute for it.
Unit tests inject small deterministic vectors; database fixtures use synthetic
768-dimensional vectors under the separate `retrieval-test` model identity.

Topic BM25 search matches `papers.title` or `papers.abstract` in one
`papers_title_abstract_bm25_idx`; chunk search uses `chunks.text` in
`chunks_text_bm25_idx`. Fields remain separate; no paper text is overwritten.
Both use the default Unicode tokenizer and BM25 scores, with deterministic ID
ordering for ties. Paper title matches use `::text::pdb.boost(0.5)`; abstracts
retain weight `1.0`. The explicit text cast supports psycopg-bound parameters.
Year and linked-paper/author filters also apply to title hits.
Readiness checks require valid, ready, nonpartial indexes on the correct tables
with all required columns; a matching name alone is insufficient. Indexed
passage availability is derived with `EXISTS` over nonblank chunk text. This says
whether passages are indexed, not whether a source full-text document exists.
Search index installation is explicit, like the existing schema/index scripts.
Cosine search is correct without an ANN index; ANN performance must be measured
after data loading. BM25 requires a ParadeDB index on each searched relation.

## Access and operations

The database image is pinned in Compose. Enable `vector`, `pg_search` and
`pg_trgm`, then explicitly apply `schema.sql`, `indexes.sql` and the retrieval
index script `retrieval_indexes.sql`. Initialization enables extensions only.
Existing abstract-only indexes first require
`docker/db/migrations/002_bm25_title_abstract.sql`, which atomically replaces the
paper index without changing rows or the chunk index. Plan for searches to wait
during the rebuild. See [D-15](d15_bm25_index.md) for deployment and verification.

The explicit role setup is `docker/db/roles.sql`, applied by the schema owner
after `schema.sql` and the existing relational indexes. It creates the configured
`APP_DB_USER` and `INGEST_DB_USER`; Compose supplies their environment values but
does not run role setup automatically. Application reads use the SELECT-only app
login; import writes use the separate ingest login. Existing tables and roles are
initialization prerequisites, so do not rerun initial creation scripts blindly.

Keep `APP_DB_USER`, `APP_DB_PASSWORD`, `INGEST_DB_USER` and `INGEST_DB_PASSWORD`
before the interpolated URLs in `.env`. The local `DATABASE_URL` uses the app
credentials and the published host port; `INGEST_DATABASE_URL` is for data loaders.
Compose reaches the same database through host `db`. The SQL setup files are
mounted read-only at `/opt/scilitrag/db`; apply retrieval indexes as the database
owner. Application code needs no CREATE, DDL or write privileges.

An existing development volume is retained when adding role setup. Check actual
app and ingest password authentication and permissions as described in README.
Neither application nor ingest credentials are suitable for the integration
fixture harness, which requires schema creation in a disposable test database.

Every application read uses a read-only transaction and a statement timeout capped
at ten seconds. These settings complement role grants. All SQL templates are
application constants with bound values; the LLM never supplies SQL.

`tests/test_app_role_readonly.py` (S-08) runs when `DATABASE_URL` is set. It tries
INSERT, UPDATE, DELETE, TRUNCATE and DDL with the app login inside rolled-back
transactions and requires the database to answer with insufficient privilege.

Tests use `RETRIEVAL_TEST_DATABASE_URL` only when explicitly set to a disposable
ParadeDB instance. The harness applies the actual checked-in schema in a unique
schema, inserts synthetic rows, and removes that schema afterward. Run those
tests before accepting the database integration; skipped tests are not SQL proof.

## Counting rule

S3 credits each author of a cited paper once per distinct
`(citing_id, cited_id, author_id)` triple. Topic/year filters select citing papers;
they do not filter the cited paper's year. The count covers the indexed corpus,
not a retrieval top-k sample. A separate SQL formulation verifies the result.
