# Retrieval database contract

The authoritative schema is `docker/db/schema.sql`. Retrieval reads its existing
relations and uses `schema_metadata` row `id=1`, schema version 1. Both embedding
columns are `vector(768)`; dimension is read from PostgreSQL's type catalog.
The application does not require an `embed_dim` metadata column.

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

The production embedding entry point is `scilitrag.data.embedding.embed(texts)`.
It must return one finite, nonzero, 768-dimensional vector per input. That module
is a shared data-layer dependency. There is no automatic substitute for it.
Unit tests inject small deterministic vectors; database fixtures use synthetic
768-dimensional vectors under the separate `retrieval-test` model identity.

Topic search uses `papers.abstract`; chunk search uses `chunks.text`. Indexed
passage availability is derived with `EXISTS` over nonblank chunk text. This says
whether passages are indexed, not whether a source full-text document exists.
Search index installation is explicit, like the existing schema/index scripts.
Cosine search is correct without an ANN index; ANN performance must be measured
after data loading. BM25 requires a ParadeDB index on each searched relation.

## Access and operations

The database image is pinned in Compose. Enable `vector`, `pg_search` and
`pg_trgm`, then explicitly apply `schema.sql`, `indexes.sql` and the retrieval
index script when it is added. Initialization currently enables extensions only.

Provision the configured application login separately. Compose's `app` connection
does not create that role. Give the application CONNECT, schema USAGE and SELECT
on the six relations; it does not need CREATE or write privileges. Keep role
provisioning in the database owner's deployment process.

Every application read uses a read-only transaction and a statement timeout capped
at ten seconds. These settings complement role grants. All SQL templates are
application constants with bound values; the LLM never supplies SQL.

Tests use `RETRIEVAL_TEST_DATABASE_URL` only when explicitly set to a disposable
ParadeDB instance. The harness applies the actual checked-in schema in a unique
schema, inserts synthetic rows, and removes that schema afterward. Run those
tests before accepting the database integration; skipped tests are not SQL proof.

## Counting rule

S3 credits each author of a cited paper once per distinct
`(citing_id, cited_id, author_id)` triple. Topic/year filters select citing papers;
they do not filter the cited paper's year. The count covers the indexed corpus,
not a retrieval top-k sample. A separate SQL formulation verifies the result.