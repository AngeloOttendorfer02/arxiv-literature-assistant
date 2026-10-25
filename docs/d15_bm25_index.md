# D-15: BM25 indexes with pg_search

The pinned `paradedb/paradedb:0.26.2-pg18` image provides `pg_search`.
`docker/db/retrieval_indexes.sql` installs one ParadeDB index per table:

| Index | Columns | Search predicate |
| --- | --- | --- |
| `papers_title_abstract_bm25_idx` | `corpus_id, title, abstract, year` | `title ||| topic::text::pdb.boost(0.5) OR abstract ||| topic` |
| `chunks_text_bm25_idx` | `chunk_id, text, corpus_id` | `text ||| topic`, scoped to selected papers |

Title and abstract remain separate nullable columns. A match in either field is
sufficient; the application binds the same topic to both fields. The default
Unicode tokenizer splits words and lowercases them. `|||` matches any query
token; `pdb.score` supplies BM25 relevance. Scores sort descending, then IDs sort
ascending for stable ties. Existing year, author and paper filters apply to the
whole paper predicate. Title matches use weight `0.5`; abstract matches retain
the default weight `1.0`. The explicit `::text` before `::pdb.boost(0.5)` is
required for psycopg-bound query parameters to avoid an `invalid typemod` error.
Dense retrieval and RRF remain unchanged.

The weight follows the team's reported LitSearch measurements using the D-22
dev/test split: adding titles improved Recall@20 by about 7 percentage points
when titles were downweighted; `0.5` performed best on dev. This measurement was
provided by the team and has not been reproduced here. The existing index name
is retained consistently in SQL, migration and readiness checks; a weight change
only affects the query and does not require rebuilding the index.

This follows ParadeDB's [index API](https://www.paradedb.com/docs/reference/indexing/create-index)
and [multi-field query examples](https://www.paradedb.com/blog/v2api).

## Install or upgrade

After schema creation and bulk loading, run as the database owner (adjust the
administrator and database names to the local configuration):

```sh
docker compose exec db psql -X -v ON_ERROR_STOP=1 -U postgres -d scilitrag -f /opt/scilitrag/db/retrieval_indexes.sql
```

If `papers_abstract_bm25_idx` already exists, first run:

```sh
docker compose exec db psql -X -v ON_ERROR_STOP=1 -U postgres -d scilitrag -f /opt/scilitrag/db/migrations/002_bm25_title_abstract.sql
```

The migration replaces the paper index in one transaction and retains all rows
and the existing chunk index. Schedule the rebuild when searches can wait for
locks. Reapplying either script is safe for its checked-in definitions;
`IF NOT EXISTS` does not repair an incorrectly defined index. Readiness verifies
table identity, column coverage and valid, ready, nonpartial indexes. Installation
requires owner privileges; application reads retain their SELECT-only role.
Later inserts, updates and deletes maintain the indexes automatically.

## Verification

```sh
uv run --locked --group rag pytest -q tests/rag/test_search_readiness.py tests/rag/test_search_sources.py tests/test_bm25_search.py
uv run --locked --group rag python scripts/check_database.py --require-search
```

The database tests require `RETRIEVAL_TEST_DATABASE_URL` pointing to a disposable
ParadeDB instance with schema-creation privileges. Each test uses its own schema,
applies the checked-in schema/index SQL and cleans up afterward. They cover
title-only and abstract-only hits, title weighting, NULL/empty text, case folding, score ordering,
year/author/paper filters, limits, chunk scope, real query plans, reapplication
and the legacy-index migration. CI provides the pinned database image. Skipped
database tests do not establish live SQL correctness. Full search preflight
additionally requires published dataset and embedding metadata.
