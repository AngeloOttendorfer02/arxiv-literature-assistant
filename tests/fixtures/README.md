# D-07 local LitSearch fixture

LitSearch records are not committed. Each teammate builds the same fixture from
their own D-03 output. Requires D-01, D-03 and the D-04 to D-06 SQL setup.

Run in this order:

1. In your own `.env`, set `EMBED_MODEL=ibm-granite/granite-embedding-english-r2`
   (D044). Git does not update `.env`; chunks are counted with this tokenizer.
2. Install the groups (the `data` group now includes `transformers` for the tokenizer):
   ```sh
   uv sync --locked --group data --group rag
   ```
3. Select the fixture (once; it is never overwritten):
   ```sh
   uv run --locked --group data python scripts/select_fixture.py
   ```
4. Validate without the database. The first run downloads the pinned tokenizer
   (about 4 MB). Expect `Chunks: 6918 from 300 papers; 512 tokens with 10 % overlap`:
   ```sh
   uv run --locked --group data --group rag python scripts/load_fixture.py --check-only
   ```
5. Apply the additive embedding fingerprint migration as administrator (safe to
   re-run; new schemas already include the column):
   ```sh
   docker compose exec db psql -X -U postgres -d scilitrag -f /opt/scilitrag/db/migrations/001_embedding_fingerprint.sql
   ```
   If the database already holds data, reset it as administrator (command below).
6. Load:
   ```sh
   uv run --locked --group data --group rag python scripts/load_fixture.py
   ```
7. Embed papers (title + abstract) and chunks with the pinned granite model (D-14).
   This needs the `embed` group (CPU torch) and takes several minutes on a CPU; an
   interrupted run continues where it stopped:
   ```sh
   uv sync --locked --group data --group rag --group embed
   uv run --locked --group data --group rag --group embed python scripts/embed_database.py
   ```
   A resumed or completed run must match its saved model revision and preprocessing
   fingerprint. Old vectors without that fingerprint are refused and must be rebuilt.
8. As administrator, add the HNSW indexes (safe to re-run) and update statistics:
   ```sh
   docker compose exec db psql -X -U postgres -d scilitrag -f /opt/scilitrag/db/indexes.sql
   ```

`--chunk-tokens` (256, 384, 512) and `--overlap` (0, 10, 20 %) change the chunk
window; the defaults stay until D-20 decides.

**Selection** (defaults: 300 papers, seed 42): five queries from each of the four
query sets, all their target papers, then cited neighbors, citing neighbors and
distractors in turn. Distractors have no direct citation link to a target, which
does not make them irrelevant. `--papers` (200 to 500) and `--seed` change the
selection. Output goes to `DATA_DIR/fixtures/v0` and is never overwritten.

**Team check**: both scripts print a `Manifest SHA-256`. With the defaults (20
queries, 448 internal citations in the manifest) it must be:

```
727bae1cdf2d0affdd8e140ef53d2e31e01f9fca0c6623575c951ee6c438ce35
```

A different value means different D-03 inputs, settings or dependency versions.
`query_indices` in the manifest are row numbers at the pinned revision; stable
query IDs follow in D-18.

**Loading** uses `INGEST_DATABASE_URL` and only works on empty data tables. It
copies titles, abstracts, years and the citations between selected papers in one
transaction, with e-mail addresses replaced by `<EMAIL>` (O13). It also loads
authors and their paper links (D-08, see D026), plus section-based chunks (D-13,
see D049). Self-citations are dropped (D-09, see D032), so 447 citations are loaded.
`--check-only` reports full-text cleaning (D045) and chunk statistics without
writing to the database. Author IDs are numbered from 1 in
`normalized_name` order, so an existing database is first reset as administrator:

```sh
docker compose exec db psql -X -U postgres -d scilitrag -c "TRUNCATE papers, authors, paper_authors, citations, chunks RESTART IDENTITY; UPDATE schema_metadata SET embed_model = NULL, embed_fingerprint = NULL WHERE id = 1;"
```

The reset clears both the published embedding model and its permanent fingerprint,
because the loader rejects either field being set before it copies new data. Step 8 ends with
`ANALYZE papers, authors, paper_authors, citations, chunks;`.

Unit tests use synthetic records and need neither LitSearch nor a database.
