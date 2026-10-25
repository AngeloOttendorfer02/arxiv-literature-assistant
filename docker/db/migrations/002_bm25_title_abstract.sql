-- D-15: replace the old abstract-only index atomically, as the schema owner.
-- Run with psql -X -v ON_ERROR_STOP=1; searches may wait during the rebuild.
-- The separate chunk index is unchanged. Reapplying this migration is safe.
BEGIN;

DROP INDEX IF EXISTS papers_abstract_bm25_idx;
CREATE INDEX IF NOT EXISTS papers_title_abstract_bm25_idx
    ON papers USING paradedb (corpus_id, title, abstract, year);

COMMIT;

ANALYZE papers;
