-- D-15: apply explicitly as the schema owner after schema creation and data loading.
-- Existing abstract-only indexes must first use migrations/002_bm25_title_abstract.sql.
-- IF NOT EXISTS checks names only; readiness also verifies table and indexed columns.
BEGIN;

CREATE INDEX IF NOT EXISTS papers_title_abstract_bm25_idx
    ON papers USING paradedb (corpus_id, title, abstract, year);

CREATE INDEX IF NOT EXISTS chunks_text_bm25_idx
    ON chunks USING paradedb (chunk_id, text, corpus_id);

COMMIT;

ANALYZE papers;
ANALYZE chunks;
