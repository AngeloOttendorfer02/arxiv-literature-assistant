-- Apply explicitly after schema creation and data loading.
CREATE INDEX IF NOT EXISTS papers_abstract_bm25_idx
    ON papers USING paradedb (corpus_id, abstract, year);

CREATE INDEX IF NOT EXISTS chunks_text_bm25_idx
    ON chunks USING paradedb (chunk_id, text, corpus_id);

ANALYZE papers;
ANALYZE chunks;