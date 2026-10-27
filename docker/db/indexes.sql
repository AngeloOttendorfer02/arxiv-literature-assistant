-- Apply explicitly with psql after schema.sql.
-- IF NOT EXISTS checks names, not definitions; changed definitions need a migration.
\set ON_ERROR_STOP on

BEGIN;

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Year filters (S1, S3).
CREATE INDEX IF NOT EXISTS papers_year_idx
    ON papers (year);

-- The primary key already covers the citing_id-first direction.
CREATE INDEX IF NOT EXISTS citations_cited_citing_idx
    ON citations (cited_id, citing_id);

-- The primary key already covers the author_id-first direction.
CREATE INDEX IF NOT EXISTS paper_authors_corpus_author_idx
    ON paper_authors (corpus_id, author_id);

-- chunks.corpus_id is covered by UNIQUE (corpus_id, seq_no) in schema.sql.

-- EntityLinker.paper: case-insensitive exact title, ordered by corpus_id.
CREATE INDEX IF NOT EXISTS papers_title_lower_idx
    ON papers (lower(title), corpus_id);

-- EntityLinker.paper: case-insensitive title prefix before a colon.
CREATE INDEX IF NOT EXISTS papers_title_prefix_idx
    ON papers (lower(btrim(split_part(title, ':', 1))), corpus_id);

-- EntityLinker.paper: candidate filtering with the trigram similarity operator.
CREATE INDEX IF NOT EXISTS papers_title_trgm_idx
    ON papers USING gin (title gin_trgm_ops);

-- EntityLinker.author: non-unique name lookup, ordered by author_id.
CREATE INDEX IF NOT EXISTS authors_normalized_name_idx
    ON authors (normalized_name, author_id);

-- D-14 dense search: retrieval orders by cosine distance (<=>); pgvector defaults
-- m = 16 and ef_construction = 64. Rows without a vector are not indexed.
CREATE INDEX IF NOT EXISTS papers_embedding_hnsw_idx
    ON papers USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Search defaults for new sessions (pgvector >= 0.8): retrieval asks for up to 100
-- dense candidates, more than the default candidate list of 40, and year or paper
-- filters drop index hits; strict_order keeps scanning without changing the order.
DO $$
BEGIN
    EXECUTE format('ALTER DATABASE %I SET hnsw.ef_search = 100', current_database());
    EXECUTE format('ALTER DATABASE %I SET hnsw.iterative_scan = strict_order', current_database());
END
$$;

COMMIT;

-- Repeat after bulk imports; empty tables do not produce new statistics.
ANALYZE papers, authors, paper_authors, citations, chunks;
