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

COMMIT;

-- Repeat after bulk imports; empty tables do not produce new statistics.
ANALYZE papers, authors, paper_authors, citations, chunks;
