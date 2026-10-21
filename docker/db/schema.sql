-- D-04 initial schema, applied explicitly with psql.
-- Existing tables cause an error; later changes require a versioned migration.
\set ON_ERROR_STOP on

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE papers (
    corpus_id bigint PRIMARY KEY,
    -- Protected query targets may have empty titles.
    title text,
    abstract text,
    year integer,
    embedding vector(768)
);

CREATE TABLE authors (
    author_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    first_name text,
    last_name text,
    name text NOT NULL,
    -- A normalized name is a search value, not a unique person identifier.
    normalized_name text NOT NULL
);

CREATE TABLE paper_authors (
    author_id bigint NOT NULL REFERENCES authors (author_id),
    corpus_id bigint NOT NULL REFERENCES papers (corpus_id),
    PRIMARY KEY (author_id, corpus_id)
);

CREATE TABLE citations (
    citing_id bigint NOT NULL REFERENCES papers (corpus_id),
    cited_id bigint NOT NULL REFERENCES papers (corpus_id),
    PRIMARY KEY (citing_id, cited_id)
);

CREATE TABLE chunks (
    chunk_id text PRIMARY KEY,
    corpus_id bigint NOT NULL REFERENCES papers (corpus_id),
    section text,
    seq_no integer NOT NULL CHECK (seq_no >= 0),
    text text NOT NULL,
    embedding vector(768),
    UNIQUE (corpus_id, seq_no)
);

CREATE TABLE schema_metadata (
    id smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    schema_version integer NOT NULL,
    dataset_revision text,
    embed_model text
);

-- Dataset revision and embedding model are recorded when their data are built.
INSERT INTO schema_metadata (id, schema_version) VALUES (1, 1);

COMMIT;
