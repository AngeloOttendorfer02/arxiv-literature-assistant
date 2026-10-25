-- D-14 additive ingestion metadata; the retrieval schema contract stays at version 1.
-- Run as the schema owner before embedding or resetting an existing database.
-- Existing vectors are not assigned a fingerprint: their origin must be verified
-- by rebuilding them, rather than trusting a model name alone.
\set ON_ERROR_STOP on

BEGIN;

ALTER TABLE public.schema_metadata
    ADD COLUMN IF NOT EXISTS embed_fingerprint text;

COMMIT;
