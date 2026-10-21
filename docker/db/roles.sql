-- D-06 initial role setup, run explicitly with psql as the schema-owning superuser.
-- Existing role names cause an error; they are never silently repurposed.
\set ON_ERROR_STOP on
\set ECHO none
\set app_user ''
\set app_password ''
\set ingest_user ''
\set ingest_password ''
\getenv app_user APP_DB_USER
\getenv app_password APP_DB_PASSWORD
\getenv ingest_user INGEST_DB_USER
\getenv ingest_password INGEST_DB_PASSWORD

BEGIN;

-- pg_stat_statements would otherwise store CREATE ROLE text with its password.
SET LOCAL pg_stat_statements.track_utility = off;

-- URL-safe credentials also work in the connection URLs assembled by Compose.
SELECT :'app_user' ~ '^[a-z_][a-z0-9_]{0,62}$'
   AND :'ingest_user' ~ '^[a-z_][a-z0-9_]{0,62}$'
   AND :'app_user' <> :'ingest_user'
   AND :'app_user' <> current_user
   AND :'ingest_user' <> current_user
   AND :'app_password' ~ '^[A-Za-z0-9._~-]+$'
   AND :'ingest_password' ~ '^[A-Za-z0-9._~-]+$' AS valid_config
\gset
\if :valid_config
\else
    DO $$ BEGIN
        RAISE EXCEPTION 'Use distinct role names and nonempty URL-safe passwords; see README.';
    END $$;
\endif

-- Reject existing roles before sending any password-bearing CREATE statement.
SELECT NOT EXISTS (
    SELECT FROM pg_roles WHERE rolname IN (:'app_user', :'ingest_user')
) AS new_roles
\gset
\if :new_roles
\else
    DO $$ BEGIN
        RAISE EXCEPTION 'Role name already exists; use new names or review existing roles manually.';
    END $$;
\endif

-- Default privileges below must belong to the role that creates the tables.
SELECT count(*) = 6 AND bool_and(pg_get_userbyid(relowner) = current_user) AS owns_tables
FROM pg_class
WHERE oid IN ('public.papers'::regclass, 'public.authors'::regclass,
              'public.paper_authors'::regclass, 'public.citations'::regclass,
              'public.chunks'::regclass, 'public.schema_metadata'::regclass)
\gset
\if :owns_tables
\else
    DO $$ BEGIN
        RAISE EXCEPTION 'Run schema.sql first, then run roles.sql as the same table owner.';
    END $$;
\endif

CREATE ROLE :"app_user" LOGIN PASSWORD :'app_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
CREATE ROLE :"ingest_user" LOGIN PASSWORD :'ingest_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

SELECT current_database() AS database_name
\gset
GRANT CONNECT ON DATABASE :"database_name" TO :"app_user", :"ingest_user";
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO :"app_user", :"ingest_user";

-- PUBLIC grants would otherwise bypass the separation between the two roles.
REVOKE ALL ON TABLE public.papers, public.authors, public.paper_authors,
    public.citations, public.chunks, public.schema_metadata FROM PUBLIC;
GRANT SELECT ON TABLE public.papers, public.authors, public.paper_authors,
    public.citations, public.chunks, public.schema_metadata TO :"app_user";
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.papers, public.authors,
    public.paper_authors, public.citations, public.chunks, public.schema_metadata
    TO :"ingest_user";

REVOKE ALL ON SEQUENCE public.authors_author_id_seq FROM PUBLIC;
GRANT USAGE ON SEQUENCE public.authors_author_id_seq TO :"ingest_user";

-- Applies only to future objects created by the current schema owner in public.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO :"app_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"ingest_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO :"ingest_user";

-- Check effective permissions, including any pre-existing PUBLIC column grants.
SELECT bool_and(has_table_privilege(:'app_user', t, 'SELECT')
    AND NOT has_any_column_privilege(:'app_user', t, 'INSERT')
    AND NOT has_any_column_privilege(:'app_user', t, 'UPDATE')
    AND NOT has_table_privilege(:'app_user', t, 'DELETE')
    AND NOT has_table_privilege(:'app_user', t, 'TRUNCATE')
    AND has_table_privilege(:'ingest_user', t, 'SELECT')
    AND has_table_privilege(:'ingest_user', t, 'INSERT')
    AND has_table_privilege(:'ingest_user', t, 'UPDATE')
    AND has_table_privilege(:'ingest_user', t, 'DELETE'))
    AND NOT has_schema_privilege(:'app_user', 'public', 'CREATE')
    AND NOT has_schema_privilege(:'ingest_user', 'public', 'CREATE')
    AND NOT has_database_privilege(:'app_user', current_database(), 'CREATE')
    AND NOT has_database_privilege(:'ingest_user', current_database(), 'CREATE')
    AS valid_permissions
FROM unnest(ARRAY['public.papers', 'public.authors', 'public.paper_authors',
    'public.citations', 'public.chunks', 'public.schema_metadata']) AS tables(t)
\gset
\if :valid_permissions
\else
    DO $$ BEGIN
        RAISE EXCEPTION 'Unexpected effective table permissions; role setup rolled back.';
    END $$;
\endif

COMMIT;
