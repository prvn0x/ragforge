-- RAGForge database bootstrap (ADR-0013). Run ONCE per database as a superuser or the
-- managed provider's admin user. Idempotent: safe to re-run (also rotates passwords).
-- Brace placeholders (e.g. dbname, api_password) are filled with psycopg.sql.Literal /
-- Identifier by db_bootstrap. Never write literal braces in this file.

-- ------------------------------------------------------------------ roles
DO $$
DECLARE r text;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rf_system') THEN
    CREATE ROLE rf_system NOLOGIN;            -- owns every table, view and definer function
  END IF;
  FOREACH r IN ARRAY ARRAY['rf_migrator','rf_api','rf_worker','rf_metrics','rf_backup'] LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('CREATE ROLE %I LOGIN', r);
    END IF;
  END LOOP;
END $$;

-- No role here is superuser, can create roles/dbs, or bypasses RLS.
ALTER ROLE rf_system   NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOLOGIN;
ALTER ROLE rf_migrator NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS LOGIN PASSWORD {migrator_password};
ALTER ROLE rf_api      NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS LOGIN PASSWORD {api_password};
ALTER ROLE rf_worker   NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS LOGIN PASSWORD {worker_password};
ALTER ROLE rf_metrics  NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS LOGIN PASSWORD {metrics_password};
ALTER ROLE rf_backup   NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS LOGIN PASSWORD {backup_password};

-- The migrator may SET ROLE rf_system to run DDL, but does not inherit its privileges.
GRANT rf_system TO rf_migrator WITH INHERIT FALSE, SET TRUE;

-- Backups read everything through RLS policies `FOR SELECT TO rf_backup USING (true)`.
GRANT pg_read_all_data TO rf_backup;

-- ------------------------------------------------------------------ database hardening
REVOKE ALL ON DATABASE {dbname} FROM PUBLIC;
GRANT CONNECT ON DATABASE {dbname} TO rf_migrator, rf_api, rf_worker, rf_metrics, rf_backup;
-- TEMP is revoked from PUBLIC (by the REVOKE ALL above), so temp tables cannot shadow
-- relations inside SECURITY DEFINER functions (whose search_path also ends with pg_temp).
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- ------------------------------------------------------------------ schemas + extension
CREATE SCHEMA IF NOT EXISTS rf AUTHORIZATION rf_system;
CREATE SCHEMA IF NOT EXISTS ext;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA ext;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA ext;

GRANT USAGE ON SCHEMA ext TO rf_system, rf_migrator, rf_api, rf_worker, rf_metrics, rf_backup;
GRANT USAGE ON SCHEMA rf  TO rf_api, rf_worker, rf_metrics, rf_backup;

-- Functions created by rf_system are not executable by PUBLIC unless granted explicitly.
-- (Must be the GLOBAL default for the role: a per-schema REVOKE cannot remove the built-in
-- PUBLIC EXECUTE default. A test asserts no rf function has a PUBLIC ACL entry.)
ALTER DEFAULT PRIVILEGES FOR ROLE rf_system REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;

-- ------------------------------------------------------------------ per-role session defaults
ALTER ROLE rf_migrator IN DATABASE {dbname} SET search_path = rf, ext;
ALTER ROLE rf_migrator IN DATABASE {dbname} SET lock_timeout = '5s';
ALTER ROLE rf_api      IN DATABASE {dbname} SET search_path = rf, ext;
ALTER ROLE rf_api      IN DATABASE {dbname} SET statement_timeout = '30s';
ALTER ROLE rf_api      IN DATABASE {dbname} SET idle_in_transaction_session_timeout = '60s';
ALTER ROLE rf_worker   IN DATABASE {dbname} SET search_path = rf, ext;
ALTER ROLE rf_worker   IN DATABASE {dbname} SET statement_timeout = '120s';
ALTER ROLE rf_worker   IN DATABASE {dbname} SET idle_in_transaction_session_timeout = '120s';
ALTER ROLE rf_metrics  IN DATABASE {dbname} SET search_path = rf;
ALTER ROLE rf_metrics  IN DATABASE {dbname} SET statement_timeout = '5s';
ALTER ROLE rf_backup   IN DATABASE {dbname} SET search_path = rf, ext;
