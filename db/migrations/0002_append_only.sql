-- 0002_append_only.sql — make the ledger's immutability a privilege, not a habit
--
-- §5.2 asks for this to be "enforced by a database constraint or trigger, not
-- convention". A trigger alone is not enough: the table owner can run
--
--     ALTER TABLE theses DISABLE TRIGGER theses_deny_mutation;
--     SET session_replication_role = 'replica';
--
-- and both silently switch it off. So this migration does three things:
--
--   1. Runs the application as a role that is not the table owner, and revokes
--      UPDATE / DELETE / TRUNCATE from it. A privilege cannot be disabled from
--      inside a session that does not hold it.
--   2. Adds a deny trigger as a backstop, declared ENABLE ALWAYS so that
--      session_replication_role = 'replica' does not bypass it.
--   3. Revokes the schema-level default that would let the app create its own
--      tables.
--
-- Operational requirement that the schema cannot enforce for you: migrations
-- run as co_agent_migrate (the owner), the application connects as co_agent_app
-- and never as the owner. If the app connects as owner, none of this holds.

BEGIN;

-- ------------------------------------------------------------------- app roles

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'co_agent_app') THEN
        CREATE ROLE co_agent_app NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'co_agent_ro') THEN
        CREATE ROLE co_agent_ro NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO co_agent_app, co_agent_ro;
REVOKE CREATE ON SCHEMA public FROM co_agent_app, co_agent_ro;

-- Dashboard and scorer read paths.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO co_agent_ro;

-- ------------------------------------------------------- mutable working state

-- These legitimately change: a run ends, a task is claimed, a cycle completes,
-- a batch moves from building to decided. NFR3 depends on tasks being mutable.
GRANT SELECT, INSERT, UPDATE, DELETE ON
    cycles, batches, agent_runs, tasks, tool_cache
TO co_agent_app;

-- --------------------------------------------------------- append-only ledger

-- The ledger's entire value is that it cannot be edited with hindsight.
GRANT SELECT, INSERT ON
    snapshots,
    theses,
    falsifier_gate_events,
    red_team_notes,
    decisions,
    fills,
    outcomes,
    trace_events
TO co_agent_app;

REVOKE UPDATE, DELETE, TRUNCATE ON
    snapshots,
    theses,
    falsifier_gate_events,
    red_team_notes,
    decisions,
    fills,
    outcomes,
    trace_events
FROM co_agent_app;

-- Sequences are not used (uuid defaults), but be explicit in case one is added.
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM co_agent_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO co_agent_app;

-- ---------------------------------------------------------- backstop trigger

CREATE FUNCTION deny_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION
        'table %.% is append-only (ledger integrity, TRD 5.2): % rejected',
        TG_TABLE_SCHEMA, TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'restrict_violation',
              HINT = 'Record a correcting row instead of editing this one.';
END;
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'snapshots','theses','falsifier_gate_events','red_team_notes',
        'decisions','fills','outcomes','trace_events'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION deny_mutation()',
            t || '_deny_mutation', t);
        -- ENABLE ALWAYS: fires even under session_replication_role = 'replica',
        -- which is how this check is usually circumvented by accident.
        EXECUTE format('ALTER TABLE %I ENABLE ALWAYS TRIGGER %I',
                       t, t || '_deny_mutation');
    END LOOP;
END
$$;

-- Restricting TRUNCATE by privilege is enough for the app role, but the owner
-- can still truncate; a statement trigger closes that for accidents.
DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'snapshots','theses','falsifier_gate_events','red_team_notes',
        'decisions','fills','outcomes','trace_events'
    ]
    LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE TRUNCATE ON %I '
            'FOR EACH STATEMENT EXECUTE FUNCTION deny_mutation()',
            t || '_deny_truncate', t);
        EXECUTE format('ALTER TABLE %I ENABLE ALWAYS TRIGGER %I',
                       t, t || '_deny_truncate');
    END LOOP;
END
$$;

COMMIT;
