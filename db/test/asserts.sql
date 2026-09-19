-- asserts.sql — behavioural tests for the ledger schema.
-- Run by db/test/run_migrations.sh against a throwaway cluster.
\set ON_ERROR_STOP on

-- ------------------------------------------------------------------- fixtures

INSERT INTO cycles (id, expected_at, triggered_by)
VALUES ('11111111-1111-1111-1111-111111111111', '2026-01-05 14:00+00', 'eventbridge');

INSERT INTO snapshots (id, cycle_id, taken_at, s3_key, s3_version_id,
                       universe_size, quote_mode)
VALUES ('22222222-2222-2222-2222-222222222222',
        '11111111-1111-1111-1111-111111111111',
        '2026-01-05 14:05+00', 'snapshots/2026-01-05.json', 'v-abc', 42, 'snap');

INSERT INTO batches (id, cycle_id, snapshot_id, presented_count, candidate_count)
VALUES ('33333333-3333-3333-3333-333333333333',
        '11111111-1111-1111-1111-111111111111',
        '22222222-2222-2222-2222-222222222222', 4, 7);

-- resolve_by is deliberately omitted: the trigger derives it from the
-- snapshot's taken_at, never from emission time.
INSERT INTO theses (id, batch_id, snapshot_id, symbol_id, symbol, sector,
                    currency, claim, mechanism, falsifier, falsifier_class,
                    snapshot_taken_at, horizon_days, confidence,
                    null_probability, sim_method, sim_params, p95_drawdown,
                    proposed_weight, prompt_version, s3_key)
VALUES ('44444444-4444-4444-4444-444444444444',
        '33333333-3333-3333-3333-333333333333',
        '22222222-2222-2222-2222-222222222222',
        9876, 'FAKE.TO', 'materials', 'CAD',
        'claim text', 'mechanism text', 'closes below -12% within horizon',
        'price_path', '2026-01-05 14:05+00', 60, 0.620,
        0.480, 'stationary_bootstrap', '{"paths":10000}'::jsonb,
        0.1830, 0.0250, 'research@v3', 'theses/4444.md');

DO $$
DECLARE rb timestamptz;
BEGIN
    SELECT resolve_by INTO rb FROM theses
     WHERE id = '44444444-4444-4444-4444-444444444444';
    IF rb <> '2026-01-05 14:05+00'::timestamptz + interval '60 days' THEN
        RAISE EXCEPTION 'FAIL: resolve_by not anchored to snapshot (got %)', rb;
    END IF;
    RAISE NOTICE 'ok  resolve_by derived from snapshot.taken_at';
END $$;

DO $$
DECLARE hb text;
BEGIN
    SELECT horizon_bucket::text INTO hb FROM theses
     WHERE id = '44444444-4444-4444-4444-444444444444';
    IF hb <> 'standard' THEN RAISE EXCEPTION 'FAIL: bucket = %', hb; END IF;
    RAISE NOTICE 'ok  horizon_bucket generated';
END $$;

-- A caller-supplied resolve_by must not win.
DO $$
DECLARE rb timestamptz;
BEGIN
    INSERT INTO theses (id, batch_id, snapshot_id, symbol_id, symbol, sector,
        currency, claim, mechanism, falsifier, falsifier_class,
        snapshot_taken_at, horizon_days, resolve_by, confidence,
        null_probability, sim_method, sim_params, p95_drawdown,
        proposed_weight, prompt_version, s3_key)
    VALUES ('55555555-5555-5555-5555-555555555555',
        '33333333-3333-3333-3333-333333333333',
        '22222222-2222-2222-2222-222222222222', 9877, 'FAKE2.TO', 'energy',
        'USD', 'c', 'm', 'f', 'event', '2026-01-05 14:05+00', 200,
        '2099-01-01 00:00+00', 0.550, 0.500, 'explicit_prior',
        '{"source":"base rate 2019-2025"}'::jsonb, 0.2000, 0.0100, 'research@v3',
        'theses/5555.md');
    SELECT resolve_by INTO rb FROM theses
     WHERE id = '55555555-5555-5555-5555-555555555555';
    IF rb = '2099-01-01 00:00+00'::timestamptz THEN
        RAISE EXCEPTION 'FAIL: caller-supplied resolve_by was honoured';
    END IF;
    RAISE NOTICE 'ok  supplied resolve_by overridden';
END $$;

DO $$
DECLARE hb text;
BEGIN
    SELECT horizon_bucket::text INTO hb FROM theses
     WHERE id = '55555555-5555-5555-5555-555555555555';
    IF hb <> 'long' THEN RAISE EXCEPTION 'FAIL: 200d bucket = %', hb; END IF;
    RAISE NOTICE 'ok  >120d horizon lands in the long bucket';
END $$;

-- --------------------------------------------------------------- constraints

DO $$
BEGIN
    BEGIN
        INSERT INTO decisions (thesis_id, batch_id, decision, human_confidence)
        VALUES ('44444444-4444-4444-4444-444444444444',
                '33333333-3333-3333-3333-333333333333', 'killed', 0.3);
        RAISE EXCEPTION 'FAIL: killed without reason accepted';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'ok  killed requires a typed reason';
    END;
END $$;

DO $$
BEGIN
    BEGIN
        INSERT INTO decisions (thesis_id, batch_id, decision)
        VALUES ('44444444-4444-4444-4444-444444444444',
                '33333333-3333-3333-3333-333333333333', 'approved');
        RAISE EXCEPTION 'FAIL: presented row without human_confidence accepted';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'ok  presented rows require a stated human probability';
    END;
END $$;

DO $$
BEGIN
    BEGIN
        INSERT INTO decisions (thesis_id, batch_id, decision, human_confidence)
        VALUES ('55555555-5555-5555-5555-555555555555',
                '33333333-3333-3333-3333-333333333333', 'not_presented', 0.5);
        RAISE EXCEPTION 'FAIL: not_presented row carried a human probability';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'ok  not_presented rows cannot carry a human probability';
    END;
END $$;

INSERT INTO decisions (thesis_id, batch_id, decision, human_confidence)
VALUES ('44444444-4444-4444-4444-444444444444',
        '33333333-3333-3333-3333-333333333333', 'approved', 0.700);
INSERT INTO decisions (thesis_id, batch_id, decision)
VALUES ('55555555-5555-5555-5555-555555555555',
        '33333333-3333-3333-3333-333333333333', 'not_presented');

DO $$
BEGIN
    BEGIN
        INSERT INTO outcomes (thesis_id, resolved_at, result, evidence_s3_key,
                              attribution, scorer_run_id, blinded, resolver)
        VALUES ('44444444-4444-4444-4444-444444444444', now(), 'false',
                'evidence/4444.json', 'claim_wrong',
                '66666666-6666-6666-6666-666666666666', false, 'llm+evidence');
        RAISE EXCEPTION 'FAIL: non-blinded resolution accepted';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'ok  ledger refuses a non-blinded resolution';
    END;
END $$;

DO $$
BEGIN
    BEGIN
        INSERT INTO outcomes (thesis_id, resolved_at, result, evidence_s3_key,
                              attribution, scorer_run_id, blinded, resolver)
        VALUES ('55555555-5555-5555-5555-555555555555', now(), 'unresolvable',
                'evidence/5555.json', 'timing',
                '66666666-6666-6666-6666-666666666666', true, 'llm+evidence');
        RAISE EXCEPTION 'FAIL: unresolvable carried an attribution';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'ok  unresolvable outcomes carry no attribution';
    END;
END $$;

INSERT INTO outcomes (thesis_id, resolved_at, result, evidence_s3_key,
                      attribution, scorer_run_id, blinded, resolver)
VALUES ('44444444-4444-4444-4444-444444444444', now(), 'false',
        'evidence/4444.json', 'claim_wrong',
        '66666666-6666-6666-6666-666666666666', true, 'llm+evidence');

DO $$
BEGIN
    BEGIN
        UPDATE batches SET presented_count = 7
         WHERE id = '33333333-3333-3333-3333-333333333333';
        RAISE EXCEPTION 'FAIL: presented_count above the FR5 cap accepted';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'ok  FR5 batch cap enforced';
    END;
END $$;

INSERT INTO fills (thesis_id, broker_exec_id, side, filled_at, qty, price, currency)
VALUES ('44444444-4444-4444-4444-444444444444', 'EXEC-1', 'buy', now(), 100, 12.5, 'CAD');

DO $$
BEGIN
    BEGIN
        -- The daily reconciliation job replaying yesterday's executions must be
        -- a no-op, not a duplicate position (FR6).
        INSERT INTO fills (thesis_id, broker_exec_id, side, filled_at, qty,
                           price, currency)
        VALUES ('44444444-4444-4444-4444-444444444444', 'EXEC-1', 'buy',
                now(), 100, 12.5, 'CAD');
        RAISE EXCEPTION 'FAIL: duplicate broker_exec_id accepted';
    EXCEPTION WHEN unique_violation THEN
        RAISE NOTICE 'ok  reconciliation is idempotent on broker_exec_id';
    END;
END $$;

-- Partial fills: two rows, one thesis. The replaced `positions` table could not
-- represent this, and FR6 measures deviation against it.
INSERT INTO fills (thesis_id, broker_exec_id, side, filled_at, qty, price, currency)
VALUES ('44444444-4444-4444-4444-444444444444', 'EXEC-2', 'buy', now(), 60, 12.40, 'CAD'),
       ('44444444-4444-4444-4444-444444444444', 'EXEC-3', 'buy', now(), 40, 12.55, 'CAD');
DO $$
DECLARE n int;
BEGIN
    SELECT count(*) INTO n FROM fills
     WHERE thesis_id = '44444444-4444-4444-4444-444444444444';
    IF n <> 3 THEN RAISE EXCEPTION 'FAIL: expected 3 fills, got %', n; END IF;
    RAISE NOTICE 'ok  partial fills representable';
END $$;

-- ------------------------------------------------- append-only, as the owner

DO $$
BEGIN
    BEGIN
        UPDATE theses SET claim = 'rewritten with hindsight'
         WHERE id = '44444444-4444-4444-4444-444444444444';
        RAISE EXCEPTION 'FAIL: UPDATE on theses succeeded';
    EXCEPTION WHEN restrict_violation THEN
        RAISE NOTICE 'ok  UPDATE on theses denied by trigger (owner)';
    END;
END $$;

DO $$
BEGIN
    BEGIN
        DELETE FROM outcomes
         WHERE thesis_id = '44444444-4444-4444-4444-444444444444';
        RAISE EXCEPTION 'FAIL: DELETE on outcomes succeeded';
    EXCEPTION WHEN restrict_violation THEN
        RAISE NOTICE 'ok  DELETE on outcomes denied by trigger (owner)';
    END;
END $$;

-- The usual accidental bypass. ENABLE ALWAYS is what closes it.
SET session_replication_role = 'replica';
DO $$
BEGIN
    BEGIN
        UPDATE decisions SET decision = 'passed'
         WHERE thesis_id = '44444444-4444-4444-4444-444444444444';
        RAISE EXCEPTION 'FAIL: session_replication_role bypassed the trigger';
    EXCEPTION WHEN restrict_violation THEN
        RAISE NOTICE 'ok  session_replication_role = replica does not bypass';
    END;
END $$;
RESET session_replication_role;

-- Mutable working state must still be mutable, or NFR3 cannot reclaim tasks.
INSERT INTO tasks (id, kind, payload)
VALUES ('77777777-7777-7777-7777-777777777777', 'research', '{}'::jsonb);
UPDATE tasks SET state = 'running', claimed_at = now(), claimed_by = 'w1'
 WHERE id = '77777777-7777-7777-7777-777777777777';
UPDATE cycles SET state = 'complete', ended_at = now()
 WHERE id = '11111111-1111-1111-1111-111111111111';
DO $$ BEGIN RAISE NOTICE 'ok  mutable tables still mutable'; END $$;

-- ------------------------------------------ append-only, as the app role

SET ROLE co_agent_app;

DO $$
BEGIN
    BEGIN
        UPDATE theses SET confidence = 0.99;
        RAISE EXCEPTION 'FAIL: app role updated theses';
    EXCEPTION
        WHEN insufficient_privilege THEN
            RAISE NOTICE 'ok  app role lacks UPDATE on theses (privilege)';
        WHEN restrict_violation THEN
            RAISE NOTICE 'ok  app role UPDATE on theses denied (trigger)';
    END;
END $$;

DO $$
BEGIN
    BEGIN
        EXECUTE 'TRUNCATE theses';
        RAISE EXCEPTION 'FAIL: app role truncated theses';
    EXCEPTION
        WHEN insufficient_privilege THEN
            RAISE NOTICE 'ok  app role lacks TRUNCATE on theses (privilege)';
        WHEN restrict_violation THEN
            RAISE NOTICE 'ok  app role TRUNCATE on theses denied (trigger)';
    END;
END $$;

DO $$
BEGIN
    BEGIN
        EXECUTE 'ALTER TABLE theses DISABLE TRIGGER theses_deny_mutation';
        RAISE EXCEPTION 'FAIL: app role disabled the deny trigger';
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'ok  app role cannot disable the deny trigger';
    END;
END $$;

-- Inserting is the whole point; it must still work.
INSERT INTO red_team_notes (thesis_id, run_id, body, recommendation)
VALUES ('44444444-4444-4444-4444-444444444444',
        '88888888-8888-8888-8888-888888888888', 'the mechanism is priced in',
        'flag');
DO $$ BEGIN RAISE NOTICE 'ok  app role can still append'; END $$;

RESET ROLE;
