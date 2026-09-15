-- 0001_core.sql — core ledger schema (TRD §5.1)
--
-- Design notes where this departs from the TRD, with the reason:
--
--   * `cycles` is new. G1 ("≥90% of cycles complete without manual
--     intervention") has no denominator unless something records that a cycle
--     was *expected*. The scheduler writes the row before the supervisor is
--     involved, so an instance that is down still leaves evidence it missed a
--     cycle. §4.1's stated rationale ("an instance failure is visible rather
--     than silent") is only true if this row exists.
--
--   * `batches` is new. batch_id appears on four tables in §5.1 with nothing
--     to reference.
--
--   * `positions` is replaced by `fills`. One row per thesis cannot represent
--     partial fills, multiple lots or scale-ins, and FR6's deviation
--     measurement breaks on all three. `broker_exec_id` is unique so the daily
--     reconciliation job is idempotent.
--
--   * `theses.embedding` moves to its own table (0003). §5.2 forbids UPDATE on
--     theses, but re-embedding on a model change is inevitable, and pgvector
--     wants a fixed dimension per column — so a new model would need both an
--     UPDATE and a schema change to a table declared immutable.
--
--   * `falsifier_class` is new. FR9's 0.3–0.7 gate assumes every falsifier is
--     price-path expressible. A block bootstrap over returns cannot price
--     "guides below $X in Q3". Without this column the gate silently forces
--     every thesis into a price bet.
--
--   * `falsifier_gate_events` is new. Acceptance criterion 4a requires the
--     rejection gate be "demonstrably firing"; rejected candidates never
--     become theses, so the evidence has to live somewhere else.
--
--   * `tool_cache` is new. §4.3 keys recorded tool output by run_id, which
--     cannot be looked up on replay: a changed prompt issues different calls,
--     misses, and fetches live — putting post-hoc information into a replayed
--     cycle. Keyed by a content hash of (tool, normalised args) instead.
--
--   * `resolve_by` is derived by trigger from the snapshot's taken_at, not
--     from emission or fill time. FR4 requires approved and passed theses to
--     resolve on the same schedule, which only holds if the anchor is the
--     snapshot both arms were drawn from.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------------------------------------------------------------- enumerations

CREATE TYPE cycle_state          AS ENUM ('expected','running','complete','empty','failed');
CREATE TYPE batch_state          AS ENUM ('building','presented','decided','abandoned');
CREATE TYPE falsifier_class      AS ENUM ('price_path','event');
CREATE TYPE horizon_bucket       AS ENUM ('standard','long');
CREATE TYPE red_team_rec         AS ENUM ('flag','kill','pass');
-- 'not_presented' is distinct from 'passed': FR5 caps the batch at 6 while
-- FR4 produces ~6/week, so candidates get dropped. Recording a dropped
-- candidate as 'passed' would attribute to the human a decision no human made,
-- contaminating the G4 comparison.
CREATE TYPE decision_kind        AS ENUM ('approved','passed','killed','not_presented');
CREATE TYPE fill_side            AS ENUM ('buy','sell');
CREATE TYPE outcome_result       AS ENUM ('true','false','unresolvable');
CREATE TYPE outcome_attribution  AS ENUM ('claim_wrong','timing','market','execution','fx');
CREATE TYPE run_status           AS ENUM ('running','ok','failed','aborted_cost');
CREATE TYPE trace_kind           AS ENUM ('recv','think','send','done');
CREATE TYPE task_state           AS ENUM ('pending','running','done','failed','dead');
CREATE TYPE gate_verdict         AS ENUM ('accept','reject_too_easy','reject_too_hard','indeterminate');

-- ---------------------------------------------------------------------- cycles

CREATE TABLE cycles (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    expected_at   timestamptz  NOT NULL,
    triggered_by  text         NOT NULL,          -- 'eventbridge' | 'manual' | ...
    state         cycle_state  NOT NULL DEFAULT 'expected',
    started_at    timestamptz,
    ended_at      timestamptz,
    manual_intervention boolean NOT NULL DEFAULT false,  -- the G1 numerator
    note          text
);
CREATE UNIQUE INDEX cycles_expected_at_key ON cycles (expected_at);
COMMENT ON TABLE cycles IS
    'One row per scheduled cycle, written by the scheduler before any work '
    'starts. Supplies the denominator for G1.';

-- ------------------------------------------------------------------- snapshots

CREATE TABLE snapshots (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id           uuid REFERENCES cycles(id),
    taken_at           timestamptz NOT NULL,
    s3_key             text        NOT NULL,
    s3_version_id      text        NOT NULL,   -- §4.3 immutable + versioned
    universe_size      int         NOT NULL CHECK (universe_size >= 0),
    -- NFR5: staleness is recorded, never silently substituted. Per-symbol
    -- delay flags and quote timestamps live in the S3 object; these are the
    -- summary fields the scorer filters on.
    quote_mode         text        NOT NULL CHECK (quote_mode IN ('realtime','snap','delayed','mixed')),
    stale_symbol_ids   bigint[]    NOT NULL DEFAULT '{}',
    delayed_symbol_ids bigint[]    NOT NULL DEFAULT '{}',
    created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX snapshots_taken_at_idx ON snapshots (taken_at DESC);

-- --------------------------------------------------------------------- batches

CREATE TABLE batches (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cycle_id         uuid NOT NULL REFERENCES cycles(id),
    snapshot_id      uuid NOT NULL REFERENCES snapshots(id),
    state            batch_state NOT NULL DEFAULT 'building',
    presented_count  int         NOT NULL DEFAULT 0 CHECK (presented_count BETWEEN 0 AND 6),
    candidate_count  int         NOT NULL DEFAULT 0,
    created_at       timestamptz NOT NULL DEFAULT now(),
    decided_at       timestamptz
);
COMMENT ON COLUMN batches.presented_count IS 'FR5 cap of 6 applies to rows shown to the human.';
COMMENT ON COLUMN batches.candidate_count IS
    'All candidates emitted, including those not presented. FR4 scores these too.';

-- ---------------------------------------------------------------------- theses

CREATE TABLE theses (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id           uuid NOT NULL REFERENCES batches(id),
    snapshot_id        uuid NOT NULL REFERENCES snapshots(id),
    -- Questrade addresses symbols by internal integer id (§3.1); the ticker is
    -- kept for human reading but is not the join key.
    symbol_id          bigint      NOT NULL,
    symbol             text        NOT NULL,
    sector             text        NOT NULL,
    currency           char(3)     NOT NULL,

    claim              text        NOT NULL CHECK (length(btrim(claim)) > 0),
    mechanism          text        NOT NULL CHECK (length(btrim(mechanism)) > 0),
    falsifier          text        NOT NULL CHECK (length(btrim(falsifier)) > 0),
    falsifier_class    falsifier_class NOT NULL,

    snapshot_taken_at  timestamptz NOT NULL,      -- denormalised resolve anchor
    horizon_days       int         NOT NULL CHECK (horizon_days > 0),
    resolve_by         timestamptz NOT NULL,      -- set by trigger, see below
    -- FR3: horizons past the 120d cap are not invalid, they are scored
    -- separately. Pooling them in one calibration figure blends two tests.
    horizon_bucket     horizon_bucket NOT NULL
        GENERATED ALWAYS AS (CASE WHEN horizon_days <= 120
                             THEN 'standard'::horizon_bucket
                             ELSE 'long'::horizon_bucket END) STORED,

    confidence         numeric(4,3) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- FR9 / acceptance 4a: NOT NULL is what makes "computed for 100% of
    -- emitted theses" a property of the schema rather than of the code.
    null_probability   numeric(4,3) NOT NULL CHECK (null_probability >= 0 AND null_probability <= 1),
    sim_method         text         NOT NULL,
    sim_params         jsonb        NOT NULL,
    p95_drawdown       numeric(6,4) NOT NULL CHECK (p95_drawdown >= 0),
    proposed_weight    numeric(6,4) NOT NULL CHECK (proposed_weight >= 0),
    weight_reduced     boolean      NOT NULL DEFAULT false,

    prompt_version     text        NOT NULL,
    s3_key             text        NOT NULL,      -- full thesis text
    created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX theses_batch_idx    ON theses (batch_id);
CREATE INDEX theses_resolve_idx  ON theses (resolve_by) WHERE resolve_by IS NOT NULL;
CREATE INDEX theses_bucket_idx   ON theses (horizon_bucket, confidence);

-- resolve_by is derived, not supplied. A caller-supplied value would let the
-- anchor drift to emission or fill time, breaking FR4's same-schedule
-- requirement between approved and passed arms.
CREATE FUNCTION theses_set_resolve_by() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.resolve_by := NEW.snapshot_taken_at + make_interval(days => NEW.horizon_days);
    RETURN NEW;
END;
$$;
CREATE TRIGGER theses_resolve_by_biu
    BEFORE INSERT ON theses
    FOR EACH ROW EXECUTE FUNCTION theses_set_resolve_by();

-- ------------------------------------------------------- falsifier gate events

-- Every FR9 evaluation, including the rejections. Acceptance 4a asks for proof
-- the gate fires; theses that failed it are absent from `theses` by definition.
CREATE TABLE falsifier_gate_events (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id          uuid NOT NULL REFERENCES batches(id),
    thesis_id         uuid REFERENCES theses(id),   -- null when rejected
    symbol_id         bigint      NOT NULL,
    falsifier         text        NOT NULL,
    falsifier_class   falsifier_class NOT NULL,
    attempt           int         NOT NULL DEFAULT 1,  -- restatement round
    null_probability  numeric(4,3) NOT NULL,
    ci_low            numeric(4,3) NOT NULL,
    ci_high           numeric(4,3) NOT NULL,
    verdict           gate_verdict NOT NULL,
    sim_method        text        NOT NULL,
    sim_params        jsonb       NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX fge_batch_idx   ON falsifier_gate_events (batch_id);
CREATE INDEX fge_verdict_idx ON falsifier_gate_events (verdict, created_at DESC);

-- -------------------------------------------------------------- red team notes

CREATE TABLE red_team_notes (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    thesis_id      uuid NOT NULL REFERENCES theses(id),
    run_id         uuid NOT NULL,
    body           text NOT NULL,
    recommendation red_team_rec NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX rtn_thesis_idx ON red_team_notes (thesis_id);

-- ------------------------------------------------------------------- decisions

CREATE TABLE decisions (
    thesis_id             uuid PRIMARY KEY REFERENCES theses(id),
    batch_id              uuid NOT NULL REFERENCES batches(id),
    decision              decision_kind NOT NULL,
    reason                text,
    -- G4 is confounded without this. Approved and passed populations differ by
    -- construction, so an outcome gap between them measures candidate
    -- difficulty as much as gate skill. A probability stated by the human
    -- before deciding is scorable on the whole population, which removes the
    -- selection effect from the comparison.
    human_confidence      numeric(4,3) CHECK (human_confidence >= 0 AND human_confidence <= 1),
    -- The causal version: a fraction of decisions are randomly forced against
    -- the human's stated preference, giving an unconfounded holdout.
    randomized_holdout    boolean NOT NULL DEFAULT false,
    forced_against_human   boolean NOT NULL DEFAULT false,
    override_justification text,                -- FR5 cap override, logged
    decided_at            timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT killed_needs_reason
        CHECK (decision <> 'killed' OR length(btrim(coalesce(reason,''))) > 0),
    -- A row the human never saw cannot carry their probability, and a row they
    -- did see must.
    CONSTRAINT presented_needs_human_confidence
        CHECK ((decision = 'not_presented') = (human_confidence IS NULL)),
    CONSTRAINT holdout_implies_presented
        CHECK (NOT forced_against_human OR randomized_holdout)
);
CREATE INDEX decisions_batch_idx ON decisions (batch_id, decision);

-- ----------------------------------------------------------------------- fills

CREATE TABLE fills (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    thesis_id      uuid NOT NULL REFERENCES theses(id),
    broker_exec_id text NOT NULL,                -- idempotent reconciliation
    side           fill_side NOT NULL,
    filled_at      timestamptz NOT NULL,
    qty            numeric(18,6) NOT NULL CHECK (qty > 0),
    price          numeric(18,6) NOT NULL CHECK (price > 0),
    fees           numeric(18,6) NOT NULL DEFAULT 0 CHECK (fees >= 0),
    currency       char(3) NOT NULL,
    fx_to_cad      numeric(18,8),                -- rate at fill, for attribution
    corrects_fill  uuid REFERENCES fills(id),    -- corrections append, never edit
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX fills_broker_exec_key ON fills (broker_exec_id);
CREATE INDEX fills_thesis_idx ON fills (thesis_id, filled_at);

-- -------------------------------------------------------------------- outcomes

CREATE TABLE outcomes (
    thesis_id      uuid PRIMARY KEY REFERENCES theses(id),
    resolved_at    timestamptz NOT NULL,
    result         outcome_result NOT NULL,
    evidence_s3_key text NOT NULL,
    attribution    outcome_attribution,
    scorer_run_id  uuid NOT NULL,
    -- The resolving worker must not see the thesis's confidence or its
    -- approve/pass decision: both bias resolution in the direction that
    -- flatters the system, and both feed G3 and G4 directly. Writing the flag
    -- as a CHECK means the ledger cannot store a non-blinded resolution at all.
    blinded        boolean NOT NULL CHECK (blinded),
    resolver       text NOT NULL,                -- 'code' | 'llm+evidence' | 'human'
    human_audited  boolean NOT NULL DEFAULT false,
    created_at     timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT unresolvable_has_no_attribution
        CHECK ((result = 'unresolvable') = (attribution IS NULL))
);
CREATE INDEX outcomes_resolved_idx ON outcomes (resolved_at);

-- ------------------------------------------------------------------ agent runs

CREATE TABLE agent_runs (
    run_id         uuid PRIMARY KEY,
    batch_id       uuid REFERENCES batches(id),
    cycle_id       uuid REFERENCES cycles(id),
    agent          text        NOT NULL,
    started_at     timestamptz NOT NULL DEFAULT now(),
    ended_at       timestamptz,
    token_cost_usd numeric(10,4),
    input_tokens   bigint,
    output_tokens  bigint,
    status         run_status  NOT NULL DEFAULT 'running',
    replay_of      uuid REFERENCES agent_runs(run_id),  -- NFR1
    prompt_version text
);
CREATE INDEX agent_runs_cycle_idx ON agent_runs (cycle_id, agent);

CREATE TABLE trace_events (
    run_id text NOT NULL,
    seq    int  NOT NULL,
    kind   trace_kind NOT NULL,
    body   text,                    -- truncated
    s3_key text,                    -- full
    tool_cache_key text,            -- see tool_cache
    at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);

-- ------------------------------------------------------------------ tool cache

-- Keyed by a hash of (tool, normalised args) so a replay at a *different*
-- prompt version can still hit content fetched at the original cycle's date.
-- Keying by run_id alone makes replay silently fetch present-day information
-- into a past cycle, which is look-ahead leakage in exactly the comparison
-- G3 and G4 depend on.
CREATE TABLE tool_cache (
    key             text PRIMARY KEY,           -- sha256(tool || canonical args)
    tool            text        NOT NULL,
    args            jsonb       NOT NULL,
    s3_key          text        NOT NULL,
    fetched_at      timestamptz NOT NULL,
    first_run_id    uuid        NOT NULL,
    -- true only for sources that can be asked "as of" a date (EDGAR, BoC,
    -- StatCan). Open-web results are not date-boundable, so a replay that
    -- misses on one must fail rather than fetch.
    date_boundable  boolean     NOT NULL,
    as_of           timestamptz
);
CREATE INDEX tool_cache_tool_idx ON tool_cache (tool, fetched_at DESC);

-- ----------------------------------------------------------------- work queue

CREATE TABLE tasks (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind       text        NOT NULL,
    state      task_state  NOT NULL DEFAULT 'pending',
    attempts   int         NOT NULL DEFAULT 0,
    max_attempts int       NOT NULL DEFAULT 3,
    payload    jsonb       NOT NULL,
    claimed_at timestamptz,
    claimed_by text,
    last_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
-- NFR3: the supervisor reclaims anything left running on boot.
CREATE INDEX tasks_running_idx ON tasks (claimed_at) WHERE state = 'running';
CREATE INDEX tasks_pending_idx ON tasks (created_at) WHERE state = 'pending';

COMMIT;
