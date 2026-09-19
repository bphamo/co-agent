# Ledger schema

```
db/migrations/0001_core.sql        tables, enums, constraints
db/migrations/0002_append_only.sql immutability by privilege + trigger
db/migrations/0003_embeddings.sql  thesis embeddings (requires pgvector)
db/test/run_migrations.sh          applies all three to a throwaway cluster
db/test/asserts.sql                behavioural tests
```

```
./db/test/run_migrations.sh
```

creates a temporary Postgres 16 cluster, applies the migrations, runs the
assertions, and destroys it. `0003` is skipped automatically where pgvector is
not installed; nothing in v1 reads that table (FR8 is deferred), so the core
ledger does not depend on it.

## Operational requirement the schema cannot enforce for you

Migrations run as the table owner. **The application connects as
`co_agent_app`, never as the owner.** If the app connects as owner, the
append-only guarantee below is decoration.

## Append-only, and why a trigger alone is not enough

§5.2 asks for immutability "enforced by a database constraint or trigger, not
convention". A trigger by itself is defeated from inside a normal session:

```sql
ALTER TABLE theses DISABLE TRIGGER theses_deny_mutation;  -- owner can do this
SET session_replication_role = 'replica';                 -- skips normal triggers
```

So `0002` does three things instead:

1. **Revokes `UPDATE`, `DELETE` and `TRUNCATE`** from `co_agent_app` on every
   ledger table. A privilege cannot be disabled from inside a session that does
   not hold it.
2. **Adds a deny trigger as a backstop**, declared `ENABLE ALWAYS` so
   `session_replication_role = 'replica'` does not bypass it.
3. **Revokes `CREATE` on the schema** from the app role.

`db/test/asserts.sql` checks all three, including that the app role cannot
disable the trigger and that `session_replication_role` does not get through.
Working state — `cycles`, `batches`, `agent_runs`, `tasks`, `tool_cache` —
stays mutable, because NFR3's task reclamation needs it to be.

## Departures from TRD §5.1

| Change | Reason |
|---|---|
| `cycles` added | G1 ("≥90% of cycles complete") has no denominator unless something records that a cycle was *expected*. Written by the scheduler before the supervisor is involved, so an instance that is down still leaves evidence it missed a cycle — which is §4.1's stated rationale for external scheduling. |
| `batches` added | `batch_id` appears on four tables with nothing to reference. |
| `positions` → `fills` | One row per thesis cannot represent partial fills, multiple lots or scale-ins, and FR6's deviation measurement breaks on all three. `broker_exec_id` is unique, so the daily reconciliation job is idempotent. Corrections append a row pointing at the one they correct. |
| `theses.embedding` → `thesis_embeddings` | §5.2 forbids `UPDATE` on `theses`, but re-embedding after a model change is inevitable, and pgvector wants a fixed dimension per column — so a new model would need both an `UPDATE` to an immutable table and a schema change to the ledger's central table. `model` is part of the primary key, so a query that forgets to filter by model returns duplicates rather than a silently mixed neighbourhood. |
| `falsifier_class` added | FR9's 0.3–0.7 gate assumes every falsifier is price-path expressible. Without this column, the gate silently forces every thesis into a price bet. |
| `falsifier_gate_events` added | Acceptance 4a wants the rejection gate "demonstrably firing", but rejected candidates never become theses, so the evidence needs somewhere else to live. Records every evaluation including restatement rounds. |
| `tool_cache` added | §4.3 keys recorded tool output by `run_id`, which cannot be looked up on replay: a changed prompt issues different calls, misses, and fetches live — putting present-day information into a replayed cycle, in exactly the comparison G3 and G4 depend on. Keyed by a hash of `(tool, normalised args)` instead, with `date_boundable` marking the sources that can be asked "as of" a date. A replay that misses on a non-boundable source must fail rather than fetch. |
| `decisions.human_confidence` added | G4 is confounded without it. Approved and passed populations differ by construction, so an outcome gap between them measures candidate difficulty as much as gate skill. A probability the human states *before* deciding is scorable on the whole population, which takes the selection effect out of the comparison. Required on every presented row. |
| `decisions.randomized_holdout` / `forced_against_human` | The causal version of the same measurement: a fraction of decisions forced against the human's stated preference gives an unconfounded holdout. |
| `decision = 'not_presented'` added | FR5 caps the batch at 6 while FR4 produces ~6/week, so candidates get dropped. Recording a dropped candidate as `passed` attributes to the human a decision no human made, and pollutes G4. |
| `outcomes.blinded` added, `CHECK (blinded)` | The resolving worker must not see the thesis's confidence or its approve/pass decision: both bias resolution in the direction that flatters the system, and both feed G3 and G4 directly. As a `CHECK`, the ledger cannot store a non-blinded resolution at all. |
| `resolve_by` derived by trigger | Anchored to `snapshots.taken_at`, not emission or fill time. FR4 requires approved and passed theses to resolve on the same schedule, which only holds if the anchor is the snapshot both arms were drawn from. A caller-supplied value is overwritten. |
| `horizon_bucket` generated | FR3 scores >120-day horizons separately; making it generated means no report can forget to bucket. |
| `null_probability` `NOT NULL` | Acceptance 4a asks for it on 100% of emitted theses. `NOT NULL` makes that a property of the schema rather than of the code. |
| `symbol_id` added alongside `symbol` | §3.1: Questrade addresses symbols by internal integer id, and tickers change. |
| `fx_to_cad` on fills, `fx` attribution | CAD/USD exposure means FX moves P&L, and §5.1's attribution list has nowhere to put that. |

## Still open

- **Retention (§5.3)** is not implemented here. Trace pruning at 90 days, with
  indefinite retention for theses that resolved false, wants a scheduled job
  plus a partitioning decision on `trace_events`; neither belongs in a schema
  migration.
- **§5.3's nightly `pg_dump`** leaves a 24-hour RPO on the one artifact the
  project cannot regenerate, while NFR3 claims the instance is cattle. Continuous
  WAL archiving to S3 (wal-g or pgBackRest) drops that to minutes for about a
  dollar a month and answers open question 4 without moving to RDS. Also not a
  migration.
