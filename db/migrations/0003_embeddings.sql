-- 0003_embeddings.sql — thesis embeddings, split out of `theses`
--
-- REQUIRES the pgvector extension. Kept in its own migration so the core
-- ledger (0001, 0002) can be applied on a cluster without it — FR8 is deferred
-- until ≥50 theses have resolved, so nothing in v1 reads this table.
--
-- Why not a column on `theses`, as §5.1 has it:
--
--   * §5.2 forbids UPDATE on theses, but embeddings are not a property of the
--     thesis, they are a property of (thesis, model). Re-embedding after a
--     model change would require either an UPDATE to an immutable table or a
--     second nullable column per model.
--   * pgvector indexes need a fixed dimension, so a new model of a different
--     width is a schema migration on the ledger's central table.
--   * §5.2's own rule — "vectors from different models are not comparable and
--     the failure is silent" — is enforced here by making `model` part of the
--     primary key, so a query that forgets to filter by model gets duplicate
--     rows rather than a silently mixed neighbourhood.
--
-- §5.2: embed claim and mechanism only, never the full document.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE thesis_embeddings (
    thesis_id  uuid   NOT NULL REFERENCES theses(id),
    model      text   NOT NULL,
    dim        int    NOT NULL CHECK (dim > 0),
    embedding  vector NOT NULL,
    -- what was embedded, so a future model can reproduce the same input
    source     text   NOT NULL DEFAULT 'claim+mechanism',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (thesis_id, model),
    CONSTRAINT embedding_dim_matches CHECK (vector_dims(embedding) = dim)
);

COMMENT ON TABLE thesis_embeddings IS
    'One row per (thesis, embedding model). Provenance is mandatory: vectors '
    'from different models are not comparable (TRD 5.2).';

GRANT SELECT, INSERT, DELETE ON thesis_embeddings TO co_agent_app;
REVOKE UPDATE, TRUNCATE ON thesis_embeddings FROM co_agent_app;
GRANT SELECT ON thesis_embeddings TO co_agent_ro;

-- No index in v1: FR8 is deferred and an HNSW index needs a concrete dimension.
-- When a model is chosen, add a per-model partial index, e.g.
--
--   CREATE INDEX thesis_embeddings_hnsw_<model>
--       ON thesis_embeddings
--       USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
--       WHERE model = '<model>';

COMMIT;
