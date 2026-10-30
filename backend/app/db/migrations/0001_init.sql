-- Initial schema. The embedding column dimension is templated: the migration
-- runner substitutes __EMBEDDING_DIM__ with the configured EMBEDDING_DIM.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id           BIGSERIAL PRIMARY KEY,
  pageid       BIGINT UNIQUE NOT NULL,
  revid        BIGINT NOT NULL,
  title        TEXT NOT NULL,
  url          TEXT NOT NULL,
  domain       TEXT NOT NULL CHECK (domain IN ('nobel','quantum')),
  also_quantum BOOLEAN NOT NULL DEFAULT FALSE,
  is_seed      BOOLEAN NOT NULL DEFAULT FALSE,
  categories   TEXT[],
  fetched_at   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id          BIGSERIAL PRIMARY KEY,
  doc_id      BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section     TEXT,
  ord         INT NOT NULL,
  text        TEXT NOT NULL,
  token_count INT NOT NULL,
  domain      TEXT NOT NULL,
  embedding   vector(__EMBEDDING_DIM__),
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS eval_questions (
  id             TEXT PRIMARY KEY,
  question       TEXT NOT NULL,
  reference      TEXT,
  gold_chunk_ids BIGINT[],
  gold_doc_ids   BIGINT[],
  qtype          TEXT NOT NULL,
  domain         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id          BIGSERIAL PRIMARY KEY,
  config_name TEXT NOT NULL,
  config_json JSONB NOT NULL,
  started_at  TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ,
  status      TEXT NOT NULL,
  summary     JSONB
);

CREATE TABLE IF NOT EXISTS eval_results (
  run_id              BIGINT REFERENCES eval_runs(id) ON DELETE CASCADE,
  question_id         TEXT REFERENCES eval_questions(id),
  answer              TEXT,
  retrieved_chunk_ids BIGINT[],
  metrics             JSONB,
  trace               JSONB,
  latency_ms          INT,
  llm_calls           INT,
  PRIMARY KEY (run_id, question_id)
);
