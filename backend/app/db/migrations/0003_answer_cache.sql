-- Persistent cache of generated answers, so repeated queries (and the
-- pre-warmed featured queries) are served instantly across restarts. The key
-- covers query, mode, domain, LLM provider/model, prompt version and index
-- version, so changing any of them naturally misses the cache.

CREATE TABLE IF NOT EXISTS answer_cache (
  key        TEXT PRIMARY KEY,
  query      TEXT NOT NULL,
  mode       TEXT NOT NULL,
  domain     TEXT NOT NULL,
  model      TEXT NOT NULL,
  answer     TEXT NOT NULL,
  citations  JSONB NOT NULL,
  chunk_ids  BIGINT[] NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
