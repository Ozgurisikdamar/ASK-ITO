-- Dynamic Wikipedia ingestion (AI Mode adds articles when the local evidence is
-- missing). Snapshot documents keep origin 'snapshot'; dynamically fetched ones
-- are 'dynamic' with the domain 'general'. A content hash makes re-ingestion of
-- an unchanged revision a no-op; (doc_id, ord) is unique so a document's chunks
-- can be replaced atomically without duplicates.

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_domain_check;
ALTER TABLE documents
  ADD CONSTRAINT documents_domain_check
  CHECK (domain IN ('nobel','quantum','ai','general'));

ALTER TABLE documents ADD COLUMN IF NOT EXISTS origin TEXT NOT NULL DEFAULT 'snapshot';
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_origin_check;
ALTER TABLE documents
  ADD CONSTRAINT documents_origin_check CHECK (origin IN ('snapshot','dynamic'));
ALTER TABLE documents ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'en';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chunks_doc_ord_key') THEN
    ALTER TABLE chunks ADD CONSTRAINT chunks_doc_ord_key UNIQUE (doc_id, ord);
  END IF;
END $$;
