-- Extend the allowed document domains to include 'ai'. The corpus scope was
-- refined to: recent Nobel (last 5 years + a few landmark prizes), quantum
-- computing, and core AI areas. See docs/DECISIONS.md (ADR-0007).

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_domain_check;
ALTER TABLE documents
  ADD CONSTRAINT documents_domain_check CHECK (domain IN ('nobel','quantum','ai'));
