# References

External material consulted during implementation.

## Retrieval / BM25

- Qdrant 1.20.x release notes — https://qdrant.tech/blog/qdrant-1.20.x/
  Consulted for BM25 / sparse-retrieval and hybrid-search ideas while
  implementing the retrieval layer (M3/M4). Note: per spec §1.1 the project uses
  `bm25s` + pgvector and does not add Qdrant as a dependency; this reference is
  for technique, not for introducing a new store (see docs/DECISIONS.md).
