# Architecture Decisions

Each entry records a decision, its rationale, and any deviation from the
project specification.

## ADR-0001: BM25 via `bm25s`
pgvector handles dense retrieval; sparse retrieval uses the `bm25s` library
(in-memory, serialisable to disk under `backend/data/bm25/`). Postgres
`tsvector`/`ts_rank` is kept only as an extra baseline, not as the primary BM25.

## ADR-0002: Vector store is pgvector (HNSW)
Corpus size (~1k-1.5k articles, ~20-50k chunks) is small; pgvector with an HNSW
index (`vector_cosine_ops`) is sufficient. Qdrant/Milvus are not added. Dense
access goes through a `VectorStore` interface so the backend can be swapped in
one file if ever needed.

## ADR-0003: Agent loop in plain Python
The agentic layer is an explicit state machine in plain Python (no framework),
so each step is readable and traceable. No LangGraph.

## ADR-0004: Config-driven embedding dimension
The DB schema is specified with `vector(768)`, but the embedding dimension is a
configuration value (`EMBEDDING_DIM`). The migration template substitutes the
configured dimension so switching models (e.g. 768 -> 384) does not require
editing SQL by hand.

## ADR-0005: LLM abstraction with a mock mode
The LLM is accessed through an OpenAI-compatible client abstraction. Default
`LLM_MODE=ollama` targets a local Ollama endpoint. An additional
`LLM_MODE=mock` provides a deterministic, offline stub so the pipeline and
tests can run with no model downloaded. Model names come only from `.env`.

## ADR-0006: Verification in the build environment (no Docker daemon)
The CI/build sandbox used for development has no running Docker daemon, so
`make up` (Docker Compose) cannot be exercised there. The Compose files are the
shipped runtime and are validated with `docker compose config`. Milestone
acceptance checks that depend on a running stack are additionally verified by
running Postgres (with pgvector) and the backend natively. On a normal machine
with Docker, `make up && make migrate` is the supported path.
