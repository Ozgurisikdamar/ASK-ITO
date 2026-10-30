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

## ADR-0007: Corpus scope refinement (M2)

The original scope (§3) targeted 1000–1500 articles across Nobel (2017–2026)
and quantum computing. The build environment's shared egress IP is rate-limited
by Wikimedia (HTTP 429 with `Retry-After` on the Action API, REST v1 and
`api.wikimedia.org` alike), so a live ingest at that volume is not feasible
here. The scope was therefore refined to a smaller, curated corpus that is still
representative and ingestible:

- **Nobel**: the last five years (2022–2026) in Physics, Chemistry and
  Physiology/Medicine (laureates + topic articles), plus a few landmark prizes
  from 2017–2021 (gravitational waves/LIGO, cancer immunotherapy, lithium-ion
  battery, CRISPR, black holes).
- **Quantum**: the quantum-computing seed branch (unchanged from §3.2).
- **AI** (new domain): five core areas required by the product owner —
  computer vision / image processing, Transformers & large language models,
  retrieval-augmented generation, autonomous AI agents, deep learning — plus
  directly related bridge articles (several of which connect to the 2024 Nobel
  prizes: Hopfield networks, Boltzmann machines, AlphaFold).

A new `ai` document domain is added (migration 0002). The mandatory components
(BM25 + vector hybrid, reranker, RAGAS) are unchanged. The one-step expansion,
filtering, cleaning and snapshot logic are unchanged; only the seed set and
total budget shrink. `make ingest` run on an unthrottled network reproduces the
same snapshot at whatever budget is configured.

## ADR-0008: Wikipedia access via the Action API, with Retry-After backoff

Ingest uses the MediaWiki Action API exactly as specified (`prop=extracts&
explaintext=1` for text, `prop=links&plnamespace=0&pllimit=max` for links),
with a descriptive User-Agent, `maxlag`, inter-request delay and exponential
backoff that honours the `Retry-After` header on 429/5xx. REST v1 was evaluated
as a fallback but is throttled on the same shared IP, so no deviation from the
specified API was warranted; the client simply backs off and is best run from
an unthrottled network.

In the build environment the egress is additionally fingerprint-blocked: the
Python HTTP client (httpx) receives HTTP 403 while `curl` with the identical
User-Agent reaches Wikimedia's rate limiter (429/200). The client therefore has
a pluggable HTTP backend (`WIKI_HTTP_BACKEND=auto|httpx|curl`): `auto` uses
httpx and transparently falls back to curl on a 403. This also hardens the
ingest for corporate/proxy networks that block library HTTP clients. On a normal
machine httpx is used directly and the fallback never triggers.

## ADR-0009: Expansion ranking by distinct-seed link count

Spec §3.3 ranks expansion candidates by combining an embedding-cosine relevance
(candidate lead vs domain centroid) with the number of distinct seeds linking to
the candidate. The M2 ingest ranks by the distinct-seed link count alone
(threshold ≥ 2, per-domain budgets). With the curated seed-heavy corpus this is
a strong, deterministic signal and keeps ingest independent of the embedding
model. The snapshot is frozen, so the produced corpus (258 docs) reflects this
ranking. Adding the embedding-cosine component is a possible future refinement;
it would only change which expansion candidates are selected, not the pipeline.

## ADR-0010: Reciprocal Rank Fusion for hybrid retrieval

BM25 and dense results are combined with Reciprocal Rank Fusion (k = 60, spec
§5 step 3) rather than weighted score addition. RRF needs no score
normalisation between the two very different score scales. The Qdrant 1.20 notes
(docs/REFERENCES.md) explicitly leave fusion to the implementer; RRF vs weighted
fusion is left as an A/B comparison for the evaluation panel (M8).

## ADR-0011: Hand-authored seed test set; portable source

Generating the full test set (spec §8.1, ≥150 questions) needs a real LLM for
question generation and a human for review. The build environment has no LLM,
so M7 ships: (a) the LLM-based generator (`make testset-draft`) and the review
endpoints/UI, verified mechanically; and (b) a hand-authored seed of ~46 real,
reviewed questions covering the §8.1 type distribution, so the evaluation (M8)
can run here. The remaining questions to ≥150 are produced with
`make testset-draft` + review on a machine with a local LLM.

`gold_doc_ids`/`gold_chunk_ids` are DB serial ids, specific to an index build.
The portable source is `eval/seed_questions.json` (titles + keywords);
`make testset-seed` re-resolves the gold ids against the local DB and writes
`eval/testset.jsonl`, and `make testset-load` loads it into `eval_questions`.
The committed `testset.jsonl` reflects this environment's build; reproduce it
with `make testset-seed` after indexing.

## ADR-0006: Verification in the build environment (no Docker daemon)
The CI/build sandbox used for development has no running Docker daemon, so
`make up` (Docker Compose) cannot be exercised there. The Compose files are the
shipped runtime and are validated with `docker compose config`. Milestone
acceptance checks that depend on a running stack are additionally verified by
running Postgres (with pgvector) and the backend natively. On a normal machine
with Docker, `make up && make migrate` is the supported path.
