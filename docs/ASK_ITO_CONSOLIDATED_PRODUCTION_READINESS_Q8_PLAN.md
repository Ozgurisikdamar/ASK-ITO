# ASK-ITO — Consolidated Requirements, Review Findings & Next Development Plan

**Status:** Consolidated implementation brief  
**Base commit reviewed:** `b279211697fe06829fa5d3ee437437fe9584416c`  
**Branch:** `production-readiness-q8-plan`  
**Model decision:** **Q8_0 is the selected default. Do not switch to Q4_K_M.**  
**Scope:** Continue development after the Fast RAG review and Redis/Q8/KV/GPU verification.

This document consolidates the previously agreed product/architecture requirements with the latest Fast RAG review, remediation report, and Q8/Redis verification. It is a planning/implementation brief—not evidence that the listed work has already been completed.

## 1. Source documents to read first

Read these files and the relevant implementation before changing code:

- `docs/ASK_ITO_FAST_RAG_REVIEW_AND_REMAINING_WORK.md`
- `docs/ASK_ITO_FAST_RAG_REVIEW_REMEDIATION_REPORT.md`
- `docs/ASK_ITO_FAST_RAG_IMPLEMENTATION_REPORT.md`
- `docs/ASK_ITO_REDIS_Q8_KV_GPU_OPTIMIZATION_REPORT.md`
- `docs/ASK_ITO_REDIS_Q8_GPU_VERIFICATION_REMEDIATION_CLAUDE.md`
- `docs/ASK_ITO_REDIS_Q8_KV_GPU_OPTIMIZATION_CLAUDE.md`
- `docs/ASK_ITO_LOCAL_LLM_OPTIMIZATION_CLAUDE.md`
- `docs/DECISIONS.md`
- README, .env.example, Compose files, tests, and actual implementation.

Treat reports as evidence to cross-check—not unquestionable truth. Inspect the current code and commit history. Clearly distinguish verified, not verified, failed, blocked, and inferred items.

## 2. Non-negotiable architectural decisions

1. **Q8_0 remains the primary/default model quantization.**
   - Do not change the default to Q4_K_M.
   - Do not silently fall back from Q8_0 to Q4_K_M.
   - Q4_K_M may remain available only as an explicit benchmark/troubleshooting option.
   - Optimize memory, context length, inference parameters, GPU offload, and agent behavior around Q8_0.
   - If the hardware cannot run Q8_0 reliably, report the limitation and provide an explicit actionable error/configuration guidance.
2. F16 KV cache remains the selected configuration unless a measured, quality-preserving reason is presented and approved.
3. Local-first and privacy-oriented behavior: no paid/cloud inference; keep Qwen/Ollama as the primary LLM.
4. PostgreSQL is the canonical persistent store for corpus documents/chunks/embeddings. Redis is a cache, not durable storage.
5. Search must remain useful without LLM generation. Agentic RAG remains optional and bounded.
6. Do not merge into main without explicit user approval. Work on a dedicated branch.
7. No fabricated tests, benchmark numbers, GPU evidence, or claims of target-machine validation.

## 3. Current evidence and caveats

Use the latest reports to preserve measured results, but do not overgeneralize:

- Latest optimization commit: `b279211697fe06829fa5d3ee437437fe9584416c`.
- The optimization report describes CPU-only sandbox tests; target PC and GPU offload remain unverified unless later evidence says otherwise.
- Reported sandbox Q8_0 memory usage is materially higher than Q4_K_M; Q8_0 plus Agentic RAG can exceed the sandbox’s available RAM. This is a hardware/resource limitation to address—not authorization to change the chosen quantization.
- Fast RAG review/remediation has measurement and test evidence; inspect its reports for exact scope and gaps.
- Citation-ID validity must not be conflated with citation entailment/faithfulness.
- A test passing with mocks is not equivalent to a real PostgreSQL/Redis/Ollama/Wikipedia integration test.
- GPU offload, Docker runtime, and target-machine behavior must be marked NOT VERIFIED until actually run on the corresponding environment.

## 4. Product and UX requirements

### 4.1 Search and AI Mode
- Keep **Search** and **AI Mode** as the two primary user modes.
- Remove the separate redundant **AI Answer** UI tab/entry only after tracing dependencies. Do not delete or break any backend Fast Answer capability that is still used.
- Search:
  - Hybrid retrieval (Dense + BM25 + RRF).
  - Fast, source-oriented, no mandatory LLM generation.
  - Search results/sources should be available even if Ollama is down.
- AI Mode:
  - Grounded RAG with local Qwen.
  - Query analysis/rewrite and multi-step retrieval only when needed.
  - Evidence sufficiency check, citations, bounded generation.
- Put the AI Mode affordance inside or immediately adjacent to the main search bar, with a familiar clean search UX. No microphone/camera icons.
- Preserve the query when switching modes; support responsive layout and existing theme choices.
- Keep ASK-ITO’s own visual identity; do not clone Google pixel-for-pixel.

### 4.2 Homepage
- Remove duplicate corpus disclaimer text.
- Do not claim the app searches all of Wikipedia.
- Update the description to accurately reflect the current corpus and the new dynamic Wikipedia ingestion behavior once implemented.
- Preserve the suggested question:
  - “How is the 2025 Physics prize related to quantum computers?”
- Replace other suggestions with compelling AI, quantum computing, Transformers, Agentic RAG, LLM/SLM, AI agents, RAG-vs-fine-tuning, and neural-network questions.
- Evaluate suggestions using the actual retrieval/evidence path—not article-title matches alone. If evidence is missing, trigger ingestion or choose a supported question.

## 5. Dynamic Wikipedia ingestion and persistent corpus expansion

Implement only after examining existing ingestion/retrieval contracts; reuse the current pipeline and embedding model.

Expected flow:
1. Search current corpus first.
2. Assess evidence sufficiency.
3. If sufficient, answer using corpus evidence without external fetch.
4. If insufficient, query the MediaWiki API for a relevant article or a tightly bounded set of articles.
5. Normalize content, preserve source metadata, chunk with the existing chunker, embed with the existing embedding model, and persist documents/chunks/embeddings in PostgreSQL.
6. Retrieve again from the expanded corpus and answer with grounded citations.
7. New chunks must become visible to both Dense and BM25 retrieval after commit.

Requirements:
- Persist title, canonical URL, language, revision ID/timestamp, content hash, ingestion timestamp/status, document ID, and chunk metadata.
- Idempotent ingestion; prevent duplicate articles/chunks; use transactions and DB constraints where appropriate.
- Stable IDs where feasible; preserve source/citation metadata.
- Bound article count/size, concurrency, timeouts, retries, and external API rate.
- Handle partial failures and retries without corrupting corpus state.
- Wikipedia text is untrusted evidence, never system instructions. Prompt-injection defenses must remain active.
- Never imply that the entire Wikipedia corpus is locally indexed.

## 6. Redis/cache requirements

- Cache only appropriate retrieval/query embedding/Dense/BM25/RRF and answer computations.
- Cache keys must include relevant normalized query, model/version, prompt version, corpus revision, retrieval configuration, and generation parameters.
- Corpus changes must invalidate or version-bump affected caches. Stale answer/citation results must not survive ingestion.
- Redis unavailable: Search and core RAG must degrade safely; do not make startup or Search depend on Redis.
- Test hit/miss, TTL, invalidation, Redis down/reconnect, fallback, and duplicate requests.
- Measure hit rate, latency, and memory. Do not claim performance gains without reproducible benchmarks.

## 7. AI Mode / Agentic RAG efficiency

Inspect and optimize:
- Agent iterations, LLM calls, query rewrites, retrieval calls, sufficiency checks, retries, prompt/context size, and streaming.
- Avoid multi-step Agentic execution for simple questions.
- Add explicit iteration/time/token budgets, timeouts, cancellation, retry limits, and loop prevention.
- Preserve answer quality, citations, and abstention behavior.
- Compare cold/warm, cache hit/miss, Search/Fast Answer/Agentic RAG.
- Report TTFT, retrieval time, agent time, generation time, total latency, calls/query, tokens/sec, RAM/VRAM where measurable.

## 8. Streaming, disconnects, and error lifecycle

Investigate frontend and backend together:
- SSE lifecycle, disconnect handling, proxy/server timeout, Ollama generation duration, cancellation, process restarts, and partial output.
- Ensure loading indicators terminate on all paths.
- Provide retry and preserve partial answer/sources when possible.
- Distinguish timeout, connection loss, empty evidence, no-answer, and backend/model error.
- Use user-friendly error messages; never expose stack traces.
- Add request/correlation IDs and structured diagnostics.
- Specifically reproduce the observed “connection to the server was lost”, indefinite loading, and “No answer was produced” cases.

## 9. Retrieval quality, citations, and safety

- Evaluate retrieval precision/recall, context relevance, answer faithfulness, citation correctness, hallucination, and abstention.
- Citation ID validity is not enough: validate whether cited passages support the associated claims.
- Where evidence is insufficient, ingest from Wikipedia or abstain; never invent sources.
- Test prompt injection from retrieved Wikipedia passages and user inputs.
- Restrict local-model endpoints to loopback by default; explicitly controlled exceptions only. Prevent SSRF/private-network/LAN endpoint abuse.
- Validate request sizes, CORS, Redis/PostgreSQL access, external ingestion URLs, and secrets handling.
- Logs must avoid credentials and unnecessary sensitive prompt/content data.

## 10. Observability

Add/verify structured logs and useful timings for:
- request/correlation ID
- API errors and timeouts
- retrieval stages and candidate counts
- agent step count and model calls
- cache hit/miss
- Wikipedia ingestion and DB persistence
- Ollama/model health and generation duration
- streaming disconnect/cancellation

Avoid logging secrets or unnecessary user data.

## 11. Tests and acceptance

### Backend
- Search, Fast Answer, Agentic RAG, sufficiency/abstention.
- Wikipedia fetch, chunking, embedding, PostgreSQL persistence, idempotency, duplicate handling.
- Redis cache behavior, invalidation, outage fallback.
- Timeout, cancellation, malformed external data, and citation validation.

### Frontend
- Search/AI Mode switching and query preservation.
- Loading, retry, disconnect, empty result, citations, suggestions, themes, responsive behavior.
- AI Mode error lifecycle and partial answer handling.

### Integration scenarios
1. Query → existing corpus retrieval → answer.
2. Query → evidence insufficient → Wikipedia fetch → PostgreSQL ingestion → retrieval → answer.
3. Redis cache hit and invalidation after corpus update.
4. Redis unavailable while Search remains usable.
5. Wikipedia unavailable → graceful fallback/abstention.
6. Citation verification and source link correctness.
7. Q8_0 selected in configuration and no silent Q4 fallback.

Clearly label mock-only vs real integration tests. Run lint, typecheck, build, unit, and integration suites where available. Report exact commands/results.

## 12. Implementation sequence

1. Inspect current branch and all source docs; produce a concise audit of remaining gaps.
2. Fix P0 streaming/error lifecycle and correctness defects.
3. Implement Search/AI Mode UI and homepage cleanup.
4. Validate/revise suggested questions.
5. Implement dynamic Wikipedia ingestion and PostgreSQL persistence.
6. Ensure Dense + BM25 retrieval sees newly ingested chunks.
7. Add Redis invalidation/fallback tests and fix stale cache issues.
8. Optimize AI Mode and Agentic RAG while preserving Q8_0.
9. Add observability, security checks, and integration tests.
10. Update docs/README and provide final evidence report.

Do not perform broad unrelated refactors. If a major schema/architecture blocker appears, stop and document the decision required.

## 13. Deliverables

- Audit report with P0/P1/P2, file/function, root cause, repro, fix, acceptance test, risk.
- Implementation report with exact commit(s), changed files, commands, test results, performance before/after, known limitations, and manual target-machine checks.
- Updated docs/configs only where necessary.
- Separate branch; no main merge without user approval.

**Definition of done:** all acceptance criteria are supported by actual code and test evidence; remaining limitations are explicit; Q8_0 remains the primary default; no unverified hardware claims are presented as facts.
