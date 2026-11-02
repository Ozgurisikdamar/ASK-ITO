# ASK-ITO Local Inference Optimization — Redis, Q8, KV Cache & GPU Plan

**Status:** Claude implementation brief / plan (not implementation or benchmark results)  
**Repository review date:** 2026-11-03  
**Scope:** Preserve ASK-ITO as a corpus-bounded Agentic RAG over 258 Wikipedia documents / 5,988 chunks. Local inference only.

## 1. Goal and order

1. Use Redis where it avoids repeated work.
2. Use Ollama-native KV/prefix cache correctly; never store model KV tensors in Redis.
3. Prefer Q8_0 model weights over the documented Q4_K_M baseline. Do not further down-quantize weights to Q6/Q5/Q4/Q3/Q2.
4. After Q8 is configured, run exactly five representative Q4-vs-Q8 questions as a smoke comparison.
5. Then focus on prompt/context length, Agent LLM-call count, and GPU offload.
6. No broad benchmark matrix, overnight benchmark, or RAGAS in this task.

## 2. Repository facts reviewed

- FastAPI backend; React/Vite frontend; PostgreSQL + pgvector; BM25 (bm25s); RRF; optional cross-encoder.
- Default Answer/overview path: hybrid retrieval, optional reranking, one local LLM generation call.
- Sources/search mode does not call the LLM.
- Agentic mode is opt-in and implemented as an explicit Python state machine in backend/app/agent/loop.py.
- Agent configuration currently includes AGENT_MAX_ROUNDS=3, AGENT_MAX_LLM_CALLS=10, AGENT_TIMEOUT_S=180; analysis, rewriting, sufficiency and final-answer roles are present.
- Ollama client is backend/app/llm/client.py; backend/app/llm/metrics.py records token usage, load/prompt/eval timing when available.
- Retrieval has an in-process LRU in backend/app/retrieval/pipeline.py.
- PostgreSQL persistent answer cache already exists (migration 0003); answer-cache module is under backend/app/retrieval/.
- Compose already configures OLLAMA_KEEP_ALIVE=24h, OLLAMA_NUM_PARALLEL=1, OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_CONTEXT_LENGTH=4096.
- README currently describes qwen3.5:9b Q4_K_M, ~3.2 tokens/sec on a 4-vCPU sandbox. This is not target-PC performance.
- Settings include LLM_NUM_CTX=4096, OVERVIEW_CONTEXT_CHUNKS=4, OVERVIEW_CONTEXT_CHARS=700, OVERVIEW_MAX_TOKENS=320, and the Agent call budget.
- Redis is not currently a service/dependency. Add it as a local optional service, not an unconditional dependency that prevents the app from running.
- Model weight quantization and KV-cache quantization are different settings. Current KV cache q8_0 must not be confused with Q8 model weights.

## 3. Redis: use it where justified

Inspect existing cache modules and integrate rather than building parallel cache implementations.

### A. Query embedding cache
- Cache query embeddings by normalized query + embedding model/version + dimensions + query-instruction version.
- Store compact binary vector data (float32 initially).
- Avoid repeat SentenceTransformer embedding for identical queries.

### B. Retrieval cache
- Reuse/promote the current LRU as an optional shared Redis cache. Choose a clear L1/L2 strategy; avoid redundant caches without benefit.
- Key must include normalized query, mode, domain, final_k, corpus/index/BM25 version, embedding model, retrieval settings, reranker model/version, rerank enabled state and relevant thresholds.
- Store stable chunk IDs and scores/trace; hydrate text from the current chunk store.
- Invalidate on corpus/index version changes.

### C. Answer cache
- PostgreSQL answer cache is already durable. Keep it; do not blindly duplicate full answers in Redis.
- Redis may be an optional fast front-cache only if implementation is straightforward and useful.
- Include query, mode, domain, model tag + quantization, prompt version, corpus/index version, retrieval config and source IDs in the key.
- Expose cache-hit kind (Redis/PG/fresh generation) in telemetry.
- Never reuse answers across model, prompt, corpus or retrieval config changes.

### D. Agent tools/state
- Cache deterministic retrieval/tool results, not arbitrary model reasoning across unrelated runs.
- Session/run state needs run/session-scoped keys and TTL; no cross-session leakage.
- Preserve bounded and traceable Agentic RAG.

### E. Document catalog
- Corpus has only 258 documents; existing in-memory chunk store may suffice.
- Do not put every chunk/profile into Redis without evidence of a DB/IO bottleneck.

### Redis requirements
- Redis runs locally in Compose, not hosted/cloud.
- Configurable REDIS_URL and enable flag.
- Redis unavailable: safe warning + fallback to current in-process/PG cache unless explicit strict mode.
- TTL, bounded payloads, hit/miss metrics, serialization and concurrency tests.
- Never store model KV tensors in Redis.

## 4. Q8_0 model weights — no additional weight compression

README currently reports qwen3.5:9b Q4_K_M. User asks to prefer Q8.

- Verify the exact Ollama model tag and manifest quantization before selecting/pulling Q8. Do not invent a tag or label a Q4 model as Q8.
- If no verified Ollama Q8 tag exists, document a supported local import route for a verified Q8_0 GGUF. No silent fallback.
- Keep Q4 baseline available for comparison and rollback.
- Centralize model tag in env/settings; no scattered hardcoding.
- Do not introduce Q6/Q5/Q4/Q3/Q2 as a new recommended weight quantization.
- F16 weights may be a quality-first alternative only if RAM/VRAM permits; requested candidate is Q8_0.
- No cloud/paid inference or cloud fallback.

## 5. KV cache / prefix cache

- KV cache belongs to Ollama/inference backend. Redis must not hold KV tensors.
- Verify pinned Ollama version and supported env variables; do not assume a setting works merely because it is in Compose.
- Compose currently sets OLLAMA_KV_CACHE_TYPE=q8_0. User prefers F16 over additional compression: use F16 KV cache as the initial quality-first setting if supported and memory permits. Keep q8_0 KV cache as an optional memory-saving fallback only if F16 causes actual memory pressure or is otherwise unsuitable.
- Keep Flash Attention only if supported by the current runtime and verify logs.
- Preserve KEEP_ALIVE=24h and NUM_PARALLEL=1 initially.
- Keep static instructions/prefix at the beginning and query/retrieved context after it to maximize prompt-prefix reuse.
- Confirm prefix cache behavior in runtime; mark NOT VERIFIED if not observed.
- Separate cold model load, warm model, first prompt, repeated-prefix, answer cache hit and retrieval cache hit timings.

## 6. Exactly five Q4 vs Q8 smoke questions

After Q8 setup, compare current Q4_K_M baseline and Q8_0 on exactly these five questions:

1. quantum computing
2. What is machine learning?
3. popular AI fields
4. Willow processor 105 physical qubits
5. Hopfield and Hinton, 2024 Physics prize

This is a smoke comparison only, not broad benchmarking.

- Same corpus snapshot, retrieval/source IDs, prompt, temperature and generation limits for both models.
- Disable answer cache for the comparison; warm-up separately and record cold/warm status.
- Record actual model tag/quantization, TTFT, prompt-eval duration/tokens, generation duration, output tokens, tokens/sec, total latency, peak RAM/VRAM if available, answer, citations and grounding verdict.
- Compare factual coverage, citation-ID validity, manual claim support against cited source text, unsupported claims/abstention, latency and memory.
- Do not claim Q8 quality superiority from five samples.
- If target PC/GPU is unavailable, report NOT RUN, not fabricated or inferred.
- If Q8 does not fit or is unacceptably slow, report transparently; do not silently fall back to smaller quantization.

## 7. Next phase: context length, Agent call count, GPU offload

### A. Context/prefill
- Inspect prompt templates under backend/app/agent/prompts/ and context construction in backend/app/agent/answer.py.
- Measure prompt tokens/prefill separately from decode.
- Remove duplicated system instructions, repeated chunks, irrelevant chat history and unnecessary tool output.
- Preserve citation IDs and evidence.
- Fast Answer currently uses 4 passages with char caps; do not alter until Q4/Q8 smoke comparison is recorded.
- Agent should retrieve incrementally; do not append all prior tool output to every prompt.
- Do not blindly shrink LLM_NUM_CTX: context capacity is not identical to prompt length. Optimize actual input tokens and KV memory.

### B. Agent LLM call count
- Keep Agentic RAG, its explicit state machine and trace.
- Audit actual calls: analysis, rewrite, sufficiency, retry and final answer. Current maximum 10 is generous.
- Use deterministic routing/evidence checks where safe: simple query, evidence coverage, duplicate query, already searched subquestion.
- Reuse analysis subquestions; avoid rewriting each one if retrieval quality does not benefit.
- Cache repeated rewrite/sufficiency calls only with complete model/prompt/query keys.
- Reserve final answer call; enforce call budget and timeout.
- Add role-level counts (analysis/rewrite/sufficiency/answer) and compare baseline/after; no broad benchmark.
- Do not turn Agentic RAG into classic one-call RAG or remove AI Mode.

### C. GPU offload
- Detect/report CPU, RAM, GPU, VRAM, Ollama model details and actual offload state.
- If NVIDIA GPU exists, inspect Docker GPU passthrough and Ollama runtime. If absent, explicitly say CPU-only.
- For Windows/Docker Desktop, document verified GPU setup and checks (ollama ps/logs/GPU utilization).
- Apple Silicon, AMD and Linux must use the appropriate supported runtime path; do not assume CUDA universally.
- Confirm layer placement and VRAM usage; do not claim full GPU offload unless verified.
- Keep local CPU fallback. Never introduce cloud inference.

## 8. Telemetry, constraints, acceptance

- No broad benchmark matrix, overnight run, or RAGAS.
- The five-question Q4/Q8 comparison above is the only model comparison requested.
- Reuse existing instrumentation; separate cache-hit timings from inference.
- Distinguish TTFT, prefill, decode, tokens/sec, prompt/completion tokens, hardware, quantization and cold/warm status.
- Mark unavailable target-PC/GPU measurements NOT RUN.
- Add tests for Redis fallback, TTL, key versioning/invalidation, answer citation retention, Q8 model config, KV config, agent call budget, timeout/cancellation and GPU diagnostic/fallback.
- Preserve overview/Answer, search/Sources, agentic/AI Mode and evaluation modes.
- Run existing backend tests, lint, frontend typecheck/build; report actual results only.
- Do not run RAGAS or extensive benchmark.

## 9. Deliverables
1. Implement on a dedicated branch; do not merge main.
2. Keep Q4 baseline and Q8 candidate.
3. Add report: docs/ASK_ITO_REDIS_Q8_KV_GPU_OPTIMIZATION_REPORT.md
4. Include branch, commit SHA(s), changed files, tests and the five-question comparison.
5. Clearly list NOT RUN items and target hardware requirements.
