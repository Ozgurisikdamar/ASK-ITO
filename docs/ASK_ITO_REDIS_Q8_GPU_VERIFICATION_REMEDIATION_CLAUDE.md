# ASK-ITO Redis/Q8/GPU Optimization — Verification & Remediation

**Purpose:** Complete verification and evidence for the implementation already merged into `main`. This is not a request for another broad optimization pass.

## Context

The following implementation commits are already on `main`:
- `ca2c1b7` — Redis cache integration
- `57a4396` — Q8_0 model / F16 KV cache / Compose configuration
- `5add4a5` — Agentic RAG duplicate-search and sufficiency guards
- `9f59ea7` — Q4/Q8 smoke script, agent audit, hardware diagnostic

The expected report `docs/ASK_ITO_REDIS_Q8_KV_GPU_OPTIMIZATION_REPORT.md` was not found on main during review. The code exists, but the actual five-question comparison and target-hardware verification are not yet evidenced.

## Objective

Audit the existing implementation, run only the requested targeted tests that are feasible in the current environment, fix concrete bugs found, and produce a truthful final report. Avoid unnecessary refactors.

## Priority 0 — Repository/code verification

Read and verify all changed files from the implementation commits, including:
- Redis cache adapters, serialization, TTL, cache keys, invalidation and fallback.
- Query embedding cache, retrieval cache, answer cache, agent tool cache.
- Redis Compose healthcheck, startup ordering and shutdown behavior.
- Ollama Q8 model configuration, model tag and quantization identity.
- KV cache configuration and runtime verification.
- Agent state machine changes and call budget.
- Q4/Q8 smoke script, hardware diagnostics and agent audit scripts.

Check for:
1. Redis failure/timeout handling and fallback (including Redis unavailable at startup).
2. Cache key includes all relevant query/model/prompt/corpus/index/retrieval/reranker settings.
3. Cache invalidation and TTL correctness.
4. No cross-session or cross-user Agent state leakage.
5. Answer cache preserves citations and cannot reuse stale model/prompt/corpus answers.
6. No Redis use for model KV tensors.
7. Q4/Q8 script truly uses the intended model tags, same retrieved source IDs/context, same prompt and generation settings, and disables answer cache during comparisons.
8. Script handles missing model, missing Ollama, timeout, cancellation and out-of-memory clearly; never silently substitutes a different quantization.
9. Hardware diagnostic accurately distinguishes CPU, GPU, VRAM, Ollama model residency and actual offload; do not claim GPU offload based only on GPU presence.
10. Agent guards do not suppress necessary multi-hop retrieval, valid retries or answer generation.
11. No secrets, endpoint URLs with credentials, or internal exception details leak in API output/log artifacts.

Fix concrete defects with focused commits. Do not rewrite architecture.

## Priority 1 — Tests

Run actual relevant tests and report exact commands/results:
- Redis cache unit/integration tests (TTL, key dimensions, invalidation, cache miss, Redis unavailable fallback).
- Retrieval/answer cache tests and citation retention.
- Agentic RAG tests (simple, multi-hop, duplicate query, sufficiency retry, max call budget, timeout).
- Q4/Q8 smoke script validation/unit tests.
- Backend test suite.
- Backend lint/type checks if configured.
- Frontend typecheck and build if applicable.

If Redis or Ollama is unavailable in the execution environment, report those integration tests as **NOT RUN**; do not replace them with mocked tests and claim integration success. Mocked/unit tests may be reported separately.

## Priority 2 — Five-question Q4/Q8 smoke comparison

Run only these five exact questions:
1. `quantum computing`
2. `What is machine learning?`
3. `popular AI fields`
4. `Willow processor 105 physical qubits`
5. `Hopfield and Hinton, 2024 Physics prize`

Constraints:
- No broad benchmark matrix, overnight test, or RAGAS.
- Same corpus snapshot, retrieval output/source IDs, prompt, temperature, top-p, context and max generation tokens for Q4 and Q8.
- Answer cache disabled.
- Record model tag and actual quantization, cold/warm status, TTFT, prompt/prefill time and token count, generation duration, output tokens, tokens/sec, total latency, RAM/VRAM peak if available, answer, citation IDs, and claim-level support review.
- If target PC/GPU is not available, clearly mark target-machine results **NOT RUN**. Do not infer target results from another machine.
- If this environment supports only one model or cannot fit Q8, record the exact limitation and stop; do not silently fall back to lower quantization.
- This is a smoke comparison, not proof of statistical quality superiority.

## Priority 3 — Agent call-count evidence

Use a small fixed set of representative queries, including:
- a simple factual query
- a multi-hop query
- a comparison query
- a near-domain unsupported query

No broad benchmark. Record baseline/current per-role LLM calls (analysis, rewrite, sufficiency, answer), retrieval/tool calls, rounds, final result and latency. If baseline commit is unavailable or cannot be run, explicitly state that comparative reduction is **NOT MEASURED**. Do not claim calls were reduced based solely on code inspection.

## Priority 4 — Redis cache evidence

Do not benchmark extensively. Demonstrate correctness with:
- first request cache miss
- identical repeat cache hit
- version/config change invalidates key
- Redis unavailable falls back correctly
- TTL expiration behavior

Report whether these were integration-tested against actual Redis or unit-tested with a mock.

## Final report

Create/update:
`docs/ASK_ITO_REDIS_Q8_KV_GPU_OPTIMIZATION_REPORT.md`

Report structure:
1. Commit/branch and code audit summary
2. Issues found and fixes (with commit SHAs)
3. Test command, exact pass/fail counts and any failures
4. Redis test matrix (actual Redis vs mock clearly labeled)
5. Q4/Q8 five-question result table
6. Hardware/runtime facts and GPU offload evidence
7. Agent call-count table
8. Quality/citation review and limitations
9. Explicit NOT RUN items and why
10. Follow-up actions, if any

## Rules

- Local-only inference. No paid/cloud LLM APIs or cloud fallback.
- Do not add another model or change embedding model.
- Do not change Q8 to Q6/Q5/Q4 or other smaller weight quantization.
- Do not change model/prompt/context defaults merely to make results look better.
- Do not claim a test, benchmark, GPU offload, speedup or quality improvement unless the relevant evidence exists.
- No large benchmark matrix or RAGAS.
- Do not merge any new branch to main without user authorization. If fixes are needed, use a separate branch and push it.
- If the existing implementation is correct, avoid unnecessary code changes and focus on reproducible verification/reporting.

## Completion response

Provide:
- branch name
- commit SHA(s)
- report link
- test results
- Q4/Q8 smoke results or NOT RUN
- GPU offload status or NOT VERIFIED
- remaining blockers and exact steps for the user to run locally
