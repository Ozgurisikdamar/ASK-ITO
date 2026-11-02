# ASK-ITO Fast RAG — Independent Review & Remaining Work

**Review basis:** `docs/ASK_ITO_FAST_RAG_IMPLEMENTATION_REPORT.md`, README, and inspected implementation files on the repository default branch as of 2026-11-03. This is a code/report review, not a claim that I independently reran the tests or benchmarks.

## Verdict

The Fast RAG simplification is directionally correct and materially improves the architecture: Search mode avoids generation, Answer mode is a single-call grounded RAG path, Agentic remains opt-in, and RAGAS is offline.

**Do not call the project “fast” yet for CPU-only local answer generation.** The report’s own measured Fast Answer p50 is about 64.9 s on its 4-vCPU sandbox. Retrieval is fast without reranking; the LLM and CPU reranker dominate. This is a valid, honest report, but the interactive latency goal is not met.

The implementation is a useful prototype/demo, not yet a polished low-latency user experience.

## Findings

### P0 — Local-only policy does not mean loopback-only

The earlier requirement was local inference only, without LAN/private-network or hosted endpoints. Current `backend/app/llm/policy.py` allows loopback **and private-network and link-local** addresses. This permits a LAN host (e.g. a private IP) and is broader than the strict localhost-only requirement.

**Required fix:**
- For `LLM_LOCAL_ONLY=true`, accept only loopback addresses: `localhost`, `127.0.0.1`, `::1` (and equivalent IP representations).
- Reject RFC1918 private IPs, link-local, wildcard, multicast, public IPs, and hostnames resolving to a mixture of loopback and non-loopback.
- Revalidate redirects / resolved target at request time where relevant; avoid DNS rebinding / validation-to-connection mismatch if the HTTP client follows redirects.
- Add tests for localhost, IPv4/IPv6 loopback, Docker service hostname requirements, private LAN IP, public IP, link-local, wildcard, mixed DNS and malformed URLs.
- **Docker caveat:** docker-compose may use service name `ollama`, which resolves to a private bridge IP. If this is required, make the trust boundary explicit and configurable (e.g. separate trusted Docker-network mode), rather than silently calling all private networks “local”. Default strict mode should be loopback-only.

### P1 — Latency remains unacceptable for interactive Fast Answer on CPU

Report figures:
- Fast Answer p50 ~64.9 s, p95 ~73.9 s (12-question subset).
- LLM TTFT roughly 19–29 s; generation ~3.2 tokens/s.
- BGE reranker ~7.6–8.8 s/query.
- Sources mode ~8 s with reranker; ~0.1 s with reranker disabled.

**Required product decision and follow-up:**
1. Make Sources/Search truly responsive by default on CPU. Consider `FAST_RERANK=false` as default on CPU-only deployments, or expose explicit “fast retrieval” setting. Keep quality-first BGE option.
2. Add a UI option to search without generating an answer, and make latency expectations clear before generation.
3. Keep answer generation opt-in (button/toggle) or clearly label it as a long-running local operation on CPU.
4. Evaluate a smaller local answer model (e.g. Qwen 3B/4B class, if available in the selected runtime) on a small gold set before choosing; do not assume it is sufficiently good.
5. Compare first-token latency and tokens/sec on the actual user PC, not only the sandbox. A 9B CPU model may remain slow regardless of retrieval optimization.
6. Do not advertise “few seconds” or claim the target is met until target hardware measurements demonstrate it.

### P1 — Citation validation only validates citation IDs, not claim support

`validate_citations()` removes out-of-range citation markers, but it does not establish that each claim is supported by the cited passage. Report raw citation validity = 1.0 means citations point to retrieved source IDs; it does **not** mean factual correctness or entailment.

**Required fix / documentation:**
- Rename/clarify metrics as “citation ID validity” vs “claim-level citation support”.
- Add evaluation for claim-level support (manual annotation or offline local judge), explicitly not in serving path.
- Add tests for valid but irrelevant citation, uncited claims, and citation references to a passage that does not support the sentence.
- UI should not imply citation IDs guarantee correctness.

### P1 — Evidence threshold is reranker-specific and unanswerable cases can pass

Report indicates main-set unanswerable questions sometimes score 0.86–0.99 and therefore pass the 0.3 threshold. Threshold only filters obvious off-topic queries; near-domain cases rely on LLM abstention.

**Required fix:**
- Document this as a weak OOD gate, not calibrated confidence.
- Add threshold calibration set with answerable, near-domain unanswerable, and truly out-of-domain examples.
- Do not raise threshold blindly (risk false rejection).
- If switching to MiniLM, disable or recalibrate the BGE score threshold; raw logits are not comparable.
- Consider adding a separate retrieval confidence/evidence coverage heuristic (e.g. multiple distinct passages/documents, margin/score distribution) and evaluate false accept/false reject rates.

### P1 — Answer quality evaluation is too small and not independent

- End-to-end LLM quality was measured on 12 questions once.
- Manual review of six questions was done by the implementer, not independent.
- RAGAS was NOT RUN.
- Only two unanswerable questions are in the main eval set.
- The broad “popular AI fields” query is acknowledged to have partial coverage.

**Required fix:**
- Build a compact but balanced answer-quality set: direct fact, broad synthesis, multi-chunk, paraphrase, ambiguous, near-domain unsupported, out-of-domain.
- Add expected source/document labels and answerable/unanswerable label.
- Run at least 2–3 repeated answers for the small set or report stochasticity/variance (even if temperature=0, backend may vary).
- Manual review should be done by a person other than the implementation author if possible.
- RAGAS is optional/offline; do not block fast path on expensive RAGAS, but report clearly as NOT RUN.

### P2 — Timing/caching semantics need clearer separation

Report notes prompt-prefix caching changes repeat timings. Ensure benchmarks distinguish:
- cold model load;
- warm model, first prompt;
- repeated identical prompt/prefix-cache hit;
- answer-cache hit;
- retrieval-cache hit.

The UI should indicate cache hits. Do not compare cached and uncached requests as equivalent. Check that cache invalidation keys include model, prompt version, evidence threshold, reranker config and corpus/index version. In particular, answer cache must not return an answer generated by a different model/config after runtime config changes.

### P2 — Error and security hygiene

- API route catches exceptions and returns `str(exc)`; check that this cannot leak endpoint details, filesystem paths, or secrets. Return a safe error ID/message to users and log details server-side.
- Validate URL schemes (only http/https), credentials embedded in URLs, redirects, and ports in local-only policy.
- Ensure generated prompt/context does not allow corpus text to override system/developer instructions (prompt-injection tests for retrieved Wikipedia text).
- Add request size/rate limits and cancellation behavior tests for long LLM streams.

### P2 — Product scope / corpus limitations

Current corpus is only 258 Wikipedia documents / 5,988 chunks, with specific domains (Nobel, quantum, AI). The project is **not a general Wikipedia search engine**.

**Required UX/docs:**
- Show corpus scope and snapshot date prominently.
- For “popular AI fields”, clearly frame the answer as “fields represented in this corpus / retrieved sources,” not a comprehensive ranking.
- Add an “expand corpus” or ingest pipeline later; do not imply complete Wikipedia coverage.
- Consider exact title search and document-level results in Search mode for users who type article titles.

## Acceptance checklist for the follow-up

- [ ] Strict local-only endpoint policy verified with tests (or explicitly document Docker network exception).
- [ ] Search/Sources mode on target machine latency measured without reranker and with MiniLM/BGE options.
- [ ] User can choose whether to run LLM answer generation.
- [ ] Actual user-PC Qwen model latency measured (TTFT, tokens/s, total).
- [ ] Citation validity metric wording corrected; claim support evaluation added or explicitly deferred.
- [ ] Evidence cutoff documented as heuristic; threshold calibration and near-domain tests added.
- [ ] Answer-quality set expanded; limitations stated.
- [ ] Cache hit/cold/warm timings separated and cache invalidation audited.
- [ ] Prompt injection and safe API error tests added.
- [ ] README describes corpus coverage/snapshot and avoids general-Wikipedia claims.
- [ ] Tests, typecheck, lint and build run; results recorded truthfully.

## Suggested implementation priority

1. **P0:** strict local-only endpoint policy.
2. **P1:** CPU product UX (Search fast by default; Answer generation explicit/opt-in).
3. **P1:** target-machine latency and smaller-model comparison.
4. **P1:** citation claim support + abstention calibration.
5. **P2:** cache/error/prompt-injection hardening and corpus UX.

## Do not do

- Do not run another overnight benchmark matrix at this stage.
- Do not migrate from pgvector to Qdrant for this corpus.
- Do not remove Agentic mode.
- Do not claim citations are verified merely because citation IDs are valid.
- Do not claim the 3–12 s answer target is met on CPU; current reported numbers contradict that.
- Do not use hosted/paid APIs or cloud fallback.
