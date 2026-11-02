# ASK-ITO Local LLM Latency & Quality Optimization — Claude Implementation Plan

> **Goal:** Reduce ASK-ITO's end-to-end latency on a CPU-only Windows/Linux machine (Intel CPU, 36–48 GB RAM), while preserving or improving retrieval quality, grounded answers, citation correctness, and reproducibility.
>
> **Execution rule:** Claude must inspect the current code before editing, implement in small phases, run tests after each phase, record real measurements, and push both implementation and a final evidence-based report to GitHub. Do not mark a phase complete unless its acceptance criteria are met. Do not fabricate benchmark numbers or claim tests passed if they were not run.

## 0. Repository context and current known baseline

Repository: `Ozgurisikdamar/ASK-ITO`

Known architecture:
- FastAPI backend + PostgreSQL/pgvector.
- Hybrid retrieval: BM25 + vector retrieval + RRF.
- Cross-encoder reranker.
- Agentic RAG loop with analysis, query/sub-question handling, sufficiency checks, retrieval rounds, answer generation and citation validation.
- OpenAI-compatible LLM client; local Ollama endpoint is the default.
- Embeddings: `BAAI/bge-base-en-v1.5` (768-dim).
- Reranker default: `BAAI/bge-reranker-base`.
- Evaluation set in latest known report: 46 questions.

Latest known `docs/EVAL_REPORT.md` baseline (2026-10-30; re-run before using it as a comparison):
| Mode | p50 latency | p95 latency | Avg LLM calls |
|---|---:|---:|---:|
| hybrid | 94 ms | 122 ms | 1 |
| hybrid_rerank | 8,235 ms | 13,521 ms | 1 |
| hybrid_rerank_rewrite | 7,987 ms | 8,898 ms | 1 |
| agentic | 10,152 ms | 11,688 ms | 6 |

Retrieval metrics from the same report:
- Hybrid doc Recall@5: 0.716; hybrid_rerank: 0.795.
- Hybrid doc MRR@10: 0.657; hybrid_rerank: 0.741.
- Hybrid doc nDCG@10: 0.676; hybrid_rerank: 0.747.
- Agentic chunk Recall@5: 0.398; hybrid_rerank: 0.432.
- RAGAS metrics were unavailable in the old report because the judge was in mock mode.

Treat these as **historical baselines only**. Validate current code, dataset, runtime and hardware before comparing.

## 1. Hard requirements / non-negotiable invariants

1. **No regression hidden by speed:** evaluate quality and latency together.
2. Never remove the reranker or change embeddings/index as a speed hack without a controlled ablation.
3. Do not change embedding model/dimension by default. A change may require re-embedding/re-indexing the full corpus and must be a separately documented experiment.
4. Keep the original baseline configuration reproducible.
5. Keep the mock LLM mode and deterministic unit tests working.
6. Preserve citations, citation validation, traceability and refusal behavior for unsupported/unanswerable questions.
7. Do not silently truncate prompts or retrieved context. If trimming is added, log what was trimmed and test citation/context correctness.
8. Do not silently swallow exceptions. Errors should be observable and actionable; never expose secrets or full internal exception details in public API responses.
9. No API keys, local paths, personal data, or secrets in committed files. Update `.env.example` only with placeholders.
10. Keep configuration backward compatible where feasible. Document every new env variable and its default.
11. **Strictly local and zero-cost API policy:** Do not use Grok/xAI, OpenAI, Anthropic, Gemini, any other paid/hosted LLM API, or any cloud LLM endpoint. Do not request or configure API keys. All inference, including answer generation, judge/RAGAS, rewriting, and agent decisions, must use locally running models through Ollama (or another local runtime already supported by the project). No paid API calls.
12. If hardware/model downloads are unavailable in Claude's execution environment, clearly label those experiments **NOT RUN** and provide commands for the user to execute. Do not simulate local model performance.

## 2. Phase 0 — Audit and establish reproducible baseline (no code changes yet)

### Tasks
- Read the full relevant implementation and current docs:
  - `backend/app/config.py`
  - `backend/app/llm/client.py`
  - `backend/app/agent/loop.py`
  - `backend/app/agent/answer.py`
  - `backend/app/agent/rewrite.py`
  - `backend/app/agent/schemas.py`
  - `backend/app/retrieval/pipeline.py`
  - `backend/app/retrieval/reranker.py`
  - `backend/app/api/search.py`
  - `backend/app/api/health.py`
  - evaluation runner, metrics, tests, `Makefile`, Docker Compose, `.env.example`, `docs/EVAL_REPORT.md`, `docs/DECISIONS.md`.
- Confirm current git branch and working tree. Do not overwrite unrelated changes.
- Inspect how LLM calls are counted, role-specific model selection works, and how timeouts/cancellation behave.
- Run existing test suite and evaluation on the current baseline before modifying code.
- Record exact hardware/OS, CPU model/core/thread count, RAM, Python version, Ollama version, model tags and quantization, Docker status, DB/index state, and environment settings (redact secrets).
- Confirm whether latency measurements include cold-start/model loading or only warm requests. Report both where possible.

### Deliverables
- Baseline test results.
- Baseline evaluation results, including actual measured latency and quality.
- Short audit findings: bottlenecks, suspected causes, and what can/cannot be optimized safely.

### Acceptance criteria
- Existing tests are run and their status is reported.
- Baseline can be reproduced from documented commands.
- Historical values are not copied into the new report as if newly measured.

## 3. Phase 1 — Model configuration and Ollama compatibility

### Candidate models
Evaluate these model candidates, using the same quantization where possible:
- Primary candidate: **Qwen3.5 9B** (Ollama model tag must be verified against the official/current Ollama registry; do not assume a tag exists).
- Comparison candidate: **Qwen3 8B**.
- Speed-oriented optional candidate: Qwen3.5 4B, only if the exact Ollama tag is available.
- Existing configured model, if different, must remain a baseline.

Do not assume that model generation, parameter count or published benchmark implies local CPU throughput. Measure on the actual target hardware.

### Tasks
- Verify model tags and Ollama model metadata before creating config examples.
- Ensure `LLM_DEFAULT_MODEL`, `LLM_AGENT_MODEL`, `LLM_ANSWER_MODEL`, and `LLM_JUDGE_MODEL` are supported and their fallback semantics are documented.
- Check OpenAI-compatible request payload compatibility with Ollama. Avoid adding unsupported request parameters without capability checks.
- If configurable Ollama inference options are introduced (e.g. `num_ctx`, `num_predict`, `num_thread`, keep_alive), implement them in a clean config layer and add tests. Do not hardcode thread count to 8: derive or tune based on actual Intel CPU.
- Use an initial context window appropriate to the task (e.g. 8k as a test candidate), then evaluate larger contexts only if needed. Record context length and generation limits.
- Keep API keys and local endpoint config in `.env`; never commit `.env`.

### Required model experiments
At minimum:
- Existing model/configuration (baseline).
- Qwen3:8B.
- Qwen3.5:9B.
- Optional Qwen3.5:4B if available and relevant.

For every model record: exact Ollama model tag, digest if available, quantization, context setting, CPU threads, RAM footprint, prompt tokens, generated tokens, tokens/sec, time-to-first-token if available, total generation time, answer quality and error rate.

## 4. Phase 2 — Instrumentation before optimization

Add or improve instrumentation without changing answer semantics.

### Required metrics per search request
- Total end-to-end duration.
- Retrieval duration (BM25, vector, RRF/hybrid, reranking separated where possible).
- LLM calls by role: analyze, rewrite, sufficiency, final answer, judge.
- Duration per LLM call and total LLM duration.
- Model name per role (no secrets).
- Agent rounds and sub-question count.
- Candidate counts before/after reranking and final context/chunk count.
- Prompt/context token count when the provider returns it; otherwise clearly mark as estimated/unknown.
- Timeout/error/fallback count.
- Cold/warm status where determinable.

### Requirements
- Use structured logs or trace objects consistent with existing project patterns.
- Do not expose sensitive prompt content or secrets in production logs.
- Keep tests stable; avoid asserting exact wall-clock timings in unit tests.
- Add unit tests for instrumentation/counting and tests that ensure instrumentation does not alter output semantics.

## 5. Phase 3 — Agent loop optimization (primary latency target)

The historical agentic mode made ~6 LLM calls/query and took ~10 seconds p50. Optimize the number and cost of calls first; do not blindly lower limits.

### Audit current loop
- Analyze exact LLM-call path and call counts per stage.
- Check if a simple question unnecessarily runs analysis, rewrite, sufficiency and multi-round retrieval.
- Check whether each sub-question causes a full rerank and whether duplicate candidates are reranked.
- Check if the same large context is repeatedly included in prompts.
- Check whether final answer is generated when no meaningful context is available.
- Check prompt length, duplicated instructions/context, and avoidable serialization overhead.

### Desired behavior
Implement an **adaptive** flow:
1. Initial query validation and retrieval.
2. Use a lightweight/cheap decision path for straightforward queries when safe.
3. Only perform analysis/sub-question generation when query complexity warrants it.
4. Avoid rewrite when the original query is already clear.
5. Perform an additional retrieval round only when a measurable sufficiency/coverage signal indicates it is needed.
6. Merge/deduplicate candidates across rounds before reranking when semantically and technically safe.
7. Generate one grounded final answer with citations.
8. Validate citations and unsupported claims as currently required.

### Guardrails
- Do not hardwire “always one round” if that harms multi-hop questions.
- Keep maximum rounds/calls/time configurable.
- Ensure at least one call is reserved for final answer where applicable.
- Define deterministic fallback behavior when LLM budget is exhausted.
- Avoid parallel LLM calls if that would overload the CPU or create memory pressure. Benchmark concurrency before enabling.
- If adding caching, key by all relevant inputs (model, prompt/version, temperature, context, query) and make cache invalidation explicit. Do not cache user-specific/private data inappropriately.
- Use query/result caching only if freshness and evaluation behavior are clearly defined.

### Experiments (one change at a time)
- Baseline current loop.
- Disable or skip rewrite only for clearly simple queries.
- Adaptive sufficiency / one extra round maximum for normal questions.
- Deduplicated retrieval and rerank pool.
- Prompt/context trimming (only if grounded and validated).
- Each experiment must be independently measurable and reversible.

## 6. Phase 4 — Reranker optimization (secondary latency target)

Historical hybrid retrieval was ~94 ms, while hybrid + reranker was ~8.2 seconds. Determine whether this delay is model initialization, CPU cross-encoder inference, repeated instantiation, or other overhead.

### Tasks
- Verify that the CrossEncoder is not loaded repeatedly per request. If it is, implement safe singleton/lifespan initialization with correct tests and shutdown behavior.
- Measure cold initialization separately from warm inference.
- Measure reranking latency vs candidate pool size (e.g. 5, 10, 20, 30) on the real CPU.
- Test candidate pool reduction only after establishing a quality curve.
- Optional comparison: `cross-encoder/ms-marco-MiniLM-L-6-v2` (confirm license, model availability and compatibility first).
- Do not switch reranker model solely based on parameter size; evaluate retrieval metrics and answer correctness.
- Consider batch size / truncation length only if supported and quality-tested.

### Acceptance criteria
- Warm reranker latency is measured separately from model load time.
- Retrieval quality metrics are compared across pool sizes/models.
- No unacceptable regression in Recall@5, MRR@10 or nDCG@10.

## 7. Phase 5 — Evaluation design (mandatory)

Use the existing evaluation set; confirm current count and dataset integrity. Use the same dataset and DB/index snapshot across experiments.

### Retrieval metrics
- Document Recall@5.
- Chunk Recall@5.
- Document MRR@10.
- Document nDCG@10.
- Include existing metrics as appropriate; do not silently change metric definitions.

### Answer metrics
- RAGAS faithfulness.
- Answer relevancy.
- Context precision.
- Context recall.
- Citation correctness / citation coverage (add a deterministic metric if one does not exist).
- Unsupported-claim / hallucination audit using a documented method.
- Unanswerable-question refusal correctness.

**RAGAS judge must be a real locally running model, not mock and not a cloud API.** Use a local Ollama judge model, ideally a separate role/config if resources permit. Report judge model and configuration. If local judge is unavailable or CPU/RAM limits prevent running it, label metrics unavailable and do not claim quality improvement. Do not fall back to a hosted API.

### Latency / operations
- p50 and p95 end-to-end latency.
- Per-stage latency and total LLM time.
- Average and p95 LLM calls per query.
- CPU utilization, peak RAM and model memory if measurable.
- Cold-start vs warm-request latency.
- Throughput only if concurrency testing is performed.

### Statistical discipline
- Run at least 3 repetitions for latency measurements where feasible.
- Report mean/median and variance or range; avoid comparing a single noisy run.
- Use same machine, model state, context settings and test dataset.
- Do not compare different model configurations on different hardware without marking the result as non-comparable.

## 8. Phase 6 — Benchmark matrix and decision criteria

Run these variants (add columns if additional experiments are needed):

| ID | LLM | Agent strategy | Reranker | Purpose |
|---|---|---|---|---|
| A | Existing model | Existing | Existing | Reproduced baseline |
| B | Qwen3:8B | Existing | Existing | Model comparison |
| C | Qwen3.5:9B | Existing | Existing | Model comparison |
| D | Best model so far | Adaptive agent | Existing | Agent-loop improvement |
| E | Best model so far | Adaptive agent | smaller pool / candidate reranker | Reranker optimization |
| F | Best local model | Final optimized path | Final reranker | Candidate release configuration |
| G (optional) | Another locally installed model | Final optimized path | Final reranker | Additional local model comparison |

All experiments must remain local. Do not use Grok/xAI or any hosted API, even as an optional fallback. Do not add cloud model code, cloud credentials, or cloud provider configuration. If the project currently has a cloud provider implementation, do not invoke it; keep the optimization scope local-only.

### Default decision policy
Recommend the configuration with the best quality/latency trade-off, not the lowest latency alone.

Unless project owner approves otherwise, candidate release should:
- Preserve doc Recall@5 and MRR@10 within 2 percentage points of reproduced baseline, OR clearly document an approved quality trade-off.
- Not reduce faithfulness/citation correctness/unanswerable refusal quality.
- Improve agentic p50 meaningfully (target at least 30% reduction as a goal, not a guarantee).
- Show acceptable p95, memory and CPU behavior on target machine.
- Maintain offline/local operation with Ollama.

If quality and latency conflict, present the Pareto trade-off and do not silently choose the faster but worse model.

## 9. Phase 7 — Tests and validation

Run after every implementation phase and before final push:

### Unit tests
- Config defaults, env override, role fallback.
- LLM client request payload, timeout and errors (mock HTTP).
- LLM call counting and budgets.
- Agent round/call limit enforcement.
- Citation validation and answer generation.
- Reranker caching/singleton lifecycle and candidate order.
- Context truncation/deduplication.
- Mock mode determinism.
- Health endpoint behavior.

### Integration tests
- API search in mock mode.
- Hybrid retrieval.
- Hybrid + reranker.
- Agentic path with mocked LLM.
- LLM provider connectivity smoke test when local Ollama is available.
- Database-backed tests if existing test conventions require DB.

### Regression tests
- Existing test suite.
- Existing eval runner.
- Verify response schema and trace schema compatibility.
- Verify non-agentic search remains functional.
- Verify no regression to citations, source links or refusal behavior.

### Quality checks
- Run project’s documented lint/type-check/test commands (inspect Makefile/pyproject; do not invent if absent).
- Report command and exit code for each.
- Never change tests simply to make them pass unless the test is demonstrably invalid and the reason is documented.

## 10. Phase 8 — Documentation and final report

Create/update:
- `docs/ASK_ITO_LOCAL_LLM_OPTIMIZATION_REPORT.md` — full report.
- `.env.example` — only if variables/defaults changed.
- `docs/DECISIONS.md` — decision record if architecture/config decisions are made.
- README only if usage/setup instructions materially change.

### Final report must include
1. Executive summary and final recommendation.
2. Exact CPU/RAM/OS and environment.
3. Git commit SHA and code changes.
4. Baseline and final configuration tables.
5. Model IDs/digests/quantization/context/thread settings.
6. Full benchmark matrix with p50/p95, token throughput, calls, RAM/CPU and quality metrics.
7. Tests: command, status, and relevant output summary.
8. Quality regression analysis; include failures and limitations.
9. What was not tested and why.
10. Reproduction commands for the user.
11. Rollback instructions.
12. Explicit recommendation: Qwen3:8B vs Qwen3.5:9B vs other tested options, supported by measured evidence.
13. Confirm that all measured inference and evaluation remained local and no paid/cloud API was called.
14. Open issues and next steps.

## 11. Git workflow and handoff

- Do work on a dedicated branch, e.g. `claude/ask-ito-llm-optimization`, unless repo owner explicitly requests otherwise.
- Keep commits logically separated (instrumentation, agent optimization, reranker optimization, tests/docs).
- Do not commit model weights, Ollama blobs, datasets with sensitive content, secrets, generated caches, or machine-specific artifacts.
- Push branch to GitHub.
- Open a PR or provide the branch and commit SHA; do not merge automatically unless explicitly requested.
- Ensure final report is committed and pushed.
- In final response to the repository owner, include:
  - branch + PR (if created) + commit SHA,
  - changed files,
  - test commands/results,
  - benchmark report path,
  - any tests not run and reason,
  - any manual steps still required.

## 12. Recommended initial local commands (Windows PowerShell)

First record hardware:
```powershell
Get-CimInstance Win32_Processor |
  Select-Object Name, NumberOfCores, NumberOfLogicalProcessors

Get-CimInstance Win32_ComputerSystem |
  Select-Object @{Name="RAM_GB";Expression={[math]::Round($_.TotalPhysicalMemory / 1GB, 1)}}
```

Check Ollama:
```powershell
ollama --version
ollama list
ollama ps
```

Pull candidate models only after checking available disk space and current model tags:
```powershell
ollama pull qwen3:8b
# Verify the current Ollama registry tag for Qwen3.5 9B before pulling.
```

Use warm-up requests before collecting warm latency. Save raw results (JSON/CSV) under a documented evaluation output directory, but do not commit huge generated files or model artifacts.

## 13. Claude execution checklist

- [ ] Read current source and repo instructions first.
- [ ] Establish baseline and record hardware.
- [ ] Confirm model tags and quantization.
- [ ] Instrument per-stage latency and LLM calls.
- [ ] Optimize agent loop in isolated changes.
- [ ] Measure reranker initialization/inference and candidate pool.
- [ ] Run retrieval and answer quality evaluations with real judge.
- [ ] Run all unit/integration/regression tests.
- [ ] Review diff for secrets, accidental model/data artifacts and API compatibility.
- [ ] Write complete report with actual data.
- [ ] Push branch and report commit/PR details.
- [ ] Clearly mark any hardware-dependent experiments that were not run.

**Do not stop after code changes. The task is complete only when tests, evaluations, documentation and GitHub handoff are complete or when blockers are explicitly reported.**
