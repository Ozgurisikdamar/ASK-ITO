# Ask ITO — Fast RAG Implementation Report

Implements `docs/ASK_ITO_FAST_RAG_SIMPLIFICATION_CLAUDE.md`.
Branch: `fast-rag`.
All numbers below were measured in this work, on the hardware listed in §3, unless they are explicitly labelled otherwise.
Where something was not run, it says **NOT RUN**.

## 1. What changed and why

Ordinary questions previously had two problems:
- They either ran the multi-step agent (5–7 LLM calls),
- or they used `hybrid_rerank` with a 30-candidate, 512-token reranker and a long prompt. On a CPU-only machine that meant minutes, or about 1.5 minutes, per answer.

The default is now one explicit fast path.
The existing M10 `overview` mode is reused, not duplicated.

| Change | Where |
|---|---|
| **Fast Answer** (`overview`, default): BM25 + pgvector → RRF → optional bge cross-encoder (top 20 candidates, 512 tokens) → evidence cut-off → **exactly one** local LLM call. No analysis, rewrite or sufficiency call. | `app/retrieval/pipeline.py`, `service.py`, `api/stream.py` |
| **Sources** mode (`search`): the same retrieval, ranked passages only, **no LLM call** (JSON and SSE). | same |
| `FAST_RERANK` switches the reranker in the fast modes; `FAST_TOP_K` sets candidates per channel. The evaluation modes keep their fixed behaviour. | `pipeline.uses_reranker`, `channel_k` |
| **Evidence cut-off** `FAST_MIN_EVIDENCE_SCORE=0.3` (bge probability). Passages below it are not used as evidence. If none remain, the answer is "I could not find this in the sources." and the LLM is not called. | `service.evidence_filter` |
| Query embedding computed once and timed separately from the pgvector query. Context assembly, time to first token (TTFT) and generation are timed. Model and reranker names are returned. | `pipeline.candidates`, `search.vector_search`, `stream.produce` |
| Default answer model `qwen3.5:9b` (`LLM_DEFAULT_MODEL`). Thinking off (sent only to models that report the capability), temperature 0, max 320 tokens, top 4 passages × 700 chars. | `app/config.py`, `.env.example` |
| **Fallback provider removed.** A failing local model is an error; there is no retry on another endpoint. Local-only endpoint policy (`LLM_LOCAL_ONLY=true`) kept. | `app/llm/client.py`, `config.py` |
| Retrieval and answer cache keys include the settings that change the context. | `pipeline._cache_key`, `answer_cache.make_key` |
| UI: Answer / Sources / AI Mode selector, latency panel (all stages + LLM and reranker names), no-evidence explanation, "RAGAS runs offline" note on the eval page. | `frontend/src/...` |
| Eval runner excludes `search` (it has no answer). Pre-existing ruff findings fixed. | `app/eval/run.py` |
| Docs: README, `.env.example`, ADR-0022; docker-compose `OLLAMA_LOAD_TIMEOUT=30m`. | |

**Unchanged:**
- the agentic loop and its API — opt-in "AI Mode";
- the embedding model and dimension, pgvector, BM25 and the RRF implementation;
- data and migrations.

RAGAS and benchmarks are not imported by the serving path; a test checks this in a fresh interpreter.

## 2. Architecture

**Before** (main): the default mode was `hybrid_rerank`.
- Retrieval: BM25 + pgvector → RRF → bge reranker over 30 candidates at 512 tokens.
- Generation: one LLM call over 5 passages × 900 chars, up to 600 tokens.
- The UI also offered the agentic loop. That loop runs an LLM for analysis, a rewrite per sub-question and a sufficiency check before the answer.

**After:**

```
query ─┬─ BM25 (in memory, ~1 ms) ─────────────┐
       └─ embed once (~0.1–0.6 s) → pgvector ───┴─ RRF ─ [bge rerank top 20, FAST_RERANK]
          ─ evidence cut-off ─┬─ no evidence → "not found" (0 LLM calls)
                              ├─ Sources mode → ranked passages (0 LLM calls)
                              └─ Answer mode  → 1 × qwen3.5:9b (stream) → citation validation
AI Mode (opt-in): unchanged agentic loop.     RAGAS / benchmarks: offline scripts only.
```

BM25 and the embedding run one after the other, not in parallel. BM25 takes ~1 ms in memory, so concurrency would save at most that.

## 3. Hardware, runtime and model settings

| Item | Value |
|---|---|
| CPU | Intel Xeon @ 2.80 GHz, **4 vCPU**, 1 thread/core, AVX2, AVX-512 (F, VNNI) |
| RAM / GPU / swap | 15.7 GB / none / none |
| OS / Python | Linux 6.18 (cloud sandbox) / 3.13.17 |
| Database | PostgreSQL 16.15 + pgvector (HNSW, `ef_search` 100); 258 docs, 5,988 chunks |
| Ollama | **0.40.12**, built from the official source (tag commit `f067033`), CPU backend. Listening on 127.0.0.1 only, `NUM_PARALLEL=1`, `MAX_LOADED_MODELS=1`, `FLASH_ATTENTION=1`, `KV_CACHE_TYPE=q8_0`, `CONTEXT_LENGTH=4096`, `KEEP_ALIVE=24h` |
| Answer model | `qwen3.5:9b`: family qwen35, 9.0B params, **Q4_K_M**, digest `56671c2ab938…` |
| Answer settings | `think=false`, temperature 0, `num_ctx` 4096, `num_predict` 320, top 4 passages × 700 chars |
| Embedding | `BAAI/bge-base-en-v1.5` (768-d, unchanged) |
| Reranker | `BAAI/bge-reranker-base`, `RERANK_POOL=20`, `RERANK_MAX_LENGTH=512`, batch 32, CPU, process singleton |
| Inference | **local only**; no hosted API, no keys, no fallback |

This sandbox is **not** the target user machine. Absolute times are specific to this 4-vCPU VM. Ratios between configurations are the meaningful part.

## 4. How to run locally

```bash
./askito up          # or: make up
./askito model       # pulls LLM_DEFAULT_MODEL = qwen3.5:9b
./askito setup       # first run only: migrate + ingest + index
# UI: http://localhost:5173  (Answer / Sources / AI Mode)
```

Switches in `.env`:
- `FAST_RERANK=false`: fastest, RRF order only.
- `FAST_MIN_EVIDENCE_SCORE=0`: no cut-off.
- `DEFAULT_MODE=search`: no LLM.

To reproduce the measurements:

```bash
cd backend
python scripts/retrieval_bench.py --label R_bge --mode search --reps 3 --rep-questions lat12 \
  --set FAST_RERANK=true --reranker BAAI/bge-reranker-base --pool 20 --max-length 512
python scripts/retrieval_bench.py --label P_bge --mode search --reps 1 --questions probes \
  --set FAST_RERANK=true --reranker BAAI/bge-reranker-base --pool 20 --max-length 512
python scripts/fast_smoke.py --label smoke --mode overview      # 6 representative queries, TTFT
python scripts/fast_smoke.py --label smoke_s --mode search
python scripts/eval_bench.py --label FA --mode overview --model qwen3.5:9b --questions lat12
python scripts/eval_bench.py --label FAp --mode overview --model qwen3.5:9b --questions probes
python scripts/analyze_bench.py eval FA FAp ; python scripts/analyze_bench.py retrieval R_bge
```

Raw per-query results are written to `backend/eval/checkpoints/*.jsonl`. They are resumable and gitignored.

## 5. Tests (actual runs)

| Command | Result |
|---|---|
| `cd backend && pytest -q` (2026-11-02 22:09 UTC, at `268d894`; mock LLM, real index/DB) | **132 passed**, 1 warning (starlette deprecation), exit 0 |
| `pytest -q tests/test_fast_rag.py` (after the last fast-path fix) | 13 passed, exit 0 |
| `ruff check app tests scripts` | before: 6 findings, all already on the base commit → fixed → **"All checks passed!"**, exit 0 |
| `npm run typecheck` / `npm run build` | exit 0 / exit 0 |

The new tests in `tests/test_fast_rag.py` map to the acceptance criteria:
- Fast Answer makes **exactly one** LLM call, with role `overview` only, so nothing runs before the answer.
- Sources makes **zero** calls.
- Both channels are fused; RRF equals `rrf_fuse([bm25, vector])` and the fused ids are the union of both channels.
- Citations map to retrieved chunk ids; a fabricated `[9]` is stripped.
- Abstention happens without an LLM call, and the abstention response names no model.
- SSE timings and model names are reported.
- `search` is excluded from eval.
- RAGAS / `datasets` are not imported by the serving modules.

Fallback-removal tests: `tests/test_llm_client.py::test_no_fallback_provider_exists`.

The agentic tests (`tests/test_agent.py`, `tests/test_agent_adaptive.py`) still pass. AI Mode is opt-in: the default mode is `overview` and `agentic` is not a retrieval mode.

## 6. Latency per query and per stage

### 6.1 Representative queries, Fast Answer, through the SSE endpoint code

Measured with `scripts/fast_smoke.py`, final defaults, 2026-11-02 21:34 UTC, commit `7032e2c` + config. All times in ms.

| Query | Embed | BM25 | pgvector | RRF | Rerank | TTFT | Generation | **Total** | LLM calls | Prompt / output tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| `quantum computing` | 50 | 15 | 22 | 0 | 8 318 | 28 648 | 56 182 | **64 591** | 1 | 651 / 89 |
| `popular AI fields` | 245 | 1 | 7 | 0 | 8 051 | 25 861 | 47 706 | **56 013** | 1 | 672 / 69 |
| `What is machine learning?` | 218 | 1 | 6 | 0 | 8 040 | 24 516 | 53 089 | **61 356** | 1 | 586 / 94 |
| `Willow processor 105 physical qubits` (exact entity) | 205 | 1 | 8 | 0 | 8 377 | 18 879 | 42 088 | **50 683** | 1 | 465 / 75 |
| `How do you make a sourdough starter at home?` (unsupported) | 231 | 1 | 6 | 0 | 8 731 | – | – | **8 974** | **0** | – |
| Hopfield and Hinton, 2024 Physics prize (split across chunks) | 87 | 0 | 5 | 0 | 9 667 | 29 404 | 64 617 | **74 381** | 1 | 809 / 112 |

Notes:
- Context assembly measured 0 ms; it is served from the in-memory chunk store.
- "Generation" is the time from the start of the LLM call to the last token, so it includes TTFT.
- Model warm-up was done before measuring; the model was resident.
- In the separate UI check, the same Willow query showed a 1.1 s TTFT, because llama.cpp's prompt-prefix cache already held that prompt. Repeated identical prompts are therefore much faster than first runs.

**Sources mode, same queries:** 7.9–9.0 s each, of which rerank is 7.7–8.9 s; zero LLM calls. With `FAST_RERANK=false` retrieval is ~0.1 s (§7).

### 6.2 Against the pre-change baseline, equivalent conditions

Setup:
- Same 12 eval questions (`lat12`: stratified, including 1 unanswerable), same model `qwen3.5:9b`, same machine and runtime.
- `eval_bench.py` calls `run_search`, one question at a time.
- The baseline is main's default path (`hybrid_rerank`, bge / 30 / 512, `ef_search` 40, 5 × 900-char context, 600 max tokens), run on the base code `e6ca012`.

| | Pre-change default | **Fast Answer** | Change |
|---|---|---|---|
| End-to-end p50 / p95 | 91.9 / 118.9 s | **64.9 / 73.9 s** | **−29% / −38%** |
| Retrieval (incl. rerank) p50 | 18.5 s | 8.7 s | −53% |
| LLM time p50 / p95 | 72.3 / 98.3 s | 55.8 / 64.5 s | −23% |
| Prompt tokens (median) | 1 069 | 733 | −31% |
| Output tokens (median) | 66 | 78 | |
| LLM calls / question | 1 | 1 | |
| doc Recall@5 / MRR@10 / nDCG@10 | 0.636 / 0.674 / 0.633 | **0.636 / 0.674 / 0.634** | equal |
| chunk Recall@5 | 0.364 | 0.364 | equal |
| Citation coverage (sentences with ≥1 citation) | 0.917 | 0.939 | |
| Cited sources from a gold document | 0.333 | 0.318 | |
| Raw citation validity (citations pointing to a real source) | 1.0 | 1.0 | |
| "Not found" answers / unanswerable refused | 2 / 12, 1 of 1 | 1 / 12, 1 of 1 | |
| Peak RSS of the Ollama processes | 8.7 GB | 11.3 GB | (mmap'd weights; not an allocation difference attributable to this change) |

Runs: 2026-11-02 19:57 UTC (baseline) and 21:40 UTC (Fast Answer, commit `96b867a`).

The baseline's second "not found" was on an answerable question. Fast Answer answered that one.

An intermediate configuration with the reranker at 256 tokens per pair was faster: p50 59.5 s, retrieval 4.4 s. On these 12 questions it lost **9 points of doc Recall@5 and 16 points of MRR@10**, so it was rejected (§7).

**Target check:** the brief's "a few seconds overall" target is **not met on this CPU-only VM**. The LLM part (TTFT 19–29 s, generation ~3.2 tok/s) dominates. Retrieval and fusion without the reranker are sub-second (~0.1 s). The reranker adds ~8 s here and can be switched off.

## 7. Retrieval quality and reranker comparison

Setup:
- All 46 eval questions; quality from repetition 1 (it is deterministic: identical across 3 repetitions).
- Latency from 3 repetitions (46 + 2 × 12 queries).
- Identical candidates for every row (BM25 50 + pgvector 50, RRF), mode `search`, current code.
- Runs from 2026-11-02 20:48–21:11 UTC.

| Reranker (pool 20) | Retrieval p50 / p95 | Rerank p50 | doc R@5 | chunk R@5 | doc MRR@10 | doc nDCG@10 | Cold load |
|---|---|---|---|---|---|---|---|
| **A. None** (RRF only) | 87 / 248 ms | – | 0.716 | 0.318 | 0.654 | 0.667 | – |
| A'. None, 20 per channel | 94 / 152 ms | – | 0.716 | 0.330 | 0.649 | 0.659 | – |
| **C. MiniLM-L-6** (true cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`), 256 tok | 824 / 958 ms | 707 ms | 0.739 | 0.318 | 0.658 | 0.681 | 0.6 s |
| C. MiniLM-L-6, 512 tok | 1.38 / 1.67 s | 1.26 s | 0.750 | 0.330 | 0.627 | 0.666 | 0.5 s |
| **B. bge-reranker-base, 512 tok (default)** | **7.75 / 8.61 s** | 7.64 s | **0.784** | **0.432** | **0.727** | **0.729** | 2.0 s |
| B. bge-reranker-base, 256 tok | 4.09 / 4.76 s | 3.97 s | 0.773 | 0.432 | 0.701 | 0.711 | 2.2 s |
| Reference: previous default bge, pool 30, 512 tok, `ef_search` 40 | 13.9 / 16.7 s | 13.8 s | 0.807 | 0.455 | 0.742 | 0.749 | 16 s* |

\* measured earlier in the session; that cold load included a cold disk cache.

The embedding model loads in ~8 s at startup, outside the query path (warm-up). The reranker loads once per process.

**Decision: bge-reranker-base, 20 candidates, 512 tokens.**
- MiniLM is ~6–9× faster, but on this corpus its gain over no reranker is small and mixed: MRR drops at 512 tokens, and chunk R@5 does not improve.
- bge/20/512 stays within 2.3 / 1.5 / 2.0 points (R@5 / MRR / nDCG) of the previous 30-candidate setting, at ~55% of its rerank time.
- On the 12-question end-to-end set it was equal to the baseline (§6.2).
- `FAST_RERANK=false` remains available for slow machines; it loses ~7 points of R@5 and MRR on this set.

**Evidence cut-off calibration** (bge, top reranker score per query):

| Set | n | Top-score range | Below 0.3 (abstain) |
|---|---|---|---|
| Answerable eval questions | 44 | 0.70 – 1.00 | **0** |
| Off-topic probes (sourdough, Bitcoin, Mona Lisa, Lisbon, Burj Khalifa, olive oil) | 6 | 0.001 – 0.05 | 6 |
| Near-domain probes (2023 Literature, 2024 Peace, Goldin, Han Kang) | 4 | 0.05 – 0.99 | 1 |
| Unanswerable eval questions (Nobel banquet menu, Hinton's exercise hours) | 2 | 0.86 – 0.99 | 0 |

The cut-off is in the reranker's own score units. MiniLM outputs logits, so the threshold must be re-calibrated if the reranker changes.

**End-to-end abstention, Fast Answer, 10 refusal probes** (`eval/refusal_probes.json`, kept separate from the main set): **10 / 10 answered "I could not find this in the sources."**
- 7 stopped at the cut-off with **0 LLM calls** (~8–9 s).
- 3 near-domain probes reached the LLM, and the model itself declined, as the prompt instructs.

Main-set unanswerable question in §6.2: refused (1 of 1).

## 8. Answer and citation quality: manual check

Method:
- For the 6 representative queries in §6.1 I read each answer.
- I checked every claim against the **full text** of the chunk it cites (database query), not the snippet.
- This check was done by me while implementing the change; it is not an independent review.

| Query | Verdict |
|---|---|
| quantum computing | All claims supported: superposition / entanglement [2]; "exponentially faster", "breaking encryption" [2]; simulating quantum phenomena [1]; measurement-based and adiabatic / quantum-annealing paradigms [3]. |
| popular AI fields | Supported. Open-source categories (LLMs, MT, chatbots) [2]; prominent applications (web search, autonomous vehicles, content generation) [1]; hiring use [3]. It lists only areas present in the retrieved passages. As a "fields" answer it is partial: the corpus has no single list of AI fields. |
| What is machine learning? | Supported: definition [1]; learning paradigms [2][4]; 1959 / Arthur Samuel / cognitive-process origins [3]. |
| Willow processor | Supported: 105 transmon qubits [1]; "In 2025, Google announced…" [2]. |
| sourdough (unsupported) | Correct abstention, no LLM call. |
| Hopfield and Hinton | Supported: Nobel citation [1][4]; AI winter / revitalised interest [1]; "Godfather of AI" [2]; Hopfield's statistical-mechanics associative memory [3]. **Incomplete:** Hinton's specific contribution (Boltzmann machines) is not stated, although a retrieved source covers it. |

No fabricated citation IDs appeared: raw citation validity is 1.0 in both §6.2 runs. Citations show title, chunk id and Wikipedia URL in the UI.

**UI check** (Playwright, real backend + qwen3.5:9b):
- Answer mode: first results visible after 1.0 s; answer done after 39 s.
- Sources mode: no answer card; done after 11 s.
- Abstention: the no-evidence explanation is shown.
- 360 px mobile: no horizontal overflow.
- The latency panel lists all stages and the model / reranker names.

## 9. Not run, limitations, follow-ups

**NOT RUN:**
- **RAGAS:** an offline evaluation, and expensive with a local judge on this CPU (~20 min per sample with qwen3:14b in an earlier estimate). No RAGAS numbers are claimed. To run it:
  ```bash
  ollama pull qwen3:14b
  python scripts/ragas_pass.py --labels FA --judge qwen3:14b --questions rag6
  ```
- **Real-model AI Mode** with the final code: agentic behaviour was verified by tests with a scripted LLM only. An earlier measurement on the base code, qwen3:8b, 6 questions, gave p50 204 s.
- **Measurements on the target user machine:** commands are in §4.
- **The previous default model `qwen3:30b`:** it does not fit in 15 GB RAM here, so it was not compared.
- **Ollama server tuning:** flash attention off / f16 KV, thread counts.

**Limitations:**
- **Small samples:** 12 questions end to end, 46 for retrieval, 6 representative queries, 10 probes, 2 unanswerable questions in the main set.
- **Single runs:** LLM runs were done once, so there is no variance estimate. Retrieval latency comes from 3 repetitions.
- **Hardware:** a 4-vCPU VM. The LLM dominates latency and is ~3 tok/s here.
- **Evidence cut-off:** it only catches clearly off-topic queries. Near-domain unsupported questions rely on the model's not-found instruction (10 / 10 here, but not guaranteed).
- **Broad questions** ("popular AI fields") get partial, source-bound lists.
- **Prefix cache:** llama.cpp's prompt-prefix cache makes repeated identical prompts much faster. Benchmark first runs are the realistic numbers.

**Follow-ups:**
- Run §4 on the target machine; consider `FAST_RERANK=false` or MiniLM if the CPU is slow.
- A GPU or a faster CPU would cut TTFT and generation proportionally.
- Run RAGAS offline on `rag6` with a fixed local judge.
- Consider an "AI answer on/off" toggle instead of a separate Sources tab.

## 10. Branch and commits

Branch `fast-rag` (from `claude/pensive-ritchie-ruci6o` = M10 + the local-LLM tooling, with `main` merged in). Commits in this task:

| SHA | Commit |
|---|---|
| `532900a` | Remove the LLM fallback provider |
| `84173ae` | Fast RAG path: search mode, optional reranker, evidence cut-off, stage timings |
| `5235763` | UI: Answer / Sources / AI Mode selector, stage latency, model and reranker names |
| `25aa5af` | Benchmarks: fast modes and setting overrides in retrieval_bench; fast_smoke |
| `e8b1709` | Set fast-path defaults from measurements |
| `d222f3b` | Docs: Fast Hybrid RAG default path, modes, reranker choice and evidence cut-off |
| `7032e2c` | fast_smoke: store the query type under its own field |
| `96b867a` | Reranker default: 512 tokens per pair |
| `3225aa0` | Do not report an LLM model when the answer was produced without calling it |
| `268d894` | Fix pre-existing lint findings |

The report commit and the merge into `main` are listed in the handoff message.

Merging `fast-rag` into `main` also brings the earlier, not-yet-merged work on that branch line:
- M10 (`7927cc0`, `cbd83f0`, `9ee905b`);
- the local-LLM optimization tooling (`d97b1ec` … `fd02310`): local-only policy, instrumentation, adaptive agent switches (off by default), benchmark scripts.
