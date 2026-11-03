# Ask ITO — Redis / Q8 / KV cache / GPU: verification report

This answers `docs/ASK_ITO_REDIS_Q8_KV_GPU_OPTIMIZATION_CLAUDE.md` and
`docs/ASK_ITO_REDIS_Q8_GPU_VERIFICATION_REMEDIATION_CLAUDE.md`.

Everything below was run in a cloud sandbox: CPU only, no GPU, not the target
PC. Anything not executed is marked **NOT RUN**, and anything not observed is
marked **NOT VERIFIED**.

## 1. Branch, commits and code audit

### Branch and commits

**Branch:** `claude/pensive-ritchie-ruci6o`. Its fixes up to 6bcbdde were merged into `main` by PR #1 (8577b69). This report and a small UI change follow on `main` at the owner's request.

**Implementation commits already on `main`:**

| Commit | Change |
|---|---|
| ca2c1b7 | Redis L2 cache (embedding / retrieval / answer front / agent tool) |
| 57a4396 | Q8_0 default tag, F16 KV in compose, Redis service |
| 5add4a5 | Agent duplicate-search / sufficiency guards |
| 9f59ea7 | Smoke, agent audit and hardware scripts |
| 5866209 | Prompt v4 |

**Fixes from this verification, on the branch:**

| Commit | Change |
|---|---|
| 0bd2dcb | Compose healthcheck; smoke-script safety; real-Redis tests |
| e5b2515 | GPU verdict fix; prefix probe fix |
| ba2f2d2 | Remediation report |
| 6bcbdde | Measured Q8 / Q4 memory and the 16 GB limit (README, `.env.example`) |
| *this commit* | This report and the measurement files |

### Code audit

| Area | Finding | Action |
|---|---|---|
| Redis adapter (`app/redis_cache.py`) | Local-only URL check, the same rule as the LLM endpoint. JSON / bytes only (no pickle); per-kind TTL; size limit; hashed keys under `askito:v1:<kind>:`. Circuit breaker: one warning, then a pause of `REDIS_RETRY_S`; `REDIS_STRICT` turns failures into errors | OK — no change |
| Redis down at startup | The client is created lazily, so the first operation fails, the circuit opens and callers fall back to L1 / PostgreSQL | OK. New real-socket test (closed port) |
| Compose | Redis had no healthcheck | **Fixed (0bd2dcb):** `redis-cli ping` healthcheck; the backend still depends only on `service_started`, so a Redis outage never blocks it; `stop_grace_period: 5s`; no host port, no persistence |
| Query-embedding cache key | Embedding model, dimension, query instruction, whitespace-normalised text, version | OK |
| Retrieval cache key | Normalised query, mode, domain, final_k + `retrieval_fingerprint`: index version (BM25 metadata), embedding model / dim, channel k, BM25 pool, ef_search, RRF k, max chunks per doc, reranker on / off / model / pool / max length | OK. Stores chunk ids, scores and trace only; text is re-hydrated from the current store |
| Answer cache key | Model tag **plus the served weight quantization and digest** (from Ollama), think, num_ctx, prompt version, token and context limits, final_k, evidence cut-off state / threshold / model, retrieval fingerprint. A hit is also refused unless the stored source chunk ids equal the current ones | OK. Citations stay valid; a Q4 answer can never be served for Q8 |
| Agent tool cache | Key: model tag (with quantization / digest), num_ctx, temperature 0, max tokens and the **full prompt text** (template + question + context). Only parsed results are stored; off when think is on | OK. It holds tool outputs, not conversation state, so nothing per-user or per-session is stored and nothing can leak between sessions. Two users asking the identical question share the deterministic analysis, as with the answer cache |
| Model KV tensors | Redis stores only JSON / embeddings; the KV / prefix cache lives in the Ollama runner | OK |
| Agent guards (`app/agent/loop.py`) | A duplicate sub-question, an already-searched query, and "follow-up search added no new passage" stop only work whose result is already known. The sufficiency prompt is question + context, deterministic at temperature 0. Rounds, call budget, timeout and the final answer call are unchanged; distinct sub-questions are all searched | OK. Covered by agent tests: multi-hop, repeated follow-up, duplicate sub-question, budget, deadline |
| Smoke script | Did not stop on a wrong quantization, a load failure / OOM, a timeout or Ctrl-C | **Fixed (0bd2dcb):** required `--expect-quant`; stops with a recorded reason; per-question errors recorded; the answering model must equal the tag; generation settings and model parameters recorded |
| Prefix probe | Judged reuse by `prompt_eval_count`, which Ollama 0.40 does not lower on a cache hit | **Fixed (e5b2515):** judged on prefill time, with a prompt the preceding run cannot have cached |
| Hardware diagnostic | Inferred GPU offload from `/api/ps size_vram` | **Fixed (e5b2515):** see §6. On this CPU-only machine Ollama reports AMX CPU buffers as VRAM |
| Secrets / internals in output | Health shows only the Redis host name and an exception class name; logs show the host, never the URL credentials; API errors use error ids (earlier work) | OK |

## 2. Issues found and fixes

1. **The GPU diagnostic could report GPU offload on a machine with no GPU.**
   - What happened: Ollama 0.40.12 counts the AMX repacked weight buffer as VRAM. `ollama ps` showed `18%/82% CPU/GPU`, and `size_vram` was 4,893 MiB for Q4 and 8,987 MiB for Q8.
   - Fix (e5b2515): a GPU verdict now needs a GPU device and a GPU runner library from the Ollama log.
2. **The prefix-cache probe would have reported "no reuse" even when reuse happened.** Fixed in e5b2515.
3. **The smoke script could crash without a clear record, or run against an unexpected quantization.** Fixed in 0bd2dcb.
4. **Redis had no compose healthcheck.** Fixed in 0bd2dcb.
5. **Prompt injection (found by the remediation work):** with prompt v3, qwen3.5:9b switched to French when a passage told it to, and commented on an injected instruction. Prompt v4 (5866209) fixes both, see §8.
6. **Q8_0 + the app ran out of memory on 15.7 GB** (agent run, §7). The memory numbers are now documented (6bcbdde). No quantization change.

## 3. Tests

Run on the final branch code: 6bcbdde plus this report, which is docs only. Nothing else was running on the CPU.

### Commands

| Command | Result |
|---|---|
| `cd backend && .venv/bin/python -m pytest -q` | **244 passed**, 0 failed, 0 skipped, 4 min 35 s |
| `cd backend && .venv/bin/ruff check app tests scripts` | All checks passed |
| `cd frontend && npm run typecheck` | exit 0 |
| `cd frontend && npm run build` | exit 0 (bundle-size warning only) |

There is no separate Python type checker configured (mypy / pyright: none).

### Environment for the run

- PostgreSQL 16 + pgvector with the built index: the integration tests ran, none were skipped.
- `redis-server` 7.0.15, started by the tests on free loopback ports.
- The LLM side uses the mock or scripted clients in tests; the real local model was used only in the measurements below.

### Relevant test files

| File | Tests | What |
|---|---|---|
| `test_redis_cache.py` | 30 | Redis policy, fallback, TTL / expiry, key versioning, L2 round trips, end-to-end Search miss / hit / invalidation, Redis stopped mid-run. Real redis-server, except the explicitly mocked fallback cases |
| `test_tool_cache.py` | 7 | Agent tool cache: hit only on identical model + prompt + limits; parsed results only; off with think |
| `test_agent.py`, `test_agent_adaptive.py` | 3 + 11 | Simple, multi-hop, duplicate query, duplicate sub-question, repeated follow-up, adaptive switches, call budget, deadline. Real retrieval + reranker, scripted LLM |
| `test_quant_smoke.py` | 5 | Five exact questions; identical messages / limits for both models; refuses a missing runtime, a missing model or a different quantization |
| `test_model_config.py` | 7 | Single model-tag source (Q8 default, Q4 rollback, no lower quantization); compose KV f16; Redis healthcheck and ordering; runtime quantization / digest |
| `test_hw_diag.py` | 7 | CPU-only verdict even when `size_vram` > 0; GPU verdicts need a device and a GPU runner library |
| `test_stream.py`, `test_hardening.py`, `test_health.py` | 11 + 9 + 4 | Answer-cache citations and source check, cache kind, prompt guards, safe errors, Redis health |

## 4. Redis evidence

All rows below ran against a **real `redis-server` 7.0.15** started by the tests on a loopback port, unless marked otherwise.

| Behaviour | Test | Result |
|---|---|---|
| First request misses, identical repeat hits (end-to-end Search: BM25 + embedding + pgvector) | `test_search_miss_hit_and_invalidation_end_to_end` | first: not cached, embedding computed. After dropping L1: `cache_tier=l2`, identical results |
| Config change invalidates | same test | RRF k + 1 is recomputed (the embedding is still reused from L2) |
| Index rebuild invalidates | `test_retrieval_l2_round_trip_and_invalidation` | new index version: miss |
| Embedding model / instruction change | `test_query_embedding_l2` | miss |
| Answer cache: citations kept, sources must match | `test_answer_front_cache_keeps_citations_and_checks_sources` | Redis hit keeps citations; reordered sources: miss |
| Answer key dimensions | `test_answer_key_changes_with_model_quantization_and_prompt` (unit) | Q8 vs Q4 identity vs prompt version: all distinct keys |
| TTL set | `test_json_round_trip_with_ttl` | 0 < TTL ≤ configured |
| TTL expiry | `test_ttl_expiry_on_real_redis` | 1 s TTL: gone after 2.2 s |
| Oversized values / scoped clear / concurrency | three tests | skipped / only this app's keys of one kind / 8 threads, no errors |
| Redis unavailable at startup | `test_redis_unavailable_at_startup_falls_back` | real redis-py client against a closed port (connection refused): one warning, L1 serves, health "unavailable" |
| Redis stops mid-run | `test_search_keeps_working_when_redis_stops` | server terminated between two searches: the second search still returns results |
| Local-only URL policy | 12 parametrised cases (unit) | loopback ok; LAN, public, hosted, `0.0.0.0` and non-redis schemes refused; allowed container host must resolve to a private address |
| Failure circuit / strict mode | mock client | one warning, no retry storm; strict mode raises |

**NOT RUN:** the Docker Compose stack itself (no Docker daemon in the sandbox), so the healthcheck and service ordering are validated from the YAML only (`tests/test_model_config.py`).

## 5. Q4_K_M vs Q8_0 — the five questions

Raw data is in `docs/optimization/checkpoints/redis_q8/`: `frozen.json` (the
exact prompts and sources) and `run_qwen3.5_9b_kv-f16.json` /
`run_qwen3.5_9b-q8_0_kv-f16.json`.

### Setup

- **Hardware:** Intel Xeon @ 2.1 GHz, 4 vCPU, AVX-512 + AMX; 15.7 GB RAM; no GPU.
- **Runtime:** Ollama 0.40.12. Read from the running server's environment: `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_FLASH_ATTENTION=1`, context 4096, `NUM_PARALLEL=1`. The runner log shows `K (f16) / V (f16)`, 128 MiB.
- **Identical input for both models:** retrieval ran once (hybrid + bge reranker + evidence cut-off, 3–4 passages per question) and the exact chat messages were frozen (`frozen.json`).
- **Identical generation settings:** temperature 0, `num_predict` 320, num_ctx 4096, think off.
- **Identical model defaults:** both tags share the same Modelfile parameters (top_p 0.95, top_k 20, presence_penalty 1.5).
- **Caching:** the answer cache is not involved. Each model was loaded cold, after a server restart for Q8 and the first load for Q4.

### Models

| | Q4_K_M baseline | Q8_0 |
|---|---|---|
| Tag | `qwen3.5:9b` (= `qwen3.5:9b-q4_K_M`) | `qwen3.5:9b-q8_0` |
| Quantization reported by Ollama | Q4_K_M | Q8_0 |
| Local digest | 56671c2ab938 | a8ee95bc3592 |
| Model file (incl. projector) | 6.6 GB | 10.45 GB |
| Cold load (wall) | 21.0 s | 32.8 s |
| Ollama RSS after load / peak during the 5 questions | 6.7 GB / **7.7 GB** | 11.2 GB / **12.3 GB** |
| `size_vram` reported | 4,893 MiB (= AMX CPU buffer) | 8,987 MiB (= AMX CPU buffer) |

### Per question

TTFT is wall time to the first token. Prefill and decode come from Ollama's own durations.

| Question | Model | TTFT | Prefill | Decode | Total | Peak RSS |
|---|---|---|---|---|---|---|
| quantum computing | Q4 | 22.2 s | 801 tok / 22.2 s | 98 tok, 3.69 tok/s | 48.8 s | 6.9 GB |
| | Q8 | 17.0 s | 801 tok / 17.0 s | 104 tok, 5.18 tok/s | 37.1 s | 11.5 GB |
| What is machine learning? | Q4 | 18.6 s | 736 / 18.3 s | 89, 3.46 | 44.3 s | 7.1 GB |
| | Q8 | 15.5 s | 736 / 14.8 s | 91, 5.09 | 33.3 s | 11.6 GB |
| popular AI fields | Q4 | 14.2 s | 822 / 13.8 s | 68, 3.83 | 32.0 s | 7.3 GB |
| | Q8 | 13.5 s | 822 / 12.5 s | 70, 4.65 | 28.5 s | 11.9 GB |
| Willow processor 105 physical qubits | Q4 | 9.2 s | 602 / 8.4 s | 80, 3.72 | 30.7 s | 7.5 GB |
| | Q8 | 9.5 s | 602 / 8.3 s | 78, 5.12 | 24.7 s | 12.1 GB |
| Hopfield and Hinton, 2024 Physics prize | Q4 | 15.7 s | 895 / 15.0 s | 108, 3.62 | 45.5 s | 7.7 GB |
| | Q8 | 15.5 s | 895 / 14.8 s | 96, 4.67 | 36.1 s | 12.3 GB |
| **Mean** | Q4 | 16.0 s | 53 tok/s | 3.66 tok/s | 40.3 s | max 7.7 GB |
| | Q8 | 14.2 s | 59 tok/s | 4.94 tok/s | 32.0 s | max 12.3 GB |

### Observations

- On this AMX-capable Xeon, Q8_0 decoded about 35% faster than Q4_K_M, and its prefill was equal or faster.
  - A plausible cause is that the runner repacks Q8_0 weights for the AMX int8 path. This was not profiled.
  - This is specific to this CPU and does **not** predict the target PC: on CPUs without AMX, or on GPUs, Q4_K_M is usually the faster one. Measure it there.
- Q8_0 needs about 4.6 GB more RAM. On 15.7 GB the machine had about 2.7 GB available while Q8 was loaded.
  - Q8 fits here, so no lower quantization was used.
  - On a machine where it does not fit, Ollama returns a load error. The smoke script records it and stops; it never substitutes another quantization.
- Peak RSS rises across the five questions for both models, consistent with the runner keeping a host-memory prompt cache (see §6).

### Citations and grounding

Manual review by the implementer, not independent. Each answer sentence was checked against the passages it cites.

| | Sentences | Supported | Partial | Unsupported | Citation-ID validity |
|---|---|---|---|---|---|
| Q4_K_M | 15 | 12 | 3 | 0 | 1.0 (5/5 answers) |
| Q8_0 | 15 | 12 | 3 | 0 | 1.0 (5/5 answers) |

**Partial claims.** Q4:
- ML S1 says "unseen tasks" where the passage says "unseen data".
- ML S3 says coining the term "marked the beginning" of cognitive-science efforts.
- Hopfield S3 credits the prize with "revitalising AI", which the passage says of Hopfield's work, not of the prize.

Q8:
- ML S1, the same "unseen tasks" wording.
- Willow S2 says Willow itself was "announced in 2025"; the passage dates the quantum-advantage announcement.
- Hopfield S3, the same over-reach as Q4.

**Coverage and abstention.**
- Both models covered the same facts.
- Neither abstained; all five questions have supporting passages.
- For "popular AI fields", neither presented the list as complete.

**Five samples do not show a quality difference in either direction.**

## 6. Hardware, runtime and GPU offload

- **Host:** Intel Xeon @ 2.1 GHz, 4 vCPU, flags avx2 / avx512f / avx512_vnni / amx_tile; 15.7 GB RAM; no `/dev/nvidia*`, `/dev/kfd` or `/dev/dri`; no `nvidia-smi` / `rocm-smi`.
- **Ollama 0.40.12:** built from source; the runner is llama-server.
- **Server log:** `msg="inference compute" id=cpu library=cpu … total="15.7 GiB"`.
- **`scripts/hw_diag.py --log …` verdict: `cpu_only`** (`redis_q8/hw_diag.json`). The report also states that `size_vram` (4,893 MiB for Q4) is the AMX buffer, not GPU memory. Runner log lines: `redis_q8/ollama_runner_log_excerpt.txt`.
- **GPU offload: NOT VERIFIED.** There is no GPU here. Do not read `ollama ps` "CPU/GPU" percentages on this kind of CPU as GPU use.
- **KV cache:**
  - Compose default is f16; runtime verified as f16 (128 MiB for 4096 cells, because only 8 of 32 layers have attention KV).
  - q8_0 would be 68 MiB, so the 60 MiB saving is not a reason to compress. q8_0 stays an opt-in (`OLLAMA_KV_CACHE_TYPE=q8_0`).
- **Prefix / prompt cache** (`prefix_*.json`; the remediation quality set showed the same):

  | Request | Q4 prefill | Q8 prefill |
  |---|---|---|
  | New prompt (801 tok) | 13.9 s | 13.6 s |
  | Identical repeat | 0.62 s | 0.44 s |
  | Same passages, new question | 12.9 s | 12.1 s |

  - Exact-prompt reuse is **verified**, and it also works across other requests: 13 other prompts in between in the quality set.
  - Partial prefix reuse is **not observed**. qwen3.5 is a hybrid recurrent model; the runner logs `KV cache shifting is not supported`, and it only resumes from exact cached states.
  - Moving static text to the front of the prompt therefore gives no speed-up with this model. The prompt layout was left unchanged.
  - `prompt_eval_count` still shows the full prompt length on a cache hit.

### Target PC / GPU checks — NOT RUN

To verify on the target machine:

1. Compare the CPU and GPU readings:
   - In Docker: `docker compose exec ollama ollama ps`. The PROCESSOR column should read `100% GPU` for full offload.
   - On the host: `nvidia-smi`. Check memory used by the ollama process and GPU utilisation while an answer is generated.
2. Check the Ollama log: `docker compose logs ollama | grep -E "inference compute|offloaded|model buffer"`.
   - It must show `library=cuda` (or rocm / metal) and `offloaded N/N layers to GPU`.
3. Run `python scripts/hw_diag.py --log <ollama log>` in the backend container.
   - GPU passthrough needs the NVIDIA Container Toolkit and a `deploy.resources.reservations.devices` entry for the `ollama` service. Docker Desktop on Windows needs a WSL 2 GPU driver. These are not enabled by default in compose.
   - Apple Silicon: run Ollama natively, since Docker on macOS has no Metal; then point `LLM_BASE_URL` at it.
4. Run the five-question smoke on the target PC:

   ```bash
   cd backend
   python scripts/quant_smoke.py freeze
   python scripts/quant_smoke.py run --model qwen3.5:9b --expect-quant Q4_K_M --tag target
   python scripts/quant_smoke.py run --model qwen3.5:9b-q8_0 --expect-quant Q8_0 --tag target
   ```

## 7. Agent call counts

`scripts/agent_audit.py`, real retrieval, reranker and local model. Four queries:
- **simple factual:** "What is a transmon qubit?"
- **multi-hop:** Hopfield and Hinton's contributions
- **comparison:** superconducting vs trapped-ion qubits
- **near-domain unsupported:** 2024 Nobel Peace Prize

**Baseline and model**
- The baseline is commit **4433ebd** (before the guards and the tool cache), run from a separate checkout against the same database, index and model (`--app-root`).
- Both runs use **qwen3.5:9b Q4_K_M**. The Q8_0 attempt was killed by the out-of-memory killer (see below), so the model was held at Q4 on both sides. Call counts depend on the code and the model's decisions, not on the weight quantization of an identical tag pair, but Q8 call counts themselves are **NOT MEASURED**.
- Limits on both sides: max rounds 3, max LLM calls 10, timeout 180 s.

| Query | Run | analyse | rewrite | sufficiency | answer | LLM calls | searches | rounds | guards fired | answer |
|---|---|---|---|---|---|---|---|---|---|---|
| simple factual | baseline | 1 | 1 | 1 | 1 | 4 | 1 | 1 | — | answered [1][2][3] |
| | current, pass 1 | 1 | 1 | 1 | 1 | 4 | 1 | 1 | — | identical text |
| | current, pass 2 | 0 | 0 | 0 | 1 | **1** | 1 | 1 | tool cache ×3 | identical text |
| multi-hop | baseline | 1 | 3 | 1 | 1 | 6 | 3 | 1 | — | answered [1][4][5] |
| | current, pass 1 | 1 | 3 | 1 | 1 | 6 | 3 | 1 | — | same content, [1][5] |
| | current, pass 2 | 0 | 0 | 0 | 1 | **1** | 3 | 1 | tool cache ×5 | identical to pass 1 |
| comparison | baseline | 1 | 2 | 2 | 1 | 6 | 4 | 3 | — | "context has no direct comparison…" [2][3] |
| | current, pass 1 | 1 | 2 | 2 | 1 | 6 | **3** | **2** | `sufficiency_stopped` (follow-up query already searched) | identical text |
| | current, pass 2 | 0 | 0 | 0 | 1 | **1** | 3 | 2 | same + tool cache ×5 | identical text |
| near-domain unsupported | baseline | 1 | 1 | 2 | 1 | 5 | 3 | 3 | — | "I could not find this in the sources." |
| | current, pass 1 | 1 | 1 | 2 | 1 | 5 | 3 | 3 | — | identical |
| | current, pass 2 | 0 | 0 | 0 | 1 | **1** | 3 | 3 | tool cache ×4 | identical |
| **Total** | baseline | | | | | **21** | **11** | | | |
| | current, pass 1 | | | | | **21** | **10** | | | |
| | current, pass 2 | | | | | **4** | 10 | | | |

**Reading it honestly**
- **Guards, first run of a query:** no reduction in LLM calls (21 → 21). One search and one round were saved on the comparison query, where the follow-up the model asked for had already been searched. Multi-hop retrieval was not suppressed: all 3 sub-question searches ran. No answer changed in substance, and the unsupported query was still declined.
- **Tool cache, repeated identical question:** analyse / rewrite / sufficiency come from the cache, so 21 → 4 calls (only the final answer is generated each time).
  - This helps only for repeated questions. Nothing per-user is cached.
  - It is on by default (`AGENT_TOOL_CACHE=true`), in-process L1, Redis L2 when enabled.
- **Latency:** not compared.
  - The baseline started with a cold model load: analyse took 60 s on its first query.
  - The current run then reused the runner's prompt cache from the baseline for identical analyse prompts (60 s → 9 s).
  - Pass-2 times (23–47 s) combine the tool cache with the runner's prompt cache.
  - So the time columns in `agent_audit_*.json` are kept as raw data only; they are not evidence of a speed-up from the guards.
- **Prompt version:** the baseline used prompt v3 for the final answer and the current code uses v4. This does not change the agent's analyse / rewrite / sufficiency prompts.

**Q8_0 agent run: out of memory**
- Run with `qwen3.5:9b-q8_0`, the kernel's cgroup out-of-memory killer stopped both processes after 1 min 45 s:
  - the Ollama runner, at 13.0 GB anonymous RSS;
  - the agent process, holding embedding + reranker, at 0.84 GB.
- Log: `oom_q8_agent_run.txt`.
- The Q8 model itself loaded and answered the five smoke questions, peaking at 12.3 GB; the runner's in-memory prompt cache keeps growing with use.
- On this 15.7 GB machine, Q8_0 plus the rest of the app therefore does not fit reliably. README and `.env.example` now state the measured numbers and the Q4_K_M rollback (6bcbdde). The default stays Q8_0 as requested, so on a 16 GB target PC the user must check this.

## 8. Quality, citations and limitations

- **Prompt-injection smoke, prompt v4:** both cases resisted with both models (`injection_smoke_v4_q4.json`, `injection_smoke_v4_q8.json`): English answer, correct citations, no instruction following, no meta-commentary. With prompt v3, the same Q4 model answered in French and commented on the injected instruction. One run per case, so this is indicative only.
- **Sample size:** five questions. No statistical claim; no RAGAS (not requested).
- **Host change:** the sandbox changed host during the work (2.8 GHz → 2.1 GHz + AMX). All Q4/Q8, prefix and agent numbers above come from the 2.1 GHz host only.

## 9. NOT RUN / NOT VERIFIED

- Target PC: everything (latency, memory, Q8 fit, GPU offload). **NOT RUN.**
- Agent call counts with Q8_0: **NOT MEASURED** (out of memory on this machine; measured with Q4_K_M on both sides).
- Agent latency reduction: **NOT MEASURED** (runner prompt cache shared between the baseline and current runs).
- GPU offload: **NOT VERIFIED.** No GPU present.
- Docker Compose stack (Redis healthcheck, startup ordering, GPU passthrough): **NOT RUN.** No Docker daemon.
- Independent human review of answers.
- RAGAS, overnight or large benchmarks: not run, by instruction.

## 10. Follow-up

- Measure Q4 vs Q8 on the target PC with the commands above. Keep Q4_K_M as the rollback if Q8 does not fit or is slower there.
- If the target has a GPU, verify offload with `ollama ps` + `nvidia-smi` + the log, not `size_vram` alone.
- Mobile Search view label "Sources used in the overview" in Search mode (wording only).
