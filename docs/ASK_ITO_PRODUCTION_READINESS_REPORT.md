# Ask ITO — production readiness (Q8_0) report

Implements `docs/ASK_ITO_CONSOLIDATED_PRODUCTION_READINESS_Q8_PLAN.md` on `main`.
Fixed decisions kept: **Q8_0 weights stay the default** (no Q4_K_M switch, no
automatic fallback), **F16 KV cache** (`OLLAMA_KV_CACHE_TYPE=f16`), local-only
inference, no Qdrant, Agentic RAG kept.

Status words: **VERIFIED** (run here, evidence below), **NOT VERIFIED** (not run
on that environment), **PARTIAL** (part verified, gap stated).

Environment of every measurement: CPU-only sandbox, 4 vCPU Xeon (AMX), 15.7 GB
RAM, no GPU, Ollama 0.40.12 (llama-server runner), PostgreSQL + pgvector,
`askito/qwen3.5:9b-q8_0-text` (Q8_0), F16 KV cache. No result below is from a
GPU, from Docker Desktop on macOS or from the target PC.

## 1. Commits (all on `main`)

| Commit | Content | Files |
|---|---|---|
| `7d42565` | AI Mode P0: cold model loads, classified model errors, stream lifecycle | 20 |
| `b04a445` | Search / AI Mode UI, adaptive AI Mode routing, homepage and suggestions | 12 |
| `0c7af73` | Q8_0 in a 14 GB Docker VM: text-only Q8_0 tag, prompt cache off, `./askito doctor` | 17 |
| `2dc2c70` | UI scales with the window | 4 |
| `95252a1` | AI Mode adds Wikipedia articles when the local evidence is weak | 20 |
| `1080ea7` | Evaluation page: test set for this corpus, run status and errors | 12 |
| `2936de0` | Request ids on every response, visible askito log lines | 6 |
| `abef4f5` | Answer prompts: partial evidence gives a partial answer (v5) | 4 |
| the commit adding this file | Ingestion cache-invalidation test (real PG + Redis), Ollama recovery note, this report | 3 |

## 2. Audit: findings, root causes, fixes

### P0

**P0-1 AI Mode never answered after a restart.**
`app/llm/client.py` (generation call), `app/api/stream.py`.
- Reproduction: cold Ollama, AI Mode question with the Q8_0 model → "search_failed"
  and "No answer was produced".
- Root cause: a cold Q8_0 load (about 10 GB from disk) took **689 s**, longer than
  the 240 s generation timeout. The client gave up, Ollama **aborted the load
  because the requesting connection closed**, and every later request started
  the load again. `/api/ps` lists a model while it is still loading, so a
  "loaded?" check did not help.
- Fix: one shared background load per model (`ModelLoad`, `_load_for`,
  `begin_load`, `ensure_loaded`) with its own `LLM_LOAD_TIMEOUT_S` (1800 s) that
  no departing client can abort; `model_loading` status events with the elapsed
  time; the stream waits for the load, then generates.
- Tests: `tests/test_llm_client.py` (load survives a waiter leaving, load failure
  reported). Real run: AI Mode answered end to end over SSE (183 s) and in the
  browser (136.6 s). **VERIFIED (sandbox).**

**P0-2 After a mode switch or page reload every answer request returned "busy".**
`app/api/stream.py::event_stream`.
- Root cause: the generation slot was released only when the abandoned sync
  generator was garbage-collected; on an idle server that could be never.
- Fix: async SSE generator (a disconnect cancels it within 0.5 s); the worker
  thread releases the slot when the work has really stopped; a new request waits
  up to `GENERATION_WAIT_S` for a stopping one (status `queued`).
- Test: `tests/test_hardening.py` runs a real uvicorn server, disconnects
  mid-answer with the garbage collector disabled; fails on the old code, passes
  now. **VERIFIED.**

**P0-3 "Not enough memory" with Q8_0 in a 14 GB Docker VM (Docker Desktop on macOS).**
`docker-compose.yml`, `backend/scripts/make_text_model.py`, `askito`.
- Measured cause (sandbox): the Q8_0 runner peaked at **12.3 GB**: about 1.1 GB
  is a vision projector (`mmproj`) the app never uses, the rest of the growth was
  llama-server's host-RAM prompt cache. With the backend (about 1.1 GB),
  PostgreSQL, Redis and nginx this is at or over a 14 GB VM (estimate: the
  VM's own overhead was not measured).
- Fix (no weight change): `askito/qwen3.5:9b-q8_0-text` points at the **same Q8_0
  GGUF blob** (same digest) without the projector; prompt cache off
  (`LLAMA_ARG_CACHE_RAM=0`). Peak runner memory **10.6 GB**, byte-identical
  answers and citations on the 5 frozen smoke questions. `./askito doctor`
  prints the VM memory, per-container memory, the served model and Ollama's
  memory log lines; model errors carry a fixed-phrase reason.
- Status: **VERIFIED on the sandbox; NOT VERIFIED on Docker Desktop for macOS.**

**P0-4 Model problems were indistinguishable ("search_failed" / "No answer was produced").**
`app/llm/client.py::classify_error_text`, `app/api/errors.py`.
- Fix: `LLMError.code` = `model_memory`, `model_not_found`, `model_unreachable`,
  `model_timeout`, `model_load_timeout`, `generation_failed`; each has an
  actionable message; memory and missing-model errors are not retried; the
  answer card shows the message, the reason and a Retry button; results,
  sources and any partial answer stay.
- Tests: `tests/test_llm_client.py`, `tests/test_hardening.py`. Browser check of
  `model_not_found` and `model_unreachable`: **VERIFIED**. Low-memory path in the
  browser: **NOT VERIFIED** (the sandbox kernel killed the memory hog instead of
  the runner; `oom_score_adj` not permitted).

### P1

**P1-1 Evaluation page: "Run selected" did nothing.**
`app/api/eval.py::start_run`, `app/eval/run.py::run`.
- Reproduction (unchanged code, empty `eval_questions` as after `./askito setup`):
  `POST /api/eval/runs` → `200 {"status":"started"}`, `GET /api/eval/runs` →
  `{"runs":[]}`; server log: `RuntimeError: no eval_questions`.
- Root cause: setup does not load the test set; the background task raised and
  the error reached only the log. In addition, a run that failed half way stayed
  "running" forever; every run generated answers and RAGAS (hours on CPU) with
  no progress shown; the committed gold ids are tied to ingestion order.
- Fix: `app/eval/prepare.py` resolves the committed test set against the loaded
  corpus before each run (seed questions re-resolved by article title; missing
  gold articles left out and reported). `POST` validates first (400 / 409 with
  a message), creates `queued` rows at once, runs them in a background job
  (`app/eval/jobs.py`); rows show `running n/46`, `done` or `error` with the
  message; a backend restart marks unfinished rows. Retrieval-only runs are the
  default on the page; answers and RAGAS are opt-in.
- Tests: `tests/test_eval_jobs.py` (12). Real: the resolver reproduces all 46
  committed gold ids on the original corpus; a retrieval-only job over 46
  questions completed in the browser (bm25 doc_recall@5 0.693, vector 0.750,
  identical to the earlier full runs); a backend restart during `hybrid_rerank`
  showed "failed: interrupted". **VERIFIED.**

**P1-2 An `.env` pinned to `qwen3.5:9b` silently served Q4_K_M.**
`askito::migrate_env_model` moves old project defaults to the Q8_0 text-only tag
(backup `.env.bak`); explicit other values are kept. Test in
`tests/test_model_config.py`. **VERIFIED.**

**P1-3 No correlation id; INFO diagnostics were never printed.**
`app/observability.py`. The `askito.*` loggers had no handler, so Python printed
only warnings: per-request timing lines and ingestion results were invisible.
Now every response has `X-Request-ID` (a safe incoming id is kept), the SSE
events carry the same `rid`, and every log line has `rid=`. Tests:
`tests/test_observability.py`. **VERIFIED.**

**P1-4 `./askito up` did not apply new migrations** (0004 would be missing on an
existing database). It now runs the idempotent migrator after start.

**P1-5 A homepage suggestion abstained.** "RAG vs fine-tuning" did not get
fine-tuning evidence in a real AI Mode run; replaced. Every suggestion was
checked on the retrieval + evidence path (`scripts/suggestion_check.py`).

**P1-6 Partial evidence produced "not found" followed by an answer.**
`app/agent/prompts/answer.txt`, `overview.txt`.
- Reproduction (real Q8_0, AI Mode): "Compare the 2023 and 2024 Nobel Prizes in
  Physics." → "I could not find this in the sources." then a note with the 2023
  laureates.
- Root cause: the prompts allowed only all or nothing ("if the context does not
  contain the answer, say exactly …"), so partial evidence produced a refusal
  plus an answer, which reads as a contradiction.
- Fix: answer the covered part with citations and name the missing part; the
  exact refusal only when no passage is relevant (prompt version v5, so cached
  answers made with the old prompt are not reused).
- Tests: `tests/test_prompts.py`; real Q8_0 in section 4 (refusals kept).
- Remaining (not fixed): in that comparison the passage naming the 2024
  laureates *was* in the final context (checked in the agent trace and the
  chunk text), yet the model answered that the context does not name them, and
  in one run it listed a researcher from the attosecond passage as a 2023
  laureate. These are reading / faithfulness errors of the 9B model at
  temperature 0 with thinking off, not retrieval misses. A change that forces
  each search's best passage into the context was tried and reverted: the real
  run showed no benefit. Citation ids were valid in both cases; the offline
  claim-support judge (`app/eval/claim_support.py`) measures this, nothing in
  the request path does.

### P2

- Evaluation page header overflowed at 360 px (7 px horizontal scroll); fixed.
- A page-started evaluation rewrote the tracked `docs/EVAL_REPORT.md`; only
  `make eval` / `./askito eval` write it now.
- Ollama 0.40 can keep listing a model whose runner was killed while idle and
  never answer; documented recovery (`docker compose restart ollama`). Not
  fixable in the app; the UI shows the loading state until the load timeout.

## 3. Product changes

- **Search / AI Mode.** Two modes; the AI Mode toggle sits in the search bar;
  switching keeps the query; no separate "AI answer" tab. Search needs no LLM
  and works while Ollama is down.
- **Adaptive AI Mode** (`app/agent/router.py`, no LLM call): a single question
  takes the fast path (hybrid + reranker + evidence cut-off + one streamed
  answer); comparisons, relations, several entities or questions run the agent.
- **Dynamic Wikipedia ingestion** (`app/ingest/live.py`, migration 0004): on weak
  evidence (top reranker score < 0.95, AI Mode, all domains) up to 2 articles
  are fetched, chunked and embedded with the snapshot pipeline and stored in
  PostgreSQL (`origin='dynamic'`, revision, content hash, ingestion time), one
  transaction per article; BM25 is rebuilt atomically and the index version
  bumps every cache key. Only the question text is sent, and only then;
  `LIVE_INGEST_ENABLED=false` switches it off.
- **Homepage:** one corpus line (counts added articles separately; does not
  claim all of Wikipedia); the 2025 Physics / quantum suggestion kept.
- **UI scales** with the window (fluid root font size, wider columns on large
  screens, no horizontal scroll at 360 px).

## 4. Measurements (sandbox, CPU only)

| What | Result |
|---|---|
| Cold Q8_0 load from a cold disk | 689 s (before the fix the request gave up at 240 s and the load was aborted) |
| Q8_0 runner peak RSS, registry tag vs text-only tag | 12.3 GB vs 10.6 GB, identical answers on 5 smoke prompts |
| AI Mode, fast route ("What is a transmon qubit?") | route at 5.4 s, first token 23.6 s, done 44.3 s; 1 LLM call (871 prompt / 88 completion tokens); runner 10.6 GB, backend 1.6 GB |
| AI Mode, agent route ("Compare the 2023 and 2024 Nobel Prizes in Physics."), before P1-6 | route at 18.8 s (evidence check plus a Wikipedia lookup that got HTTP 429), first token 163.8 s, done 189.1 s; 6 LLM calls: analyze 1 (18.0 s), rewrite 2 (14.5 s), sufficiency 2 (62.0 s), answer 1 (47.6 s); 3,261 prompt / 326 completion tokens; runner 10.5 GB, backend 1.8 GB |
| Answer prompts v5, real Q8_0 | comparison: partial answer with citations, names the missing part (no "not found" + note); 4 of 4 out-of-corpus probes still answer exactly "I could not find this in the sources." (31–35 s each) |
| Agent route after v5, same comparison | 154.6 s, 6 LLM calls, 4 searches (2023 ×2, 2024 ×2); answer: partial, cited; says the 2024 laureates are not named although the passage naming them was in the context (model reading error) |
| Agent route, relation ("How do Hopfield networks relate to the work of Geoffrey Hinton?") | 192.7 s, 7 LLM calls, 5 searches; cited answer on both sides, states what the passages do not connect |
| AI Mode with Wikipedia ingestion ("What is a small language model?", DB copy) | 2 articles / 15 chunks added, answer at 69.6 s citing the new article |
| Retrieval-only evaluation, 46 questions | bm25 doc_recall@5 0.693, vector 0.750, hybrid 0.716 (DB copy incl. 2 added articles) |

Not measured: GPU tokens/s and VRAM, Docker Desktop on macOS, the target PC,
RAGAS with the real judge (hours on CPU; not run).

## 5. Tests

Commands (backend in `backend/`, frontend in `frontend/`):

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
ASKITO_LIVE_DB_TEST=1 .venv/bin/python -m pytest -q tests/test_live_ingest.py -k dense_and_bm25
npm run typecheck
npm run build
```

Results (final run): ruff clean; **302 passed, 1 skipped** (the skipped one is
the opt-in real-database test, run separately: **1 passed**); frontend
typecheck clean, production build OK.

What is real and what is mocked:
- **Mocked:** the LLM in pytest (`LLM_MODE=mock`, scripted agent), Wikipedia
  in pytest (fake client; tests never contact it), the database in the eval-job
  and ingestion unit tests (fake connections).
- **Real in pytest:** PostgreSQL + pgvector + BM25 + embeddings (retrieval
  tests); a throwaway `redis-server` (Redis cache tests); a real uvicorn server
  (disconnect test); the opt-in ingestion test above runs on a copy of the
  database with real embeddings, BM25 and Redis and checks that after an
  ingestion the new chunks are found by BM25 and pgvector, the cached
  pre-ingestion retrieval is not served (L1 and Redis L2 miss), the answer cache
  key changes and the in-memory chunk store reloads.
- **Real outside pytest:** Ollama Q8_0 runs over SSE and in Chromium
  (Playwright) listed in section 4; live Wikipedia over SSE on a database copy.

Integration scenarios of the plan:

| # | Scenario | Status |
|---|---|---|
| 1 | Query → corpus retrieval → answer | VERIFIED (real Q8_0, SSE + browser) |
| 2 | Weak evidence → Wikipedia → PostgreSQL → retrieval → answer | VERIFIED over SSE with live Wikipedia; in the browser the same day Wikipedia answered HTTP 429 (rate limit), so the browser success path is NOT VERIFIED |
| 3 | Redis hit and invalidation after a corpus update | VERIFIED (real Redis, real ingestion) |
| 4 | Redis unavailable, Search still works | VERIFIED (pytest) |
| 5 | Wikipedia unavailable → fallback / abstention | VERIFIED in the browser (HTTP 429: "Wikipedia could not be reached…", answer from the local corpus or "not found") |
| 6 | Citation verification and source links | citation id validity and claim-support judge from the earlier remediation; unchanged here |
| 7 | Q8_0 configured, no silent Q4 fallback | VERIFIED (config tests, identity check, `.env` migration) |

## 6. Security and privacy checks

- Local-only LLM and Redis endpoints (loopback by default); unchanged and tested.
- Wikipedia: fixed endpoint `https://en.wikipedia.org/w/api.php`; titles go in
  query parameters (no URL from user input is fetched); query text capped at
  300 characters; time, size and article bounds; one ingestion at a time.
- Wikipedia text is untrusted evidence; the answer prompt's injection guard
  applies to it like to every passage.
- Request sizes: `q` ≤ 500 characters; eval configurations validated.
- `X-Request-ID` accepted only as `[A-Za-z0-9._-]{1,64}`, otherwise replaced.
- Logs: timings, counts, error classes and ids; no question or answer text, no
  credentials.

## 7. Not verified / remaining risks

- **Docker Desktop on macOS (14 GB VM): NOT VERIFIED.** The
  memory fix is measured on Linux; the VM's own overhead is unknown.
- **GPU offload and the target PC: NOT VERIFIED.**
- **Low-memory error in the browser: NOT VERIFIED** (could not be provoked here).
- **Wikipedia ingestion in the browser: NOT VERIFIED** today (HTTP 429 from
  Wikipedia); the API path was verified earlier.
- **RAGAS with the real judge and answered evaluation runs: NOT RUN** (hours on
  CPU). Retrieval-only evaluation runs are verified.
- The 9B model can attribute a fact to the wrong person even with valid
  citation ids (P1-6); citations show the source, they do not prove each claim.
- On CPU an answered evaluation run blocks AI Mode answers while it runs
  (single local model); the page says so.
- An ingested article stays in the corpus (by design); there is no UI to remove
  dynamic articles (`DELETE FROM documents WHERE origin='dynamic'` + `./askito index`).

## 8. Checks on the target machine

```bash
git pull
./askito model     # Q8_0 source + text-only tag
./askito up        # also applies migrations
./askito doctor    # Docker memory, served model (Q8_0), Ollama memory lines
```

Then: one AI Mode question (first one loads the model), one question outside
the corpus (expect "Added from Wikipedia"), and on `/eval` a retrieval-only run
of bm25, vector, hybrid, hybrid_rerank (a few minutes).
