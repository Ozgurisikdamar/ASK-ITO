# Ask ITO

Agentic RAG search engine and evaluation panel over a focused English Wikipedia
corpus: recent Nobel Prizes in Physics, Chemistry and Physiology/Medicine, a
quantum-computing branch, and core AI topics.

The default path is **Fast Hybrid RAG**: BM25 + pgvector retrieval → RRF
fusion → optional cross-encoder reranker → **one** local LLM call (Qwen3.5-9B
via Ollama) that writes a short cited answer. Results appear as soon as
retrieval finishes and the answer streams in above them. Three modes:

- **Answer** (default, `overview`): retrieval + exactly one LLM call, citations.
- **Sources** (`search`): the same retrieval, ranked passages only, no LLM call.
- **AI Mode** (`agentic`, opt-in): the multi-step agent (analysis, query
  rewriting, sufficiency loop) for harder multi-hop questions.

When no passage is relevant enough (reranker score below
`FAST_MIN_EVIDENCE_SCORE`) the answer is "not found" and the LLM is not called.
A separate evaluation panel compares pipeline configurations with **RAGAS** and
classic retrieval metrics (Recall@k, MRR, nDCG); evaluation runs offline and
never during a search. Everything runs locally; no paid or cloud service is used.

## Status

All milestones implemented (spec §12):

| # | Milestone | State |
|---|---|---|
| M1 | Skeleton: repo, Docker Compose, migrations, `/api/health`, themed home page | ✅ |
| M2 | Ingest: seed resolution, Wikipedia fetch, one-step expansion, cleaning, snapshot | ✅ |
| M3 | Indexing: section-aware chunking, bge embeddings in pgvector (HNSW), BM25 | ✅ |
| M4 | Retrieval pipeline + `/api/search` (all non-agentic modes, reranker, trace) | ✅ |
| M5 | Agentic layer: analysis, sub-questions, sufficiency loop, citation validation | ✅ |
| M6 | Search UI: answer card, clickable citations, sources, "How it searched" panel | ✅ |
| M7 | Test set: draft generator, `/eval/review`, hand-authored seed | ✅ |
| M8 | Eval runner, metrics, RAGAS, panel (ablation, charts, run detail, compare) | ✅ |
| M9 | Polish: README, `make test`, decisions | ✅ |
| M10 | Speed: streaming search (SSE), overview mode, chunk store, caches, warm-up, LLM adapters | ✅ |
| Fast RAG | Fast Hybrid RAG default, Sources mode, evidence cut-off, stage timings, local-only ([report](docs/ASK_ITO_FAST_RAG_IMPLEMENTATION_REPORT.md)) | ✅ |

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for architecture decisions and
deviations, and [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md) for the latest
evaluation run.

## Corpus

A frozen snapshot of **258 English Wikipedia documents** (→ **5,988 chunks**)
across three domains: `nobel` (last five years 2022–2026 plus landmark prizes),
`quantum` (quantum computing), and `ai` (computer vision, Transformers & LLMs,
RAG, autonomous agents, deep learning). The scope was refined from the spec's
original target for feasibility; see ADR-0007.

## Stack

- **DB**: PostgreSQL 16 + pgvector (HNSW, `vector_cosine_ops`)
- **BM25**: `bm25s` (English stopwords + stemming)
- **Embedding**: `BAAI/bge-base-en-v1.5` (768-dim, config-driven)
- **Reranker**: `BAAI/bge-reranker-base` (default, top 20 candidates, 512 tokens) or
  `cross-encoder/ms-marco-MiniLM-L-6-v2`; switchable off in the fast modes (`FAST_RERANK`)
- **LLM**: `qwen3.5:9b` on local Ollama (native API, streaming, thinking off);
  a locally served OpenAI-compatible endpoint optionally; `mock` mode for tests.
  Local only: non-local endpoints are refused (`LLM_LOCAL_ONLY=true`); no fallback
- **Backend**: Python 3.11+, FastAPI, psycopg 3
- **Frontend**: React + Vite + TypeScript + Tailwind, Recharts
- **Eval**: RAGAS + Recall@k / MRR / nDCG
- **Orchestration**: Docker Compose

## Quick start (Docker)

Requirements: Docker Desktop (or Docker Engine + Compose v2.24+). On Windows the
commands below run in **Git Bash**; `./askito` replaces `make` (both exist).

```bash
./askito up              # creates .env from .env.example if missing; starts everything
./askito model           # one-time: pull the LLM (LLM_DEFAULT_MODEL) into Ollama
./askito setup           # first run: migrate + ingest (Wikipedia) + index
```

- Search UI: http://localhost:5173
- Backend API + docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health (`./askito health`)

The production compose file serves the built frontend with nginx and runs the
backend without `--reload`. For hot reload during development use
`./askito dev` (vite dev server + `uvicorn --reload`).

With `make`: `make up`, `make model`, `make migrate`, `make ingest`, `make index`.

### Build the corpus and index

```bash
./askito ingest          # fetch + clean Wikipedia snapshot (needs network)
./askito index           # chunk, embed (pgvector) and build the BM25 index
```

`ingest` calls the Wikipedia Action API; run it from an unthrottled network
(a data-center/shared IP may be rate-limited — the client backs off and the run
is resumable). See ADR-0007/0008.

### LLM: local Ollama

**Local (default).** The `ollama` service starts with the stack and keeps the
model loaded in RAM (`OLLAMA_KEEP_ALIVE=24h`). The default answer model is
`qwen3.5:9b` (`LLM_DEFAULT_MODEL`; pull it once with `./askito model`). It is
the answer generator only: retrieval and reranking do not use it, and the
default path calls it once per answer (temperature 0, at most
`OVERVIEW_MAX_TOKENS`=320 tokens, top `OVERVIEW_CONTEXT_CHUNKS`=4 passages).

Reasoning output is off (`LLM_THINK=false`; sent only to models that report
the thinking capability), which matters a lot for speed.
After changing `.env` run `./askito model` (if the model is new) and
`./askito restart`.

*Windows / Docker Desktop memory:* WSL 2 gives Docker about half of the host
RAM by default. qwen3.5:9b needs ~9 GB while loaded (measured); if you use a
larger model, raise the limit in
`%UserProfile%\.wslconfig`, then run `wsl --shutdown` and restart Docker Desktop:

```ini
[wsl2]
memory=26GB
```

**Local only.** All inference — answers, agent steps and the RAGAS judge —
runs on local models. `LLM_LOCAL_ONLY=true` (default) refuses any endpoint that
does not resolve to a loopback or private-network address. A locally served
OpenAI-compatible runtime (llama.cpp server, vLLM) can be used with
`LLM_MODE=openai`.

`LLM_MODE=mock` uses a deterministic stub (dev/tests): retrieval, indexing and
retrieval metrics work fully without an LLM; answers and RAGAS need a real one.

### Evaluation

```bash
make testset-seed        # build the hand-authored seed test set
make testset-draft       # (optional) LLM-generate more drafts to review
make testset-load        # load reviewed testset.jsonl into eval_questions
make eval CONFIGS=all     # run all modes, write docs/EVAL_REPORT.md (or ./askito eval)
```

Then open the panel at http://localhost:5173/eval (ablation table, charts, run
detail, compare) and the test-set review at `/eval/review`.

## Running without Docker (dev)

```bash
# Backend (needs a reachable Postgres 16 + pgvector)
cd backend
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
pip install -e ".[dev,index,eval]"
python -m app.db.migrate
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Make targets

| Target | Description |
|---|---|
| `make up` / `make down` | Start / stop Docker Compose services |
| `make up-dev` | Start with hot reload (vite + `uvicorn --reload`) |
| `make model` | Pull the LLM into Ollama (`MODEL=...` to override) |
| `make bench ARGS=...` | Latency benchmark (`ARGS=--llm` also times the streamed overview) |
| `make migrate` | Apply DB migrations |
| `make ingest` | Resolve seeds, fetch, expand, clean (snapshot); `FORCE=1` to re-fetch |
| `make index` | Chunk, embed, write pgvector + build BM25 index |
| `make testset-seed` | Build the hand-authored seed test set into `testset.jsonl` |
| `make testset-draft` | LLM-generate a draft test set for review (needs an LLM) |
| `make testset-load` | Load reviewed `testset.jsonl` into `eval_questions` |
| `make eval CONFIGS=...` | Run evaluation (`all` or e.g. `hybrid,agentic`) and write `EVAL_REPORT.md` |
| `make test` | Backend tests + frontend typecheck |
| `make dev` | Print dev run commands |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | DB, BM25, embedding, reranker, RAGAS and LLM status |
| `POST` | `/api/search` | `{query, mode, domain, final_k?}` → answer, citations, sources, trace, results |
| `GET` | `/api/search/stream?q&mode&domain` | Server-Sent Events: `results` → `reranked` → `sources` → `token`… → `done` / `error` |
| `GET` | `/api/suggest/featured` | Featured queries (pre-computed in the background at startup) |
| `GET/POST` | `/api/eval/review` | Test-set review (list / accept / reject) |
| `GET` | `/api/eval/runs`, `/api/eval/runs/{id}` | Eval runs and per-question results |
| `POST` | `/api/eval/runs` | Start an eval run in the background |
| `GET` | `/api/eval/compare?a=&b=` | Compare two runs per question |

Modes: `overview` (default, Answer: hybrid + rerank + one LLM call), `search`
(Sources: same retrieval, no LLM call), `agentic` (AI Mode, opt-in), and the
evaluation modes `closed_book`, `bm25`, `vector`, `hybrid`, `hybrid_rerank`,
`hybrid_rerank_rewrite`. Responses carry per-stage timings (embedding, BM25,
pgvector, RRF, rerank, context, first token, generation, total) and the names
of the LLM and reranker used.

## Hardware notes

Embedding, reranking and the LLM run on CPU. Measured on a 4-vCPU Intel Xeon
(2.8 GHz, AVX-512), 15 GB RAM, no GPU — see
[`docs/ASK_ITO_FAST_RAG_IMPLEMENTATION_REPORT.md`](docs/ASK_ITO_FAST_RAG_IMPLEMENTATION_REPORT.md)
for the full tables:

- Retrieval without reranker (embedding + BM25 + pgvector + RRF): ~0.1 s p50.
- bge-reranker-base over 20 candidates: ~7.7 s p50 at 512 tokens (default),
  ~4 s at 256 tokens; MiniLM ~0.8 s; `FAST_RERANK=false` skips it.
- qwen3.5:9b (Q4_K_M) on this CPU: prefill ~19 tok/s, generation ~3 tok/s, so a
  full answer takes about a minute; a faster CPU or a GPU shortens this roughly
  in proportion. A repeated query is served from the answer cache.

## Tests

`make test` runs the backend test suite (`pytest`) and the frontend typecheck.
Backend tests that need the index/DB are skipped automatically when it is not
built, so the suite is green on a fresh checkout; build the index to exercise
the retrieval/eval integration tests.
