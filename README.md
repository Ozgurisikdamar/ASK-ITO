# Ask ITO

Agentic RAG search engine and evaluation panel over a focused English Wikipedia
corpus: recent Nobel Prizes in Physics, Chemistry and Physiology/Medicine, a
quantum-computing branch, and core AI topics.

The pipeline combines **BM25 + vector hybrid retrieval**, a **cross-encoder
reranker**, LLM query rewriting and a multi-round **agentic** loop, and answers
with citations. The search page works like a web search engine: results appear
as soon as retrieval finishes and a short cited **AI Overview** streams in above
them; the multi-step agent runs in the **AI Mode** tab. A separate evaluation panel compares pipeline configurations
with **RAGAS** and classic retrieval metrics (Recall@k, MRR, nDCG). Everything
is open source and runs locally; no paid service is required.

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
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default) or `BAAI/bge-reranker-base`
- **LLM**: local Ollama (native API, streaming) by default; any OpenAI-compatible
  endpoint (e.g. Groq free tier) optionally; `mock` mode for tests
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

### LLM: local Ollama (default) or Groq (optional, free tier)

**Local (default).** The `ollama` service starts with the stack and keeps the
model loaded in RAM (`OLLAMA_KEEP_ALIVE=24h`). Pick the model in `.env`:

| `LLM_DEFAULT_MODEL` | Type | On a CPU-only machine |
|---|---|---|
| `qwen3:30b` (default) | 30B mixture-of-experts, ~3B active | usable: overview starts in seconds to tens of seconds |
| `qwen3.8:27b` | dense 27B | higher quality, several times slower per token |

Reasoning output is off (`LLM_THINK=false`), which matters a lot for speed.
After changing `.env` run `./askito model` (if the model is new) and
`./askito restart`.

*Windows / Docker Desktop memory:* WSL 2 gives Docker about half of the host
RAM by default, which is less than a ~20 GB model needs. Raise it in
`%UserProfile%\.wslconfig`, then run `wsl --shutdown` and restart Docker Desktop:

```ini
[wsl2]
memory=26GB
```

**Groq (optional).** For a fast hosted model without a GPU, create a free key
at https://console.groq.com/keys, then in `.env` uncomment the Groq block
(`LLM_MODE=openai`, `LLM_BASE_URL=https://api.groq.com/openai/v1`,
`LLM_API_KEY`, `LLM_DEFAULT_MODEL=qwen/qwen3-32b`,
`LLM_EXTRA_BODY={"reasoning_effort":"none"}`) and `./askito restart`. Check
current model ids at https://console.groq.com/docs/models. Queries and the
retrieved Wikipedia passages are sent to Groq when it is enabled. Set
`LLM_FALLBACK_MODE=ollama` (+ `LLM_FALLBACK_MODEL`) to fall back to the local
model when the free-tier limit is reached.

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

Modes: `overview` (default: hybrid + rerank + short streamed answer),
`closed_book`, `bm25`, `vector`, `hybrid`, `hybrid_rerank`,
`hybrid_rerank_rewrite`, `agentic` ("AI Mode").

## Hardware notes

Embedding, reranking and (by default) the LLM run on CPU. Measured in the build
environment (CPU-only, warm, `backend/scripts/bench.py`):

- Results list (BM25 + vector + RRF): ~60–100 ms after the query arrives.
- Cross-encoder rerank (MiniLM, 20 passages): ~0.5 s p50 (bge-reranker-base: ~3 s).
- A repeated query or a featured query (cached): ~50 ms end to end.
- Startup warm-up (models + chunk store): ~10 s, then background pre-compute of
  the featured queries.

LLM time depends on the machine and model and is not included above. Measure it
on your machine with `./askito bench --llm` (time to first token and total for
the streamed overview). Indexing 258 docs → 5,988 chunks takes a few minutes.

## Tests

`make test` runs the backend test suite (`pytest`) and the frontend typecheck.
Backend tests that need the index/DB are skipped automatically when it is not
built, so the suite is green on a fresh checkout; build the index to exercise
the retrieval/eval integration tests.
