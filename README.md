# Ask ITO

Agentic RAG search engine and evaluation panel over a focused English Wikipedia
corpus: recent Nobel Prizes in Physics, Chemistry and Physiology/Medicine, a
quantum-computing branch, and core AI topics.

The default path is **Fast Hybrid RAG**: BM25 + pgvector retrieval → RRF
fusion → ranked sources, and — only when the user asks for it — a cross-encoder
rerank and **one** local LLM call (Qwen3.5-9B via Ollama) that writes a short
cited answer. Three modes:

- **Search** (default, `search`): ranked passages, no LLM call (~0.1 s
  retrieval on a 4-core CPU).
- **Answer** (`overview`, opt-in: "Generate AI answer"): bge rerank + exactly one
  local LLM call, citations. On a CPU-only machine this takes about a minute.
- **AI Mode** (`agentic`, opt-in): the multi-step agent (analysis, query
  rewriting, sufficiency loop) for harder multi-hop questions.

When no passage passes the evidence cut-off (a heuristic on the reranker score,
`FAST_MIN_EVIDENCE_SCORE`; it stops clearly off-topic queries, not every
unsupported question) the answer is "not found" and the LLM is not called.
Citation numbers point to retrieved passages; they do not by themselves verify
each claim.
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

**Ask ITO searches a small, fixed corpus — it is not a general Wikipedia search
engine.** The snapshot holds **258 English Wikipedia documents** (→ **5,988
chunks**) in three domains: `nobel` (Physics, Chemistry and Physiology/Medicine
prizes of the last five years 2022–2026 plus landmark prizes), `quantum`
(quantum computing) and `ai` (computer vision, Transformers & LLMs, RAG,
autonomous agents, deep learning). Other Nobel categories (Literature, Peace,
Economics) and topics outside these domains are not covered; such questions
get a "not found" answer. Broad questions ("popular AI fields") are answered
from what the retrieved passages contain, not as a complete or ranked list.
The UI shows the article count and snapshot date. The scope was refined from
the spec's original target for feasibility; see ADR-0007. Growing it means
running the ingest pipeline with more seeds (`./askito ingest`, then `index`).

### AI Mode adds Wikipedia articles when the corpus has no good evidence

In AI Mode (all domains), when the best reranker score for a question is below
`LIVE_INGEST_MIN_SCORE` (0.95), the backend searches English Wikipedia for the
question, fetches up to `LIVE_INGEST_MAX_ARTICLES` (2) articles, cleans and
chunks them with the snapshot pipeline (same chunker, same embedding model) and
stores them in PostgreSQL, one transaction per article, as `origin='dynamic'`
documents in the `general` domain. The BM25 index is then rebuilt from the
database, which bumps the index version, so every retrieval and answer cache
picks up the new chunks; the answer is generated afterwards and can cite them.
The page shows "Added from Wikipedia" with the article titles, and the corpus
line counts them separately. Search mode never does this.

- **What leaves the machine:** only the question text, as the Wikipedia search
  query (plus the article titles fetched). Nothing is sent when the local
  evidence is good. Set `LIVE_INGEST_ENABLED=false` to never contact Wikipedia
  at query time.
- **Bounds:** `LIVE_INGEST_TIMEOUT_S` (30 s) for search + fetch, 10 s per HTTP
  request, `LIVE_INGEST_MAX_CHARS` (40,000) per article, one ingestion at a time.
  If Wikipedia is unreachable the answer is generated from the local corpus.
- **Idempotent:** an article already stored with the same revision is skipped;
  a newer revision replaces its chunks; snapshot articles are never rewritten.
  Added articles are also saved under `backend/data/raw/`, so a later full
  `./askito index` keeps them.
- Wikipedia text is untrusted evidence: the answer prompt treats every passage
  as data, never as instructions.

## Stack

- **DB**: PostgreSQL 16 + pgvector (HNSW, `vector_cosine_ops`)
- **BM25**: `bm25s` (English stopwords + stemming)
- **Embedding**: `BAAI/bge-base-en-v1.5` (768-dim, config-driven)
- **Reranker**: `BAAI/bge-reranker-base` (default, top 20 candidates, 512 tokens) or
  `cross-encoder/ms-marco-MiniLM-L-6-v2`; off by default in Search (`SEARCH_RERANK`),
  on for answers (`ANSWER_RERANK`)
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
./askito model           # optional: pull the LLM for generated answers
./askito setup           # first run: migrate + ingest (Wikipedia) + index
```

- Search UI: http://localhost:5173
- Backend API + docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health (`./askito health`)

The production compose file serves the built frontend with nginx and runs the
backend without `--reload`. For hot reload during development use
`./askito dev` (vite dev server + `uvicorn --reload`).

With `make`: `make up`, `make model`, `make migrate`, `make ingest`, `make index`.

### Local use and Netlify

Docker Compose runs the frontend, FastAPI backend, PostgreSQL, Redis and Ollama
on the same machine. No separately hosted backend is needed for local use. Open
the UI at `http://localhost:5173`; the frontend's nginx proxy sends `/api/*` to
the local backend.

On a fresh checkout, `./askito setup` downloads the Wikipedia snapshot and builds
the pgvector and BM25 indexes. The first embedding/index pass is CPU-intensive
and can take a while. Search is ready when `./askito health` reports a nonzero
chunk count and BM25 as available. Ollama is not needed for Sources/Search mode;
`./askito model` is only needed for locally generated AI answers.

Netlify publishes only the static frontend; the catch-all redirect in
`netlify.toml` sends unknown paths (including `/api/*`) to `index.html`. To use a
separately hosted backend, set the frontend build environment variable
`VITE_API_BASE` to its public HTTPS origin (for example,
`https://api.example.com`) and set the backend's `FRONTEND_ORIGIN` to
`https://ask-ito.netlify.app`, then redeploy. Do not use `localhost` for
`VITE_API_BASE` on Netlify: it refers to each visitor's own computer. A Netlify
deploy does not provision the backend, database, Redis or Ollama services.

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
`qwen3.5:9b-q8_0` — Q8_0 weights (`LLM_DEFAULT_MODEL`; pull it once with
`./askito model`, which also prints the quantization Ollama reports). The
Q4_K_M build `qwen3.5:9b-q4_K_M` (same as `qwen3.5:9b`) is the baseline and the
rollback if Q8_0 does not fit in memory; no lower weight quantization is
recommended. The tag is set only in `.env` / `app/config.py`. It is
the answer generator only: retrieval and reranking do not use it, and the
default path calls it once per answer (temperature 0, at most
`OVERVIEW_MAX_TOKENS`=320 tokens, top `OVERVIEW_CONTEXT_CHUNKS`=4 passages).

Reasoning output is off (`LLM_THINK=false`; sent only to models that report
the thinking capability), which matters a lot for speed.

*KV cache.* The attention KV cache and prompt-prefix reuse live inside Ollama,
never in Redis. Compose sets `OLLAMA_KV_CACHE_TYPE=f16` (no extra compression)
with flash attention; set `OLLAMA_KV_CACHE_TYPE=q8_0` in `.env` only if f16
runs out of memory. This is independent of the weight quantization in the tag.

*Redis (optional cache).* Compose runs a local `redis` service (no host port,
no persistence, 256 MB LRU) as a shared L2 behind the in-process caches: query
embeddings, retrieval results (chunk ids + scores), a front cache for the
PostgreSQL answer cache and the agent's deterministic tool calls. Keys carry the
index version, models (including the served weight quantization), prompt
version and retrieval settings, and every entry has a TTL. If Redis is down the
backend logs one warning and keeps working on its in-process and PostgreSQL
caches. Outside Docker it is off (`REDIS_ENABLED=false`).
After changing `.env` run `./askito model` (if the model is new) and
`./askito restart`.

*Memory and first load.* Q8_0 is the default and stays the default. The registry
tag `qwen3.5:9b-q8_0` also contains a vision projector that Ollama loads with
every model load (about 1.1 GB) although Ask ITO never sends images, so
`./askito model` pulls that tag and then creates `askito/qwen3.5:9b-q8_0-text`:
the **same Q8_0 GGUF blob** (no copy, same digest) with the same template,
renderer, parser and parameters, without the projector
(`backend/scripts/make_text_model.py`). Measured on a CPU-only sandbox with the
5 smoke questions and identical prompts: **byte-identical answers and citations**;
peak Ollama memory **10.6 GB instead of 12.3 GB**, flat across questions because
llama-server's own prompt cache is now off (`LLAMA_ARG_CACHE_RAM=0`; repeats are
served by the app's answer and agent tool caches). The backend's embedding and
reranker models need about 1.1 GB more, so give Docker **at least 14 GB**. On a
16 GB Mac that leaves macOS about 2 GB, which is the practical limit. With too
little memory AI Mode shows "not enough memory" plus what Ollama reported (Search
keeps working); `./askito doctor` prints Docker's memory, per-container memory,
the loaded model and Ollama's own memory messages. An older `.env` that still
names `qwen3.5:9b` (that tag is Q4_K_M) or the plain Q8_0 tag is moved to the new
default by `./askito up` / `./askito model` (backup in `.env.bak`).
The first answer after a restart loads about 10 GB from disk: 689 s measured on
the sandbox with a cold disk cache. The UI shows "Loading the local model…" with
the elapsed time; the load runs in one shared background request
(`LLM_LOAD_TIMEOUT_S`, default 30 min), so leaving the page does not abort it.
Q4_K_M (`qwen3.5:9b-q4_K_M`) is kept only as an explicit benchmark /
troubleshooting option; nothing switches to it automatically.

*Docker Desktop on macOS.* Docker Desktop runs Linux containers in a VM without
GPU access, so the `ollama` service runs CPU-only there, inside the VM's memory
limit (Settings → Resources → Memory; give it at least 14 GB for Q8_0). Running
Ollama natively on the Mac (Metal GPU) and pointing the backend at it with
`LLM_BASE_URL=http://host.docker.internal:11434` plus
`LLM_ALLOWED_HOSTS=host.docker.internal` is possible but **NOT VERIFIED** in this
project (`host.docker.internal` must resolve to a private address to pass the
local-only check).

*Windows / Docker Desktop memory:* WSL 2 gives Docker about half of the host
RAM by default. If the model does not fit, raise the limit in
`%UserProfile%\.wslconfig`, then run `wsl --shutdown` and restart Docker Desktop:

```ini
[wsl2]
memory=26GB
```

**Local only.** All inference — answers, agent steps and the RAGAS judge —
runs on local models. `LLM_LOCAL_ONLY=true` (default) accepts **loopback
endpoints only** (`localhost`, `127.0.0.0/8`, `::1`; http/https, no credentials
in the URL); private-network, LAN, link-local and public addresses are refused,
every request is re-checked and redirects are never followed. The single
explicit exception is `LLM_ALLOWED_HOSTS`: docker-compose sets it to `ollama`,
which may resolve only to a private bridge-network address. A locally served
OpenAI-compatible runtime (llama.cpp server, vLLM) can be used with
`LLM_MODE=openai`.

`LLM_MODE=mock` uses a deterministic stub (dev/tests): retrieval, indexing and
retrieval metrics work fully without an LLM; answers and RAGAS need a real one.

### Evaluation

The panel at http://localhost:5173/eval works on a fresh setup: the committed
test set (`backend/eval/testset.jsonl`, 46 hand-authored questions) is loaded
when a run starts. Gold ids depend on ingestion order, so every run first
resolves the gold articles by title against the corpus that is loaded (the same
way `make testset-seed` builds them); a question whose gold article is not in
the corpus is left out and reported. On the corpus the test set was built from
this reproduces the committed ids exactly.

"Run selected" queues one run per configuration and the page shows each one as
queued, running (questions done / total), done, or failed with the reason. By
default a run scores the ranked passages only (Recall@k, MRR, nDCG; a few
minutes in total). "Generate answers" also calls the local model for every
question (needed for `closed_book`, `agentic`, refusal and RAGAS metrics; about
a minute per question on CPU with Q8_0, and AI Mode waits for the model
meanwhile); "RAGAS judge" adds many more model calls. One evaluation runs at a
time; a run cut short by a backend restart is marked as failed.

```bash
make testset-seed        # rebuild the seed test set against this corpus
make testset-draft       # (optional) LLM-generate more drafts to review
make testset-load        # load reviewed testset.jsonl into eval_questions
make eval CONFIGS=all     # all modes with answers + RAGAS, write docs/EVAL_REPORT.md
./askito eval bm25,hybrid,hybrid_rerank --retrieval-only   # passages only, no LLM
```

Run detail, compare and the test-set review (`/eval/review`) are linked from
the panel.

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
| `POST` | `/api/eval/runs` | `{configs, answers?, ragas?}`: queue eval runs (409 if one is running or no question fits the corpus) |
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
  ~4 s at 256 tokens; MiniLM ~0.8 s. Search skips it by default
  (`SEARCH_RERANK=false`); answers use it (`ANSWER_RERANK=true`).
- qwen3.5:9b (Q4_K_M) on this CPU: prefill ~19–26 tok/s, generation ~3.2 tok/s;
  a Fast Answer took 64 s median end to end (first token after ~19–29 s) on 12
  eval questions, versus 92 s for the previous default path with the same model.
  A faster CPU or a GPU shortens the LLM part roughly in proportion. A repeated
  query is served from the answer cache.

## Tests

`make test` runs the backend test suite (`pytest`) and the frontend typecheck.
Backend tests that need the index/DB are skipped automatically when it is not
built, so the suite is green on a fresh checkout; build the index to exercise
the retrieval/eval integration tests.
