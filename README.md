# Ask ITO

Agentic RAG search engine and evaluation panel over a focused English Wikipedia
corpus: recent Nobel Prizes in Physics, Chemistry and Physiology/Medicine, a
quantum-computing branch, and core AI topics.

The pipeline combines **BM25 + vector hybrid retrieval**, a **cross-encoder
reranker**, LLM query rewriting and a multi-round **agentic** loop, and answers
with citations. A separate evaluation panel compares pipeline configurations
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
- **Reranker**: `BAAI/bge-reranker-base` (cross-encoder)
- **LLM**: OpenAI-compatible client; default local Ollama, plus a `mock` mode
- **Backend**: Python 3.11+, FastAPI, psycopg 3
- **Frontend**: React + Vite + TypeScript + Tailwind, Recharts
- **Eval**: RAGAS + Recall@k / MRR / nDCG
- **Orchestration**: Docker Compose

## Quick start (Docker)

```bash
cp .env.example .env     # set WIKI_USER_AGENT contact and your LLM model
make up                  # start db, backend, frontend
make migrate             # apply DB migrations
```

- Frontend: http://localhost:5173
- Backend API + docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

### Build the corpus and index

```bash
make ingest              # fetch + clean Wikipedia snapshot (needs network)
make index               # chunk, embed (pgvector) and build the BM25 index
```

`make ingest` calls the Wikipedia Action API; run it from an unthrottled network
(a data-center/shared IP may be rate-limited — the client backs off and the run
is resumable). See ADR-0007/0008.

### Local LLM (for answering, rewriting, agentic and RAGAS)

Query/answer generation needs an OpenAI-compatible LLM; by default a local
[Ollama](https://ollama.com):

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull qwen3:30b   # or your model
```

Set `LLM_DEFAULT_MODEL` (and optionally `LLM_AGENT_MODEL` / `LLM_ANSWER_MODEL` /
`LLM_JUDGE_MODEL`) in `.env`. For dev/offline tests set `LLM_MODE=mock` to use a
deterministic stub — retrieval, indexing and retrieval metrics work fully
without an LLM; answers and RAGAS need a real one.

### Evaluation

```bash
make testset-seed        # build the hand-authored seed test set
make testset-draft       # (optional) LLM-generate more drafts to review
make testset-load        # load reviewed testset.jsonl into eval_questions
make eval CONFIGS=all     # run all modes, write docs/EVAL_REPORT.md
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
| `POST` | `/api/search` | `{query, mode, domain, final_k?}` → answer, citations, sources, trace |
| `GET/POST` | `/api/eval/review` | Test-set review (list / accept / reject) |
| `GET` | `/api/eval/runs`, `/api/eval/runs/{id}` | Eval runs and per-question results |
| `POST` | `/api/eval/runs` | Start an eval run in the background |
| `GET` | `/api/eval/compare?a=&b=` | Compare two runs per question |

Modes: `closed_book`, `bm25`, `vector`, `hybrid`, `hybrid_rerank`,
`hybrid_rerank_rewrite`, `agentic` (UI default).

## Hardware notes

Embedding and reranking run on CPU. Reference timings on this project's build
machine (CPU-only):

- Indexing 258 docs → 5,988 chunks (embed + BM25): a few minutes.
- A warm `hybrid_rerank` query: ~10–15 s (cross-encoder reranking dominates).
- `bm25` / `vector` queries: tens to a few hundred ms.

For faster indexing, switch to `bge-small-en-v1.5` (`EMBEDDING_DIM=384`) and
re-run `make migrate` + `make index`. A lighter reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) can be set via `RERANKER_MODEL`.

## Tests

`make test` runs the backend test suite (`pytest`) and the frontend typecheck.
Backend tests that need the index/DB are skipped automatically when it is not
built, so the suite is green on a fresh checkout; build the index to exercise
the retrieval/eval integration tests.
