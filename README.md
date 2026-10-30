# Ask ITO

Agentic RAG search engine and evaluation panel over a focused English
Wikipedia corpus: recent (2017–2026) Nobel Prizes in Physics, Chemistry and
Physiology/Medicine, plus a quantum-computing branch.

The pipeline combines **BM25 + vector hybrid retrieval**, a **cross-encoder
reranker**, LLM query rewriting and a multi-round **agentic** loop, and answers
with citations. A separate evaluation panel compares pipeline configurations
with **RAGAS** and classic retrieval metrics. Everything is open source and runs
locally; no paid service is required.

## Status

Under construction, milestone by milestone (see the project spec §12).

- **M1 — skeleton**: repo layout, Docker Compose, DB migrations, `/api/health`,
  themed frontend home page with the Ask ITO logo and search bar. ✅
- **M2 — ingest**: seed resolution, Wikipedia fetch, one-step expansion,
  cleaning, snapshot. Corpus: 258 documents (nobel / quantum / ai). ✅
- **M3 — indexing**: section-aware chunking, bge embeddings in pgvector (HNSW),
  BM25 (bm25s). 5,988 chunks; bm25 / vector / hybrid retrieval working. ✅
- **M4 — retrieval pipeline**: all non-agentic modes (closed_book, bm25, vector,
  hybrid, hybrid_rerank, hybrid_rerank_rewrite) with cross-encoder reranking,
  document diversity, step trace, and `POST /api/search`. ✅
- **M5 — agentic layer**: question analysis + sub-questions, multi-round
  hybrid_rerank search with sufficiency checks, citation validation, bounded
  LLM calls, full trace; `agentic` mode wired into `/api/search`. ✅

## Stack

- **DB**: PostgreSQL 16 + pgvector (HNSW)
- **BM25**: `bm25s` (added in M3)
- **Embedding**: `BAAI/bge-base-en-v1.5` (768-dim, config-driven)
- **Reranker**: `BAAI/bge-reranker-base`
- **LLM**: OpenAI-compatible client; default local Ollama, plus a `mock` mode
- **Backend**: Python 3.11+, FastAPI, psycopg 3
- **Frontend**: React + Vite + TypeScript + Tailwind
- **Eval**: RAGAS + Recall@k / MRR / nDCG
- **Orchestration**: Docker Compose

## Quick start (Docker)

```bash
cp .env.example .env     # adjust as needed
make up                  # start db, backend, frontend
make migrate             # apply DB migrations
```

- Frontend: http://localhost:5173
- Backend API + docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

### Local LLM (optional, for answering)

Query/answer generation needs an OpenAI-compatible LLM. By default this is a
local [Ollama](https://ollama.com):

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull <your-model>   # set the name in .env
```

Set `LLM_DEFAULT_MODEL` (and optionally per-role models) in `.env`. For dev or
offline tests, set `LLM_MODE=mock` to use a deterministic stub with no model.

## Running without Docker (dev)

```bash
# Backend
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload            # needs a reachable Postgres + pgvector
python -m app.db.migrate                 # apply migrations

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
| `make ingest` | [M2] Resolve seeds, fetch, expand, clean (snapshot) |
| `make index` | [M3] Chunk, embed, write pgvector + BM25 index |
| `make testset-draft` | [M7] Generate draft eval test set |
| `make eval CONFIGS=...` | [M8] Run evaluation and produce `EVAL_REPORT.md` |
| `make test` | Backend tests + frontend typecheck |
| `make dev` | Print dev run commands |

## Hardware notes

Embedding and reranking run on CPU. Indexing the full corpus (~20–50k chunks)
on CPU takes a while; the embedding model can be switched to
`bge-small-en-v1.5` (`EMBEDDING_DIM=384`) for faster indexing. Sizing estimates
are added as milestones land.

## Architecture decisions

See [`docs/DECISIONS.md`](docs/DECISIONS.md).
