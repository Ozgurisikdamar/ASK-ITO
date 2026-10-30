"""Retrieval pipeline and run modes (spec §5, §5.1).

Produces the final chunk list plus a step-by-step trace. Each stage is a
RetrievalStep (name, duration, candidate ids) so the trace shows exactly what
ran. The agentic mode (M5) reuses hybrid_rerank through this module.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import psycopg

from app.config import settings
from app.index.embed import embed_query
from app.index import bm25_index
from app.retrieval.search import rrf_fuse
from app.retrieval.vectorstore import PgVectorStore

RETRIEVAL_MODES = [
    "closed_book", "bm25", "vector", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite",
]

_vs = PgVectorStore()
_bm25: bm25_index.Bm25Searcher | None = None


@dataclass
class RetrievalStep:
    step: str
    duration_ms: int
    candidate_ids: list[int] = field(default_factory=list)
    note: str | None = None


@dataclass
class PipelineResult:
    mode: str
    final: list[tuple[int, float]]
    trace: list[RetrievalStep]
    rewritten_query: str | None
    timings: dict[str, int]
    ranked: list[tuple[int, float]] = field(default_factory=list)  # pre-diversity, for eval

    def trace_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.trace]


def _bm25s() -> bm25_index.Bm25Searcher:
    global _bm25
    if _bm25 is None:
        _bm25 = bm25_index.Bm25Searcher()
    return _bm25


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = int((time.perf_counter() - self.t0) * 1000)


def _doc_ids(chunk_ids: list[int]) -> dict[int, int]:
    if not chunk_ids:
        return {}
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (chunk_ids,)
        ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def _chunk_texts(chunk_ids: list[int]) -> dict[int, str]:
    if not chunk_ids:
        return {}
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT id, text FROM chunks WHERE id = ANY(%s)", (chunk_ids,)
        ).fetchall()
    return {int(r[0]): r[1] for r in rows}


def _filter_domain(pairs: list[tuple[int, float]], domain: str | None) -> list[tuple[int, float]]:
    if domain in (None, "all") or not pairs:
        return pairs
    ids = [c for c, _ in pairs]
    with psycopg.connect(settings.database_url) as conn:
        keep = {
            int(r[0])
            for r in conn.execute(
                "SELECT id FROM chunks WHERE id = ANY(%s) AND domain = %s", (ids, domain)
            ).fetchall()
        }
    return [(c, s) for c, s in pairs if c in keep]


def apply_diversity(
    ranked: list[tuple[int, float]], max_per_doc: int, final_k: int
) -> list[tuple[int, float]]:
    doc_of = _doc_ids([c for c, _ in ranked[: max(final_k * 6, 30)]])
    seen: dict[int, int] = {}
    out: list[tuple[int, float]] = []
    for cid, score in ranked:
        doc = doc_of.get(cid)
        if doc is not None and seen.get(doc, 0) >= max_per_doc:
            continue
        out.append((cid, score))
        if doc is not None:
            seen[doc] = seen.get(doc, 0) + 1
        if len(out) >= final_k:
            break
    return out


def run_retrieval(
    query: str, mode: str, domain: str | None = None, final_k: int | None = None
) -> PipelineResult:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown mode: {mode}")
    final_k = final_k or settings.final_k
    trace: list[RetrievalStep] = []
    timings: dict[str, int] = {}
    rewritten = None

    if mode == "closed_book":
        return PipelineResult(mode, [], trace, None, timings)

    eff_query = query
    if mode == "hybrid_rerank_rewrite":
        from app.agent.rewrite import rewrite_query

        with _Timer() as t:
            rewritten = rewrite_query(query)
        eff_query = rewritten or query
        trace.append(RetrievalStep("rewrite", t.ms, [], note=rewritten))
        timings["rewrite"] = t.ms

    # ---- candidate generation ----
    if mode in ("bm25",):
        with _Timer() as t:
            fused = _filter_domain(_bm25s().search(eff_query, settings.bm25_top_k), domain)
        trace.append(RetrievalStep("bm25", t.ms, [c for c, _ in fused[:settings.bm25_top_k]]))
        timings["bm25"] = t.ms
    elif mode in ("vector",):
        with _Timer() as t:
            fused = _vs.search(embed_query(eff_query), settings.vector_top_k, domain=domain)
        trace.append(RetrievalStep("vector", t.ms, [c for c, _ in fused]))
        timings["vector"] = t.ms
    else:  # hybrid family
        with _Timer() as t:
            bm = _filter_domain(_bm25s().search(eff_query, settings.bm25_top_k), domain)
        trace.append(RetrievalStep("bm25", t.ms, [c for c, _ in bm]))
        timings["bm25"] = t.ms
        with _Timer() as t:
            vec = _vs.search(embed_query(eff_query), settings.vector_top_k, domain=domain)
        trace.append(RetrievalStep("vector", t.ms, [c for c, _ in vec]))
        timings["vector"] = t.ms
        with _Timer() as t:
            fused = rrf_fuse([bm, vec])
        trace.append(RetrievalStep("rrf", t.ms, [c for c, _ in fused[:settings.rerank_pool]]))
        timings["rrf"] = t.ms

    # ---- rerank ----
    if mode in ("hybrid_rerank", "hybrid_rerank_rewrite"):
        from app.retrieval import reranker

        pool = fused[: settings.rerank_pool]
        texts = _chunk_texts([c for c, _ in pool])
        with _Timer() as t:
            reranked = reranker.rerank(eff_query, [(c, texts.get(c, "")) for c, _ in pool])
        trace.append(RetrievalStep("rerank", t.ms, [c for c, _ in reranked[:final_k]]))
        timings["rerank"] = t.ms
        ranked = reranked
    else:
        ranked = fused

    final = apply_diversity(ranked, settings.max_chunks_per_doc, final_k)
    return PipelineResult(mode, final, trace, rewritten, timings, ranked=ranked[:30])
