"""Retrieval pipeline and run modes (spec §5, §5.1).

Stages are exposed separately (`candidates` -> `rerank_stage` -> `finalize`) so
the streaming endpoint can show hybrid results immediately and re-order them
once the cross-encoder has scored them. `run_retrieval` composes the stages and
memoises whole results in an LRU keyed by query, mode, domain and index version.
Every stage is recorded as a RetrievalStep (name, duration, candidate ids, scores).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

from app.config import settings
from app.index.bm25_index import index_version
from app.retrieval.search import bm25_search, rrf_fuse, vector_search
from app.retrieval.store import store

# "overview" retrieves exactly like hybrid_rerank; only answer generation differs.
RETRIEVAL_MODES = [
    "closed_book", "bm25", "vector", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite",
    "overview",
]
RERANK_MODES = {"hybrid_rerank", "hybrid_rerank_rewrite", "overview"}


@dataclass
class RetrievalStep:
    step: str
    duration_ms: int
    candidate_ids: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    note: str | None = None


def _step(name: str, ms: int, pairs: list[tuple[int, float]], note: str | None = None):
    return RetrievalStep(
        name, ms, [c for c, _ in pairs], [round(float(s), 4) for _, s in pairs], note
    )


@dataclass
class PipelineResult:
    mode: str
    final: list[tuple[int, float]]
    trace: list[RetrievalStep]
    rewritten_query: str | None
    timings: dict[str, int]
    ranked: list[tuple[int, float]] = field(default_factory=list)  # pre-diversity
    cached: bool = False

    def trace_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.trace]


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = int((time.perf_counter() - self.t0) * 1000)


@dataclass
class Candidates:
    fused: list[tuple[int, float]]
    trace: list[RetrievalStep]
    timings: dict[str, int]
    eff_query: str
    rewritten: str | None


def candidates(query: str, mode: str, domain: str | None = None) -> Candidates:
    """Candidate generation: BM25 and/or vector, fused with RRF (no reranking)."""
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown mode: {mode}")
    store.ensure_loaded()
    trace: list[RetrievalStep] = []
    timings: dict[str, int] = {}
    rewritten = None
    eff_query = query

    if mode == "hybrid_rerank_rewrite":
        from app.agent.rewrite import rewrite_query

        with _Timer() as t:
            rewritten = rewrite_query(query)
        eff_query = rewritten or query
        trace.append(RetrievalStep("rewrite", t.ms, note=rewritten))
        timings["rewrite"] = t.ms

    if mode == "closed_book":
        return Candidates([], trace, timings, eff_query, rewritten)

    if mode == "bm25":
        with _Timer() as t:
            fused = bm25_search(eff_query, settings.bm25_top_k, domain=domain)
        trace.append(_step("bm25", t.ms, fused))
        timings["bm25"] = t.ms
    elif mode == "vector":
        with _Timer() as t:
            fused = vector_search(eff_query, settings.vector_top_k, domain=domain)
        trace.append(_step("vector", t.ms, fused))
        timings["vector"] = t.ms
    else:  # hybrid family
        with _Timer() as t:
            bm = bm25_search(eff_query, settings.bm25_top_k, domain=domain)
        trace.append(_step("bm25", t.ms, bm))
        timings["bm25"] = t.ms
        with _Timer() as t:
            vec = vector_search(eff_query, settings.vector_top_k, domain=domain)
        trace.append(_step("vector", t.ms, vec))
        timings["vector"] = t.ms
        with _Timer() as t:
            fused = rrf_fuse([bm, vec])
        trace.append(_step("rrf", t.ms, fused[: settings.results_max]))
        timings["rrf"] = t.ms
    return Candidates(fused, trace, timings, eff_query, rewritten)


def rerank_stage(
    query: str, fused: list[tuple[int, float]], final_k: int
) -> tuple[list[tuple[int, float]], RetrievalStep]:
    """Cross-encoder over the top `rerank_pool` candidates; the rest keep RRF order."""
    from app.retrieval import reranker

    pool = fused[: settings.rerank_pool]
    metas = store.many([c for c, _ in pool])
    with _Timer() as t:
        reranked = reranker.rerank(query, [(c, metas[c].text) for c, _ in pool if c in metas])
    seen = {c for c, _ in reranked}
    tail = [(c, s) for c, s in fused[settings.rerank_pool:] if c not in seen]
    return reranked + tail, _step("rerank", t.ms, reranked[:final_k])


def apply_diversity(
    ranked: list[tuple[int, float]], max_per_doc: int, final_k: int
) -> list[tuple[int, float]]:
    seen: dict[int, int] = {}
    out: list[tuple[int, float]] = []
    for cid, score in ranked:
        meta = store.get(cid)
        doc = meta.doc_id if meta else None
        if doc is not None and seen.get(doc, 0) >= max_per_doc:
            continue
        out.append((cid, score))
        if doc is not None:
            seen[doc] = seen.get(doc, 0) + 1
        if len(out) >= final_k:
            break
    return out


def ranked_keep() -> int:
    """How much of the ranked list to keep: enough that document diversification
    can still fill `results_max` organic results."""
    return settings.results_max * 3


# -- result cache -----------------------------------------------------------
_cache: OrderedDict[tuple, PipelineResult] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query: str, mode: str, domain: str | None, final_k: int) -> tuple:
    return (" ".join(query.lower().split()), mode, domain or "all", final_k, index_version())


def cache_clear() -> None:
    with _cache_lock:
        _cache.clear()


def cache_get(query: str, mode: str, domain: str | None, final_k: int) -> PipelineResult | None:
    if settings.retrieval_cache_size <= 0:
        return None
    key = _cache_key(query, mode, domain, final_k)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        _cache.move_to_end(key)
    return PipelineResult(**{**hit.__dict__, "cached": True})


def cache_put(query: str, mode: str, domain: str | None, final_k: int,
              result: PipelineResult) -> None:
    if settings.retrieval_cache_size <= 0:
        return
    with _cache_lock:
        _cache[_cache_key(query, mode, domain, final_k)] = result
        while len(_cache) > settings.retrieval_cache_size:
            _cache.popitem(last=False)


def run_retrieval(
    query: str, mode: str, domain: str | None = None, final_k: int | None = None,
    use_cache: bool = True,
) -> PipelineResult:
    final_k = final_k or settings.final_k
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown mode: {mode}")
    if use_cache:
        hit = cache_get(query, mode, domain, final_k)
        if hit is not None:
            return hit

    cand = candidates(query, mode, domain)
    trace, timings = list(cand.trace), dict(cand.timings)
    if mode == "closed_book":
        return PipelineResult(mode, [], trace, None, timings)

    if mode in RERANK_MODES:
        ranked, step = rerank_stage(cand.eff_query, cand.fused, final_k)
        trace.append(step)
        timings["rerank"] = step.duration_ms
    else:
        ranked = cand.fused

    final = apply_diversity(ranked, settings.max_chunks_per_doc, final_k)
    result = PipelineResult(
        mode, final, trace, cand.rewritten, timings, ranked=ranked[:ranked_keep()]
    )
    if use_cache:
        cache_put(query, mode, domain, final_k, result)
    return result


# Backwards-compatible helper used by the agent loop.
def _chunk_texts(chunk_ids: list[int]) -> dict[int, str]:
    store.ensure_loaded()
    return {cid: m.text for cid, m in store.many(chunk_ids).items()}
