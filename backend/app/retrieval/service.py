"""End-to-end search service: retrieval pipeline -> hydration -> answer.

`run_search` returns the /api/search JSON schema (spec §9); the streaming
endpoint reuses `results_payload`, `context_sources` and `sources_payload`.
"""
from __future__ import annotations

import logging
import time

from app.agent.answer import Source, build_sources, generate_answer
from app.config import settings
from app.llm.client import LLMClient
from app.llm.metrics import log_request, recording
from app.retrieval import hydrate, pipeline

log = logging.getLogger("askito.search")


def sources_payload(sources: list[Source], query: str) -> list[dict]:
    out = []
    for s in sources:
        info = s.info
        out.append({
            "n": s.n,
            "chunk_id": s.chunk_id,
            "doc_id": info.doc_id,
            "doc_title": info.doc_title,
            "section": info.section,
            "url": info.url,
            "domain": info.domain,
            "score": round(float(s.score), 6),
            "snippet": hydrate.make_snippet(info.text, query),
        })
    return out


def results_payload(ranked: list[tuple[int, float]], query: str) -> list[dict]:
    """Organic results list (Google-style), document-diversified, best first."""
    shown = pipeline.apply_diversity(ranked, settings.max_chunks_per_doc, settings.results_max)
    infos = hydrate.fetch_chunks([c for c, _ in shown])
    out = []
    for cid, score in shown:
        info = infos.get(cid)
        if info is None:
            continue
        out.append({
            "chunk_id": cid,
            "doc_id": info.doc_id,
            "doc_title": info.doc_title,
            "section": info.section,
            "url": info.url,
            "domain": info.domain,
            "score": round(float(score), 6),
            "snippet": hydrate.make_snippet(info.text, query),
        })
    return out


def context_k(mode: str) -> int:
    return settings.overview_context_chunks if mode == "overview" else settings.final_k


def evidence_cutoff_active(mode: str) -> bool:
    """The cut-off applies in the fast modes when they rerank with the model it
    was calibrated for (EVIDENCE_SCORE_MODEL) and FAST_MIN_EVIDENCE_SCORE > 0."""
    return (settings.fast_min_evidence_score > 0 and mode in pipeline.FAST_MODES
            and pipeline.uses_reranker(mode)
            and settings.reranker_model == settings.evidence_score_model)


_warned_uncalibrated = False


def evidence_filter(ranked: list[tuple[int, float]], mode: str,
                    n_reranked: int) -> list[tuple[int, float]]:
    """Weak out-of-domain gate (a heuristic, not calibrated confidence): keep
    only cross-encoder-scored passages at or above FAST_MIN_EVIDENCE_SCORE. An
    empty result means "not found" without an LLM call. It stops clearly
    off-topic queries; questions that share vocabulary with the corpus usually
    pass and rely on the prompt's not-found instruction."""
    global _warned_uncalibrated
    if not evidence_cutoff_active(mode):
        if (settings.fast_min_evidence_score > 0 and pipeline.uses_reranker(mode)
                and mode in pipeline.FAST_MODES and not _warned_uncalibrated):
            _warned_uncalibrated = True
            log.warning("evidence cut-off disabled: calibrated for %s, reranker is %s",
                        settings.evidence_score_model, settings.reranker_model)
        return ranked
    return [(c, s) for c, s in ranked[:n_reranked] if s >= settings.fast_min_evidence_score]


def context_sources(ranked: list[tuple[int, float]], mode: str,
                    n_reranked: int | None = None) -> list[Source]:
    """The numbered passages given to the LLM (evidence filter in the fast modes,
    then top `context_k`, at most MAX_CHUNKS_PER_DOC per document)."""
    if n_reranked is None:
        n_reranked = settings.rerank_pool
    ranked = evidence_filter(ranked, mode, n_reranked)
    final = pipeline.apply_diversity(ranked, settings.max_chunks_per_doc, context_k(mode))
    return build_sources(final, hydrate.fetch_chunks([c for c, _ in final]))


def prepare_sources(query: str, mode: str, domain: str | None = "all") -> list[Source]:
    result = pipeline.run_retrieval(query, mode, domain=domain, final_k=context_k(mode))
    return context_sources(result.ranked or result.final, mode,
                           result.counts.get("reranked", 0))


def run_search(
    query: str,
    mode: str = "hybrid_rerank",
    domain: str | None = "all",
    final_k: int | None = None,
    llm: LLMClient | None = None,
) -> dict:
    t0 = time.perf_counter()
    k = final_k or context_k(mode)
    with recording() as rec:
        result = pipeline.run_retrieval(query, mode, domain=domain, final_k=k)

        tc = time.perf_counter()
        if mode in pipeline.FAST_MODES:
            sources = context_sources(result.ranked or result.final, mode,
                                      result.counts.get("reranked", 0))
        else:
            infos = hydrate.fetch_chunks([cid for cid, _ in result.final])
            sources = build_sources(result.final, infos)
        context_ms = int((time.perf_counter() - tc) * 1000)

        ta = time.perf_counter()
        if mode == "search":  # ranked sources only, no LLM call
            answer, citations = "", []
        else:
            answer, citations = generate_answer(query, sources, llm=llm, mode=mode)
        answer_ms = int((time.perf_counter() - ta) * 1000)

    timings = dict(result.timings)
    timings["context"] = context_ms
    timings["answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - t0) * 1000)
    metrics = request_metrics(mode, timings, rec, result.counts, len(sources),
                              cached=result.cached)
    log_request(mode, metrics)

    return {
        "mode": result.mode,
        "answer": answer,
        "citations": citations,
        "sources": sources_payload(sources, query),
        "results": results_payload(result.ranked or result.final, query)[:10],
        "trace": result.trace_dicts(),
        "rewritten_query": result.rewritten_query,
        "ranked_chunk_ids": [c for c, _ in result.ranked],
        "timings": timings,
        "metrics": metrics,
    }


def request_metrics(mode: str, timings: dict, rec, counts: dict, context_chunks: int,
                    cached: bool = False) -> dict:
    """Per-request observability object (timings, LLM calls, candidate counts)."""
    stages = ("embedding", "bm25", "vector", "rrf", "rerank", "rewrite")
    return {
        "mode": mode,
        "total_ms": timings.get("total"),
        "retrieval_ms": {s: timings[s] for s in stages if s in timings},
        "context_ms": timings.get("context"),
        "answer_ms": timings.get("answer"),
        "llm": rec.summary(),
        "candidates": {**counts, "context_chunks": context_chunks},
        "retrieval_cached": cached,
        "models": model_info(mode),
        "evidence": evidence_info(mode, context_chunks),
    }


def model_info(mode: str) -> dict:
    """Which models served this request (names only)."""
    return {
        "llm": None if mode == "search" else settings.model_for("answer"),
        "reranker": settings.reranker_model if pipeline.uses_reranker(mode) else None,
        "embedding": settings.embedding_model,
    }


def evidence_info(mode: str, context_chunks: int) -> dict:
    active = evidence_cutoff_active(mode)
    return {"min_score": settings.fast_min_evidence_score if active else None,
            "context_chunks": context_chunks,
            "abstained": mode != "search" and context_chunks == 0}
