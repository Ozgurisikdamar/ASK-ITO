"""End-to-end search service: retrieval pipeline -> hydration -> answer.

`run_search` returns the /api/search JSON schema (spec §9); the streaming
endpoint reuses `results_payload`, `context_sources` and `sources_payload`.
"""
from __future__ import annotations

import time

from app.agent.answer import Source, build_sources, generate_answer
from app.config import settings
from app.llm.client import LLMClient
from app.retrieval import hydrate, pipeline


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


def context_sources(ranked: list[tuple[int, float]], mode: str) -> list[Source]:
    """The numbered passages given to the LLM (top `context_k`, diversified)."""
    final = pipeline.apply_diversity(ranked, settings.max_chunks_per_doc, context_k(mode))
    return build_sources(final, hydrate.fetch_chunks([c for c, _ in final]))


def prepare_sources(query: str, mode: str, domain: str | None = "all") -> list[Source]:
    result = pipeline.run_retrieval(query, mode, domain=domain, final_k=context_k(mode))
    return context_sources(result.ranked or result.final, mode)


def run_search(
    query: str,
    mode: str = "hybrid_rerank",
    domain: str | None = "all",
    final_k: int | None = None,
    llm: LLMClient | None = None,
) -> dict:
    t0 = time.perf_counter()
    k = final_k or context_k(mode)
    result = pipeline.run_retrieval(query, mode, domain=domain, final_k=k)

    infos = hydrate.fetch_chunks([cid for cid, _ in result.final])
    sources = build_sources(result.final, infos)

    ta = time.perf_counter()
    answer, citations = generate_answer(query, sources, llm=llm, mode=mode)
    answer_ms = int((time.perf_counter() - ta) * 1000)

    timings = dict(result.timings)
    timings["answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - t0) * 1000)

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
    }
