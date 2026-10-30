"""End-to-end search service: retrieval pipeline -> hydration -> answer.

Returns a dict matching the /api/search response schema (spec §9). Used by the
API route and by tests.
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
            "chunk_id": s.chunk_id,
            "doc_title": info.doc_title,
            "section": info.section,
            "url": info.url,
            "domain": info.domain,
            "score": round(float(s.score), 6),
            "snippet": hydrate.make_snippet(info.text, query),
        })
    return out


def run_search(
    query: str,
    mode: str = "hybrid_rerank",
    domain: str | None = "all",
    final_k: int | None = None,
    llm: LLMClient | None = None,
) -> dict:
    t0 = time.perf_counter()
    result = pipeline.run_retrieval(query, mode, domain=domain, final_k=final_k or settings.final_k)

    infos = hydrate.fetch_chunks([cid for cid, _ in result.final])
    sources = build_sources(result.final, infos)

    ta = time.perf_counter()
    answer, citations = generate_answer(query, sources, llm=llm)
    answer_ms = int((time.perf_counter() - ta) * 1000)

    sources_out = sources_payload(sources, query)

    timings = dict(result.timings)
    timings["answer"] = answer_ms
    timings["total"] = int((time.perf_counter() - t0) * 1000)

    return {
        "mode": result.mode,
        "answer": answer,
        "citations": citations,
        "sources": sources_out,
        "trace": result.trace_dicts(),
        "rewritten_query": result.rewritten_query,
        "timings": timings,
    }
