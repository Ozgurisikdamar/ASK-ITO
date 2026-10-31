"""GET /api/search/stream — Server-Sent Events search (spec §9, optional endpoint).

Event sequence (Google-like progressive rendering):
  results   organic results from hybrid retrieval (BM25 + vector + RRF), ~100s of ms
  reranked  the same list re-ordered by the cross-encoder (rerank modes)
  sources   the numbered passages given to the LLM
  step      agent steps (agentic mode)
  token     answer text as it is generated
  done      validated answer + citations + timings
  error     {error, detail}

The pipeline runs in a worker thread feeding a queue; the response loop sends a
keep-alive comment while waiting (the first LLM token can take a while on CPU)
and, when the client disconnects, sets a cancel flag so the LLM request is
closed and generation stops.
"""
from __future__ import annotations

import json
import queue
import threading
import time

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.agent.answer import stream_answer
from app.config import settings
from app.retrieval import answer_cache, pipeline, service
from app.warmup import activity

router = APIRouter()
DOMAINS = ("all", "nobel", "quantum", "ai")


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def produce(q: str, mode: str, domain: str, cancel: threading.Event):
    """Yield (event, data) pairs for one search."""
    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    if mode == "agentic":
        yield from _produce_agentic(q, domain, cancel, t0)
        return

    if mode == "closed_book":
        parts = []
        for kind, payload in stream_answer(q, [], mode="closed_book", cancel=cancel):
            if kind == "token":
                parts.append(payload)
                yield "token", {"t": payload}
            else:
                timings["total"] = _ms(t0)
                yield "done", {**payload, "timings": timings, "cached": False, "mode": mode}
        return

    k = service.context_k(mode)
    cached = pipeline.cache_get(q, mode, domain, k)
    if cached is not None:
        ranked = cached.ranked or cached.final
        results = service.results_payload(ranked, q)
        timings["retrieval"] = _ms(t0)
        yield "results", {"results": results, "total": len(results), "stage": "final",
                          "timings": timings, "cached": True}
    else:
        cand = pipeline.candidates(q, mode, domain)
        timings.update(cand.timings)
        results = service.results_payload(cand.fused, q)
        timings["results"] = _ms(t0)
        yield "results", {"results": results, "total": len(results), "stage": "hybrid",
                          "timings": dict(timings), "cached": False}
        if cancel.is_set():
            return
        ranked = cand.fused
        if mode in pipeline.RERANK_MODES:
            ranked, step = pipeline.rerank_stage(cand.eff_query, cand.fused, k)
            timings["rerank"] = step.duration_ms
            results = service.results_payload(ranked, q)
            yield "reranked", {"results": results, "total": len(results),
                               "timings": dict(timings)}
        final = pipeline.apply_diversity(ranked, settings.max_chunks_per_doc, k)
        pipeline.cache_put(q, mode, domain, k, pipeline.PipelineResult(
            mode, final, cand.trace, cand.rewritten, dict(timings),
            ranked=ranked[: pipeline.ranked_keep()]))

    if cancel.is_set():
        return
    sources = service.context_sources(ranked, mode)
    yield "sources", {"sources": service.sources_payload(sources, q)}

    hit = answer_cache.get(q, mode, domain)
    if hit is not None:
        timings["total"] = _ms(t0)
        yield "token", {"t": hit["answer"]}
        yield "done", {"answer": hit["answer"], "citations": hit["citations"],
                       "timings": timings, "cached": True, "mode": mode,
                       "model": settings.model_for("answer")}
        return

    ta = time.perf_counter()
    first = None
    for kind, payload in stream_answer(q, sources, mode=mode, cancel=cancel):
        if kind == "token":
            if first is None:
                first = _ms(ta)
                timings["first_token"] = first
            yield "token", {"t": payload}
        else:
            timings["answer"] = _ms(ta)
            timings["total"] = _ms(t0)
            answer_cache.put(q, mode, domain, payload["answer"], payload["citations"],
                             [s.chunk_id for s in sources])
            yield "done", {**payload, "timings": timings, "cached": False, "mode": mode,
                           "model": settings.model_for("answer")}


def _produce_agentic(q: str, domain: str, cancel: threading.Event, t0: float):
    # Show results immediately from a hybrid_rerank pass, then run the agent.
    cand = pipeline.candidates(q, "hybrid", domain)
    yield "results", {"results": service.results_payload(cand.fused, q),
                      "total": len(cand.fused), "stage": "hybrid",
                      "timings": {"results": _ms(t0)}, "cached": False}
    if cancel.is_set():
        return
    from app.agent.loop import run_agent

    res = run_agent(q, domain=domain, final_k=settings.final_k)
    for step in res["trace"]:
        yield "step", step
    yield "sources", {"sources": res["sources"]}
    yield "token", {"t": res["answer"]}
    yield "done", {"answer": res["answer"], "citations": res["citations"],
                   "timings": {**res["timings"], "total": _ms(t0)}, "cached": False,
                   "mode": "agentic", "rounds": res.get("rounds"),
                   "llm_calls": res.get("llm_calls"), "model": settings.model_for("answer")}


def event_stream(q: str, mode: str, domain: str):
    cancel = threading.Event()
    items: queue.Queue = queue.Queue()

    def worker():
        try:
            for event, data in produce(q, mode, domain, cancel):
                if cancel.is_set():
                    break
                items.put((event, data))
        except Exception as exc:  # noqa: BLE001
            items.put(("error", {"error": "search_failed", "detail": str(exc)}))
        finally:
            items.put(None)

    activity.enter()
    threading.Thread(target=worker, name="askito-search", daemon=True).start()
    try:
        while True:
            try:
                item = items.get(timeout=settings.sse_keepalive_s)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            yield _sse(*item)
    finally:
        cancel.set()  # client gone or stream finished: stop the LLM
        activity.exit()


@router.get("/search/stream")
def search_stream(
    q: str = Query(min_length=1, max_length=500),
    mode: str = Query(default=None),
    domain: str = Query(default="all"),
):
    mode = mode or settings.default_mode
    valid = pipeline.RETRIEVAL_MODES + ["agentic"]
    if mode not in valid or domain not in DOMAINS:
        def bad():
            yield _sse("error", {"error": "bad_request",
                                 "detail": f"mode must be one of {valid}; domain one of {DOMAINS}"})
        return StreamingResponse(bad(), media_type="text/event-stream")
    return StreamingResponse(
        event_stream(q, mode, domain),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
