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

import contextvars
import json
import logging
import queue
import threading
import time
import uuid

import anyio
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.agent.answer import stream_answer
from app.api.errors import generation_limiter, llm_error, needs_generation, safe_error
from app.llm.client import LLMError, get_llm_client
from app.config import settings
from app.llm.metrics import log_request, recording
from app.retrieval import answer_cache, pipeline, service
from app.warmup import activity

router = APIRouter()
DOMAINS = ("all", "nobel", "quantum", "ai")
log = logging.getLogger("askito.stream")


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def produce(q: str, mode: str, domain: str, cancel: threading.Event):
    """Yield (event, data) pairs for one search; `done` carries `metrics`."""
    with recording() as rec:
        for event, data in _produce(q, mode, domain, cancel):
            if event == "done":
                timings = data.get("timings", {})
                metrics = {
                    "mode": mode, "total_ms": timings.get("total"),
                    "first_token_ms": timings.get("first_token"),
                    "retrieval_ms": {s: timings[s] for s in
                                     ("embedding", "bm25", "vector", "rrf", "rerank")
                                     if s in timings},
                    "context_ms": timings.get("context"),
                    "generation_ms": timings.get("answer"),
                    "llm": rec.summary(), "answer_cached": bool(data.get("cached")),
                    # "redis" | "pg" | "fresh" (generated now) | None (no answer)
                    "answer_cache": data.get("answer_cache"),
                    "models": service.model_info(mode) if mode not in ("agentic", "ai") else
                    {"llm": settings.model_for("answer"), "reranker": settings.reranker_model,
                     "embedding": settings.embedding_model},
                }
                data = {**data, "metrics": metrics}
                log_request(f"stream:{mode}", metrics)
            yield event, data


def model_ready(cancel: threading.Event, roles: tuple[str, ...] = ("answer",)):
    """Make sure the local model is loaded before generating, reporting progress.

    A cold load of a large model (Q8_0 ≈ 10 GB) can take minutes. Generation
    requests have a read timeout; if a request times out while Ollama is still
    loading, Ollama aborts the load and every later request starts it again, so
    answers never arrive. The load therefore runs in a shared background thread
    (OllamaClient.begin_load) with its own long timeout, and this generator only
    waits for it, emitting `status` events. Leaving early (cancel) does not
    abort the load."""
    client = get_llm_client()
    if not hasattr(client, "begin_load"):
        return
    for model in dict.fromkeys(settings.model_for(r) for r in roles):
        handle = client.begin_load(model)  # returns at once if the model is resident
        if handle.done.wait(1.0):
            if handle.error is not None:
                raise handle.error
            continue
        while True:
            yield "status", {"stage": "model_loading", "model": model,
                             "elapsed_s": round(handle.elapsed())}
            if handle.done.wait(settings.llm_load_poll_s):
                break
            if cancel.is_set():
                return
        if handle.error is not None:
            raise handle.error
        yield "status", {"stage": "model_loaded", "model": model,
                         "elapsed_s": round(handle.elapsed())}


def _bridge(fn):
    """Run fn(emit) in a thread and yield what it emits; returns fn's result.
    Lets blocking code with callbacks (the agent) stream through a generator."""
    q: queue.Queue = queue.Queue()
    ctx = contextvars.copy_context()  # keeps the request's LLM call recorder

    def target() -> None:
        try:
            q.put(("__result__", ctx.run(fn, lambda ev, data: q.put((ev, data)))))
        except BaseException as exc:  # noqa: BLE001  (re-raised in the consumer)
            q.put(("__error__", exc))

    threading.Thread(target=target, name="askito-agent", daemon=True).start()
    while True:
        ev, data = q.get()
        if ev == "__result__":
            return data
        if ev == "__error__":
            raise data
        yield ev, data


def _produce(q: str, mode: str, domain: str, cancel: threading.Event):
    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    if mode == "agentic":
        yield from _produce_agentic(q, domain, cancel, t0)
        return

    if mode == "ai":  # AI Mode: one grounded answer, or the agent when needed
        from app.agent.router import route

        path, reason = route(q)
        yield "status", {"stage": "route", "route": path, "reason": reason}
        inner = _produce_agentic(q, domain, cancel, t0) if path == "agent" \
            else _produce(q, "overview", domain, cancel)
        for event, data in inner:
            if event == "done":
                data = {**data, "mode": "ai", "route": path, "route_reason": reason}
            yield event, data
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
        n_reranked = cached.counts.get("reranked", 0)
        results = service.results_payload(ranked, q)
        timings["retrieval"] = _ms(t0)
        yield "results", {"results": results, "total": len(results), "stage": "final",
                          "timings": timings, "cached": True, "cache_tier": cached.cache_tier}
    else:
        cand = pipeline.candidates(q, mode, domain)
        timings.update(cand.timings)
        results = service.results_payload(cand.fused, q)
        timings["results"] = _ms(t0)
        yield "results", {"results": results, "total": len(results), "stage": "hybrid",
                          "timings": dict(timings), "cached": False}
        if cancel.is_set():
            return
        ranked, n_reranked = cand.fused, 0
        if pipeline.uses_reranker(mode):
            ranked, step = pipeline.rerank_stage(cand.eff_query, cand.fused, k)
            n_reranked = min(len(cand.fused), settings.rerank_pool)
            timings["rerank"] = step.duration_ms
            results = service.results_payload(ranked, q)
            yield "reranked", {"results": results, "total": len(results),
                               "timings": dict(timings)}
        final = pipeline.apply_diversity(ranked, settings.max_chunks_per_doc, k)
        pipeline.cache_put(q, mode, domain, k, pipeline.PipelineResult(
            mode, final, cand.trace, cand.rewritten, dict(timings),
            ranked=ranked[: pipeline.ranked_keep()],
            counts={"fused": len(cand.fused), "final": len(final), "reranked": n_reranked},
            embedding_source=cand.embedding_source))

    if cancel.is_set():
        return
    tc = time.perf_counter()
    sources = service.context_sources(ranked, mode, n_reranked)
    timings["context"] = _ms(tc)
    yield "sources", {"sources": service.sources_payload(sources, q)}

    if mode == "search":  # ranked sources only: no LLM call
        timings["total"] = _ms(t0)
        yield "done", {"answer": None, "citations": [], "timings": timings, "cached": False,
                       "mode": mode, "model": None,
                       "reranker": service.model_info(mode)["reranker"]}
        return

    hit = answer_cache.get(q, mode, domain, source_ids=[s.chunk_id for s in sources])
    if hit is not None:
        timings["total"] = _ms(t0)
        yield "token", {"t": hit["answer"]}
        yield "done", {"answer": hit["answer"], "citations": hit["citations"],
                       "timings": timings, "cached": True, "mode": mode,
                       "answer_cache": hit.get("cache_kind"),
                       "model": settings.model_for("answer") if sources else None,
                       "reranker": service.model_info(mode)["reranker"],
                       "abstained": not sources}
        return

    if sources:  # the evidence gate may already have answered "not found"
        tl = time.perf_counter()
        yield from model_ready(cancel)
        if cancel.is_set():
            return
        timings["model_load"] = _ms(tl)
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
                           "answer_cache": "fresh",
                           "model": settings.model_for("answer") if sources else None,
                           "reranker": service.model_info(mode)["reranker"],
                           "abstained": not sources}


def _produce_agentic(q: str, domain: str, cancel: threading.Event, t0: float):
    """AI Mode (agent): results first, then each agent step as it happens, the
    numbered sources, the answer token by token, and `done`."""
    cand = pipeline.candidates(q, "hybrid", domain)
    yield "results", {"results": service.results_payload(cand.fused, q),
                      "total": len(cand.fused), "stage": "hybrid",
                      "timings": {"results": _ms(t0)}, "cached": False}
    if cancel.is_set():
        return
    tl = time.perf_counter()
    yield from model_ready(cancel, roles=("agent", "answer"))
    if cancel.is_set():
        return
    load_ms = _ms(tl)
    from app.agent.loop import AgentCancelled, run_agent

    first: dict = {}

    def run(emit):
        def emit_timed(ev, data):
            if ev == "token" and "t" not in first:
                first["t"] = _ms(t0)
            emit(ev, data)

        return run_agent(q, domain=domain, final_k=settings.final_k, cancel=cancel,
                         emit=emit_timed)

    try:
        res = yield from _bridge(run)
    except AgentCancelled:
        return
    timings = {**res["timings"], "total": _ms(t0), "model_load": load_ms}
    if "t" in first:
        timings["first_token"] = first["t"]
    yield "done", {"answer": res["answer"], "citations": res["citations"],
                   "timings": timings, "cached": False,
                   "mode": "agentic", "rounds": res.get("rounds"),
                   "llm_calls": res.get("llm_calls"), "model": settings.model_for("answer")}


async def event_stream(q: str, mode: str, domain: str, rid: str | None = None):
    """SSE body. The pipeline runs in a worker thread feeding a queue; this async
    generator forwards it. When the client disconnects, Starlette cancels this
    generator at its next await (≤ 0.5 s): `cancel` is set, the worker stops
    (agent between steps, LLM streams at once) and the worker itself releases
    the generation slot once the work has really ended."""
    rid = rid or uuid.uuid4().hex[:12]
    cancel = threading.Event()
    items: queue.Queue = queue.Queue()
    holds_slot = needs_generation(mode)

    if holds_slot and not generation_limiter.try_acquire():
        # A previous answer may still be stopping (e.g. the user switched modes):
        # wait briefly instead of failing at once.
        yield _sse("status", {"stage": "queued", "rid": rid})
        deadline = time.monotonic() + settings.generation_wait_s
        while not generation_limiter.try_acquire():
            if time.monotonic() >= deadline:
                yield _sse("error", {**safe_error("busy"), "rid": rid})
                return
            await anyio.sleep(0.25)

    def worker():
        try:
            for event, data in produce(q, mode, domain, cancel):
                if cancel.is_set():
                    break
                if event in ("results", "status", "done") and isinstance(data, dict):
                    data = {**data, "rid": rid}
                items.put((event, data))
        except LLMError as exc:  # classified: tell the user what to do
            items.put(("error", {**llm_error(exc), "rid": rid}))
        except Exception as exc:  # noqa: BLE001  (details are logged, not sent)
            from app.agent.loop import AgentCancelled

            if not isinstance(exc, AgentCancelled):
                items.put(("error", {**safe_error("search_failed", exc), "rid": rid}))
        finally:
            if holds_slot:
                generation_limiter.release()  # only when the work has really ended
            items.put(None)
            if cancel.is_set():
                log.info("rid=%s stream cancelled by the client (mode=%s)", rid, mode)

    activity.enter()
    threading.Thread(target=worker, name="askito-search", daemon=True).start()
    try:
        last = time.monotonic()
        while True:
            try:
                item = await anyio.to_thread.run_sync(items.get, True, 0.5)
            except queue.Empty:
                if time.monotonic() - last >= settings.sse_keepalive_s:
                    last = time.monotonic()
                    yield ": keepalive\n\n"
                continue
            if item is None:
                break
            last = time.monotonic()
            yield _sse(*item)
    finally:
        cancel.set()  # client gone or stream finished: stop the work
        activity.exit()


@router.get("/search/stream")
def search_stream(
    q: str = Query(min_length=1, max_length=500),
    mode: str = Query(default=None),
    domain: str = Query(default="all"),
):
    mode = mode or settings.default_mode
    valid = pipeline.RETRIEVAL_MODES + ["agentic", "ai"]
    if mode not in valid or domain not in DOMAINS:
        def bad():
            yield _sse("error", {"error": "bad_request",
                                 "detail": f"mode must be one of {valid}; domain one of {DOMAINS}"})
        return StreamingResponse(bad(), media_type="text/event-stream")
    return StreamingResponse(
        event_stream(q, mode, domain, uuid.uuid4().hex[:12]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
