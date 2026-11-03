"""Agentic retrieval loop (spec §6): an explicit, traceable state machine.

analyse -> write queries -> search (hybrid_rerank) -> sufficiency check ->
(loop up to max_rounds) -> answer -> citation validation.

Bounds: total LLM calls <= agent_max_llm_calls (one reserved for the final
answer), wall-clock <= agent_timeout_s. Every step is recorded in the trace as
{step, input, output, duration_ms}.
"""
from __future__ import annotations

import time

import threading

from app.agent.answer import generate_answer, stream_answer
from app.agent.prompts import load
from app.agent.schemas import Analysis, Sufficiency, call_json
from app.config import settings
from app.llm.client import LLMClient, LLMError, get_llm_client
from app.llm.metrics import current_recorder, instrumented, llm_role, log_request, recording
from app.retrieval import hydrate, pipeline, reranker


class _CountingLLM:
    askito_instrumented = True  # delegates to an instrumented client

    def __init__(self, llm: LLMClient, max_calls: int) -> None:
        self._llm = instrumented(llm)
        self.n = 0
        self.max = max_calls

    def chat(self, messages, **kw) -> str:
        self.n += 1
        return self._llm.chat(messages, **kw)

    def stream_chat(self, messages, **kw):
        self.n += 1
        yield from self._llm.stream_chat(messages, **kw)

    def ping(self):
        return self._llm.ping()

    @property
    def remaining(self) -> int:
        return self.max - self.n


def _now() -> float:
    return time.perf_counter()


class AgentCancelled(Exception):
    """The client went away; the agent stops between steps."""


# These mean every further LLM call would fail the same way: stop at once.
FATAL_LLM_CODES = ("model_memory", "model_not_found", "model_unreachable",
                   "model_load_timeout")


class _Trace(list):
    """The step trace; with `emit`, every step is also sent to the client as it
    happens (AI Mode shows progress instead of a silent wait)."""

    def __init__(self, emit=None) -> None:
        super().__init__()
        self.emit = emit

    def append(self, item) -> None:
        super().append(item)
        if self.emit is not None:
            self.emit("step", item)


def _helper_failed(exc: LLMError, step: str, trace: list) -> None:
    """A helper call (analyse / sufficiency) failed. Fatal model errors stop the
    agent; anything else (timeout, malformed output) degrades the step."""
    if exc.code in FATAL_LLM_CODES:
        raise exc
    trace.append({"step": "llm_error", "input": step, "output": exc.code, "duration_ms": 0})


def _analyse(llm: _CountingLLM, query: str, trace: list[dict]) -> Analysis:
    t0 = _now()
    analysis = None
    if llm.remaining >= 2:
        try:
            with llm_role("analyze"):
                analysis = call_json(llm, load("analyze").replace("{question}", query),
                                     Analysis, cache_role="analyze")
        except LLMError as exc:  # degrade to a simple single-query analysis
            _helper_failed(exc, "analyze", trace)
    analysis = analysis or Analysis(qtype="simple", subquestions=[])
    subs = [s for s in analysis.subquestions if s.strip()][: settings.agent_max_subquestions]
    analysis.subquestions = subs
    trace.append({
        "step": "analyze", "input": query,
        "output": {"qtype": analysis.qtype, "subquestions": subs},
        "duration_ms": int((_now() - t0) * 1000),
    })
    return analysis


def _skip_rewrite(analysis: Analysis, is_subquestion: bool) -> str | None:
    """Reason to skip the rewrite under AGENT_REWRITE_POLICY (None = rewrite)."""
    policy = settings.agent_rewrite_policy
    if policy == "never":
        return "policy=never"
    if policy == "skip_simple" and analysis.qtype == "simple" and not is_subquestion:
        return "simple question"
    if policy == "skip_subquestions" and is_subquestion:
        return "sub-question already written by the analysis step"
    return None


def _write_query(llm: _CountingLLM, subq: str, trace: list[dict],
                 skip_reason: str | None = None) -> str:
    if skip_reason:
        trace.append({"step": "rewrite_skipped", "input": subq, "output": skip_reason,
                      "duration_ms": 0})
        return subq
    # rewrite only if we can spare a call (keep one for the final answer)
    if llm.remaining <= 2:
        return subq
    from app.agent.rewrite import rewrite_query

    t0 = _now()
    rewritten = rewrite_query(subq, llm=llm)
    if rewritten != subq:
        trace.append({
            "step": "rewrite", "input": subq, "output": rewritten,
            "duration_ms": int((_now() - t0) * 1000),
        })
    return rewritten


def _search(query_text: str, domain: str | None, trace: list[dict],
            scored: dict | None = None) -> list[tuple[int, float]]:
    t0 = _now()
    res = pipeline.run_retrieval(
        query_text, "hybrid_rerank", domain=domain, final_k=max(settings.final_k, 8)
    )
    if scored is not None:  # remember cross-encoder scores of this search's pool
        n = res.counts.get("reranked", 0)
        for cid, score in res.ranked[:n]:
            scored[(query_text, cid)] = score
    for st in res.trace:
        trace.append({
            "step": st.step, "input": query_text,
            "output": st.candidate_ids[:20], "duration_ms": st.duration_ms,
        })
    trace.append({
        "step": "search", "input": query_text,
        "output": [c for c, _ in res.final], "duration_ms": int((_now() - t0) * 1000),
    })
    return res.final


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _dedupe(subquestions: list[str], trace: list[dict]) -> list[str]:
    out, seen = [], set()
    for q in subquestions:
        if _norm(q) in seen:
            trace.append({"step": "subquestion_skipped", "input": q,
                          "output": "duplicate sub-question", "duration_ms": 0})
            continue
        seen.add(_norm(q))
        out.append(q)
    return out


def _context_for_sufficiency(cand: dict[int, float], max_chunks: int = 8) -> str:
    top = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:max_chunks]
    infos = hydrate.fetch_chunks([c for c, _ in top])
    blocks = []
    for i, (cid, _s) in enumerate(top, start=1):
        info = infos.get(cid)
        if info:
            blocks.append(f"[{i}] {info.doc_title}: {hydrate.body_without_prefix(info.text)[:400]}")
    return "\n\n".join(blocks)


def run_agent(
    query: str, domain: str | None = "all", final_k: int | None = None,
    llm: LLMClient | None = None, cancel: threading.Event | None = None, emit=None,
) -> dict:
    """Run the agent; adds a `metrics` object (timings, LLM calls by role,
    rounds, candidate counts) without changing the answer.

    With `emit(event, data)` (AI Mode streaming) every step is emitted as it
    happens, then the numbered sources, then the final answer token by token.
    `cancel` stops the agent between steps and the answer mid-stream
    (AgentCancelled)."""
    outer = current_recorder()
    if outer is not None:
        res = _run_agent(query, domain, final_k, llm, cancel, emit)
        res["metrics"] = _agent_metrics(res, outer)
        return res
    with recording() as rec:
        res = _run_agent(query, domain, final_k, llm, cancel, emit)
    res["metrics"] = _agent_metrics(res, rec)
    log_request("agentic", res["metrics"])
    return res


def _agent_metrics(res: dict, rec) -> dict:
    steps: dict[str, int] = {}
    for st in res["trace"]:
        if st["step"] in ("bm25", "vector", "rrf", "rerank", "search", "final_rerank"):
            steps[st["step"]] = steps.get(st["step"], 0) + (st.get("duration_ms") or 0)
    analysis = next((s["output"] for s in res["trace"] if s["step"] == "analyze"), {})
    return {
        "mode": "agentic",
        "total_ms": res["timings"]["total"],
        "retrieval_ms": steps,
        "llm": rec.summary(),
        "agent": {"rounds": res["rounds"], "llm_calls": res["llm_calls"],
                  "subquestions": len(analysis.get("subquestions") or []),
                  "qtype": analysis.get("qtype"), **res.get("agent_counts", {})},
        "candidates": {"final_rerank_pool": res.get("agent_counts", {}).get("union"),
                       "context_chunks": len(res["sources"])},
    }


def _run_agent(
    query: str, domain: str | None, final_k: int | None, llm: LLMClient | None,
    cancel: threading.Event | None = None, emit=None,
) -> dict:
    final_k = final_k or settings.final_k
    start = _now()
    deadline = start + settings.agent_timeout_s
    counter = _CountingLLM(llm or get_llm_client(), settings.agent_max_llm_calls)
    trace: list[dict] = _Trace(emit)

    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise AgentCancelled()
    candidates: dict[int, float] = {}

    def merge(pairs: list[tuple[int, float]]) -> None:
        for cid, score in pairs:
            if cid not in candidates or score > candidates[cid]:
                candidates[cid] = score

    # 1. analyse
    analysis = _analyse(counter, query, trace)
    check()
    subquestions = analysis.subquestions or [query]

    scored: dict[tuple[str, int], float] = {}  # (query text, chunk) -> CE score
    best_by_sub: list[float] = []

    # Deterministic guards (no LLM involved, every skip is traced): a sub-question
    # or search query that was already handled is not rewritten / searched again,
    # and the sufficiency check is not repeated when the last search added no new
    # passage (same context -> same verdict at temperature 0).
    searched: dict[str, list[tuple[int, float]]] = {}
    subquestions = _dedupe(subquestions, trace)

    # 2 + 3. write query + search for each sub-question
    for subq in subquestions:
        check()
        if _now() > deadline:
            break
        is_sub = bool(analysis.subquestions)
        search_q = _write_query(counter, subq, trace, _skip_rewrite(analysis, is_sub))
        key = _norm(search_q)
        if key in searched:
            trace.append({"step": "search_skipped", "input": search_q,
                          "output": "query already searched", "duration_ms": 0})
            found = searched[key]
        else:
            found = searched[key] = _search(search_q, domain, trace, scored)
        best_by_sub.append(max((s for _, s in found), default=float("-inf")))
        merge(found)

    # 4 + 5. sufficiency loop
    rounds = 1
    max_rounds = settings.agent_max_rounds
    if settings.agent_adaptive_sufficiency and analysis.qtype == "simple":
        max_rounds = min(max_rounds, 1 + settings.agent_max_extra_rounds_simple)
    while rounds < max_rounds and counter.remaining > 2 and _now() <= deadline:
        check()
        if settings.agent_adaptive_sufficiency and best_by_sub and \
                min(best_by_sub) >= settings.agent_coverage_min_score:
            trace.append({
                "step": "sufficiency_skipped", "input": query,
                "output": {"reason": "coverage", "min_top_score": round(min(best_by_sub), 4),
                           "threshold": settings.agent_coverage_min_score},
                "duration_ms": 0,
            })
            break
        t0 = _now()
        try:
            with llm_role("sufficiency"):
                suff = call_json(
                    counter,
                    load("sufficiency").replace("{question}", query)
                    .replace("{context}", _context_for_sufficiency(candidates)),
                    Sufficiency, cache_role="sufficiency",
                )
        except LLMError as exc:  # stop looping, answer from what was found
            _helper_failed(exc, "sufficiency", trace)
            break
        if suff is None:
            break
        trace.append({
            "step": "sufficiency", "input": query,
            "output": {"sufficient": suff.sufficient, "missing": suff.missing,
                       "next_query": suff.next_query},
            "duration_ms": int((_now() - t0) * 1000),
        })
        if suff.sufficient or not suff.next_query:
            break
        if _norm(suff.next_query) in searched:
            trace.append({"step": "sufficiency_stopped", "input": suff.next_query,
                          "output": {"reason": "next query already searched"},
                          "duration_ms": 0})
            break
        before = len(candidates)
        found = searched[_norm(suff.next_query)] = _search(suff.next_query, domain, trace,
                                                           scored)
        best_by_sub = [max((s for _, s in found), default=float("-inf"))]
        merge(found)
        rounds += 1
        if len(candidates) == before:
            trace.append({"step": "sufficiency_stopped", "input": suff.next_query,
                          "output": {"reason": "no new passages from the follow-up search"},
                          "duration_ms": 0})
            break

    # final rerank of the union against the original query -> diversity -> final_k
    t0 = _now()
    cand_ids = list(candidates.keys())
    union_size = len(cand_ids)
    if settings.agent_final_rerank_pool and len(cand_ids) > settings.agent_final_rerank_pool:
        cand_ids = [c for c, _ in sorted(candidates.items(), key=lambda x: x[1],
                                         reverse=True)[: settings.agent_final_rerank_pool]]
    reuse = {c: scored[(query, c)] for c in cand_ids
             if settings.agent_reuse_rerank_scores and (query, c) in scored}
    todo = [c for c in cand_ids if c not in reuse]
    texts = pipeline._chunk_texts(todo)
    fresh = reranker.rerank(query, [(c, texts.get(c, "")) for c in todo]) if todo else []
    reranked = sorted(fresh + list(reuse.items()), key=lambda x: x[1], reverse=True)
    ranked_ids = [c for c, _ in reranked[:30]]  # pre-diversity, for eval
    final = pipeline.apply_diversity(reranked, settings.max_chunks_per_doc, final_k)
    trace.append({
        "step": "final_rerank", "input": query,
        "output": [c for c, _ in final], "duration_ms": int((_now() - t0) * 1000),
    })

    # 6 + 7. answer + citation validation
    from app.agent.answer import build_sources

    infos = hydrate.fetch_chunks([c for c, _ in final])
    sources = build_sources(final, infos)
    from app.retrieval.service import sources_payload

    check()
    t0 = _now()
    if emit is None:
        answer, citations = generate_answer(
            query, sources, llm=counter, temperature=settings.agent_answer_temperature
        )
    else:  # AI Mode: show the sources, then stream the same answer prompt
        emit("sources", {"sources": sources_payload(sources, query)})
        done = None
        for kind, payload in stream_answer(query, sources, llm=counter,
                                           temperature=settings.agent_answer_temperature,
                                           mode="hybrid_rerank", cancel=cancel):
            if kind == "token":
                emit("token", {"t": payload})
            else:
                done = payload
        if done is None:
            raise AgentCancelled()
        answer, citations = done["answer"], done["citations"]
    trace.append({
        "step": "answer", "input": query, "output": answer[:300],
        "duration_ms": int((_now() - t0) * 1000),
    })

    return {
        "mode": "agentic",
        "answer": answer,
        "citations": citations,
        "sources": sources_payload(sources, query),
        "trace": list(trace),
        "rewritten_query": None,
        "rounds": rounds,
        "llm_calls": counter.n,
        "ranked_chunk_ids": ranked_ids,
        "timings": {"total": int((_now() - start) * 1000)},
        "agent_counts": {"union": union_size, "final_rerank_scored": len(todo),
                         "final_rerank_reused": len(reuse), "searches": sum(
                             1 for s in trace if s["step"] == "search")},
    }
