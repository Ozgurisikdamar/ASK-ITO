"""Agentic retrieval loop (spec §6): an explicit, traceable state machine.

analyse -> write queries -> search (hybrid_rerank) -> sufficiency check ->
(loop up to max_rounds) -> answer -> citation validation.

Bounds: total LLM calls <= agent_max_llm_calls (one reserved for the final
answer), wall-clock <= agent_timeout_s. Every step is recorded in the trace as
{step, input, output, duration_ms}.
"""
from __future__ import annotations

import time

from app.agent.answer import generate_answer
from app.agent.prompts import load
from app.agent.schemas import Analysis, Sufficiency, call_json
from app.config import settings
from app.llm.client import LLMClient, get_llm_client
from app.retrieval import hydrate, pipeline, reranker


class _CountingLLM:
    def __init__(self, llm: LLMClient, max_calls: int) -> None:
        self._llm = llm
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


def _analyse(llm: _CountingLLM, query: str, trace: list[dict]) -> Analysis:
    t0 = _now()
    analysis = None
    if llm.remaining >= 2:
        analysis = call_json(llm, load("analyze").replace("{question}", query), Analysis)
    analysis = analysis or Analysis(qtype="simple", subquestions=[])
    subs = [s for s in analysis.subquestions if s.strip()][: settings.agent_max_subquestions]
    analysis.subquestions = subs
    trace.append({
        "step": "analyze", "input": query,
        "output": {"qtype": analysis.qtype, "subquestions": subs},
        "duration_ms": int((_now() - t0) * 1000),
    })
    return analysis


def _write_query(llm: _CountingLLM, subq: str, trace: list[dict]) -> str:
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


def _search(query_text: str, domain: str | None, trace: list[dict]) -> list[tuple[int, float]]:
    t0 = _now()
    res = pipeline.run_retrieval(
        query_text, "hybrid_rerank", domain=domain, final_k=max(settings.final_k, 8)
    )
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
    llm: LLMClient | None = None,
) -> dict:
    final_k = final_k or settings.final_k
    start = _now()
    deadline = start + settings.agent_timeout_s
    counter = _CountingLLM(llm or get_llm_client(), settings.agent_max_llm_calls)
    trace: list[dict] = []
    candidates: dict[int, float] = {}

    def merge(pairs: list[tuple[int, float]]) -> None:
        for cid, score in pairs:
            if cid not in candidates or score > candidates[cid]:
                candidates[cid] = score

    # 1. analyse
    analysis = _analyse(counter, query, trace)
    subquestions = analysis.subquestions or [query]

    # 2 + 3. write query + search for each sub-question
    for subq in subquestions:
        if _now() > deadline:
            break
        search_q = _write_query(counter, subq, trace)
        merge(_search(search_q, domain, trace))

    # 4 + 5. sufficiency loop
    rounds = 1
    while rounds < settings.agent_max_rounds and counter.remaining > 2 and _now() <= deadline:
        t0 = _now()
        suff = call_json(
            counter,
            load("sufficiency").replace("{question}", query)
            .replace("{context}", _context_for_sufficiency(candidates)),
            Sufficiency,
        )
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
        merge(_search(suff.next_query, domain, trace))
        rounds += 1

    # final rerank of the union against the original query -> diversity -> final_k
    t0 = _now()
    cand_ids = list(candidates.keys())
    texts = pipeline._chunk_texts(cand_ids)
    reranked = reranker.rerank(query, [(c, texts.get(c, "")) for c in cand_ids])
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
    t0 = _now()
    answer, citations = generate_answer(
        query, sources, llm=counter, temperature=settings.agent_answer_temperature
    )
    trace.append({
        "step": "answer", "input": query, "output": answer[:300],
        "duration_ms": int((_now() - t0) * 1000),
    })

    from app.retrieval.service import sources_payload

    return {
        "mode": "agentic",
        "answer": answer,
        "citations": citations,
        "sources": sources_payload(sources, query),
        "trace": trace,
        "rewritten_query": None,
        "rounds": rounds,
        "llm_calls": counter.n,
        "ranked_chunk_ids": ranked_ids,
        "timings": {"total": int((_now() - start) * 1000)},
    }
