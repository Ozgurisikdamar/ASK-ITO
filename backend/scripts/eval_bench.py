"""End-to-end search benchmark with a real local LLM (resumable, sequential).

  python scripts/eval_bench.py --label B_qwen3-8b_agentic --mode agentic \
      --model qwen3:8b --questions all --reps 1
  python scripts/eval_bench.py --label B_qwen3-8b_agentic_lat --mode agentic \
      --model qwen3:8b --questions lat12 --reps 3

Runs the application's own search functions (`agent.loop.run_agent`,
`retrieval.service.run_search`) unchanged, one question at a time. The LLM
client is wrapped in a transparent proxy that only measures each call (wall
time, Ollama token counts and load time, role inferred from the prompt
template); the request and the returned text are passed through untouched.

Each (rep, question) result is appended to eval/checkpoints/eval_<label>.jsonl
as soon as it finishes; re-running the same command resumes. A summary JSON is
written at the end (also on partial runs, with status PARTIAL n/N).

Config overrides: --set KEY=VALUE (any Settings env var), applied before import.
Retrieval caches are cleared before every question so repetitions are real.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RANKED_KEEP = 30
CITE_RE = re.compile(r"\[(\d+)\]")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["agentic", "hybrid_rerank", "overview", "hybrid", "closed_book"])
    ap.add_argument("--model", help="sets LLM_DEFAULT_MODEL (roles fall back to it)")
    ap.add_argument("--questions", default="all", help="all | subset name in eval/subsets.json")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--max-questions", type=int, default=0, help="debug: stop after N units")
    return ap.parse_args()


# ---------------------------------------------------------------- measurement
ROLE_SIGNATURES = {
    "analyze": "Analyse the user's question",
    "rewrite": "Rewrite the user's question",
    "sufficiency": "Decide whether the collected context",
    "overview": 'writing a short "AI overview"',
    "answer": "You are Ask ITO, a question-answering assistant",
    "closed_book": "Answer the following question concisely",
}


def infer_role(messages: list[dict]) -> str:
    text = messages[-1].get("content", "") if messages else ""
    for role, sig in ROLE_SIGNATURES.items():
        if sig in text[:400]:
            return role
    return "other"


class ResourceSampler:
    """Samples RSS and CPU time of all ollama processes once per second."""

    def __init__(self) -> None:
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._cpu0 = self._cpu_ticks()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _pids() -> list[int]:
        out = []
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    if (p / "comm").read_text().strip().startswith("ollama"):
                        out.append(int(p.name))
                except OSError:
                    pass
        return out

    def _rss_mb(self) -> float:
        total = 0
        for pid in self._pids():
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS"):
                        total += int(line.split()[1])
            except OSError:
                pass
        return total / 1024

    def _cpu_ticks(self) -> int:
        total = 0
        for pid in self._pids():
            try:
                parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                total += int(parts[11]) + int(parts[12])  # utime + stime
            except (OSError, IndexError, ValueError):
                pass
        return total

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())

    def start(self) -> "ResourceSampler":
        self.peak_rss_mb = self._rss_mb()
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        wall = time.monotonic() - self._t0
        ticks = self._cpu_ticks() - self._cpu0
        hz = os.sysconf("SC_CLK_TCK")
        return {"ollama_peak_rss_mb": round(self.peak_rss_mb, 1),
                "ollama_cpu_pct_avg": round(100 * ticks / hz / wall, 1) if wall > 0 else None}


class MeasuringLLM:
    """Pass-through proxy that records one entry per LLM call."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict] = []
        self._last_raw: dict | None = None
        # Old (uninstrumented) clients: capture the raw provider response. Newer
        # instrumented clients expose the same data as `last_usage`.
        post = None
        if not getattr(type(inner), "askito_instrumented", False):
            post = getattr(inner, "_post", None)
        if post is not None:
            def capture(url, body, _post=post):
                data = _post(url, body)
                self._last_raw = data
                return data
            inner._post = capture

    def reset(self) -> None:
        self.calls = []

    def ping(self):
        return self.inner.ping()

    def chat(self, messages, **kw):
        role = infer_role(messages)
        self._last_raw = None
        t0 = time.perf_counter()
        rec = {"role": role, "model": kw.get("model") or "", "json_mode": bool(kw.get("json_mode")),
               "max_tokens": kw.get("max_tokens")}
        try:
            out = self.inner.chat(messages, **kw)
        except Exception as exc:  # noqa: BLE001
            rec.update(ms=round((time.perf_counter() - t0) * 1000, 1),
                       error=f"{exc.__class__.__name__}: {str(exc)[:200]}")
            self.calls.append(rec)
            raise
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        raw = self._last_raw or {}
        usage = getattr(self.inner, "last_usage", None) or {}
        if not raw and usage.get("prompt_tokens") is not None:  # instrumented client
            rec.update({k: usage.get(k) for k in ("prompt_tokens", "completion_tokens",
                                                   "prompt_eval_ms", "eval_ms", "load_ms",
                                                   "done_reason", "context_overflow")})
            rec["model"] = usage.get("model") or rec["model"]
        elif "prompt_eval_count" in raw or "eval_count" in raw:  # Ollama native
            rec.update(
                prompt_tokens=raw.get("prompt_eval_count"),
                completion_tokens=raw.get("eval_count"),
                prompt_eval_ms=round(raw.get("prompt_eval_duration", 0) / 1e6, 1),
                eval_ms=round(raw.get("eval_duration", 0) / 1e6, 1),
                load_ms=round(raw.get("load_duration", 0) / 1e6, 1),
                done_reason=raw.get("done_reason"),
                model=raw.get("model") or rec["model"],
            )
        else:
            rec["tokens"] = "unknown"
        if role in ("answer", "overview"):
            rec["raw_citations"] = [int(n) for n in CITE_RE.findall(out or "")]
            rec["output_chars"] = len(out or "")
        self.calls.append(rec)
        return out

    def stream_chat(self, messages, **kw):
        yield from self.inner.stream_chat(messages, **kw)


# --------------------------------------------------------------------- main
def main() -> None:
    args = parse()
    os.environ["WARMUP_ON_START"] = "false"
    os.environ["PREWARM_FEATURED"] = "false"
    os.environ["ANSWER_CACHE_ENABLED"] = "false"
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")
    if args.model:
        os.environ["LLM_DEFAULT_MODEL"] = args.model
    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        os.environ[k.strip().upper()] = v
        overrides[k.strip().upper()] = v

    import psycopg

    from app.config import settings
    from app.eval import metrics
    from app.llm.client import get_llm_client
    from app.retrieval import pipeline
    from app.retrieval.store import store
    from benchkit import CHECKPOINT_DIR, Checkpoint, mean_or_none, ollama_info, summary
    from subsets import load_questions

    ck = Checkpoint(CHECKPOINT_DIR / f"eval_{args.label}.jsonl")
    with psycopg.connect(settings.database_url) as conn:
        questions = load_questions(conn, args.questions)
    store.ensure_loaded()
    doc_of = {cid: m.doc_id for cid, m in store._chunks.items()}

    llm = MeasuringLLM(get_llm_client())
    ok, ping_msg = llm.ping()
    if settings.llm_mode != "mock" and not ok:
        raise SystemExit(f"LLM not reachable: {ping_msg}")
    model_meta = ollama_info(settings.llm_base_url, settings.model_for("answer"))
    config = {
        "mode": args.mode, "llm_mode": settings.llm_mode,
        "models": {r: settings.model_for(r) for r in ("agent", "answer", "judge")},
        "llm_think": settings.llm_think, "llm_num_ctx": settings.llm_num_ctx,
        "llm_num_thread": settings.llm_num_thread or "ollama default",
        "llm_keep_alive": settings.llm_keep_alive, "llm_timeout_s": settings.llm_timeout_s,
        "reranker_model": settings.reranker_model, "rerank_pool": settings.rerank_pool,
        "rerank_max_length": settings.rerank_max_length, "hnsw_ef_search": settings.hnsw_ef_search,
        "final_k": settings.final_k, "agent_max_rounds": settings.agent_max_rounds,
        "agent_max_llm_calls": settings.agent_max_llm_calls,
        "agent_timeout_s": settings.agent_timeout_s, "answer_context_chars":
            getattr(settings, "answer_context_chars", None),
        "overrides": overrides,
    }
    planned = args.reps * len(questions)
    ck.header(benchmark="eval", label=args.label, config=config, model=model_meta,
              questions=args.questions, n_questions=len(questions), reps=args.reps,
              planned_units=planned)

    # warm-up (not recorded): load reranker/embedder and the LLM
    t = time.perf_counter()
    from app.index import embed
    from app.retrieval import reranker

    embed.warmup()
    reranker.warmup()
    warm = {"retrieval_models_s": round(time.perf_counter() - t, 2)}
    if settings.llm_mode != "mock":
        t = time.perf_counter()
        llm.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=4,
                 model=settings.model_for("answer"))
        warm["llm_first_call_s"] = round(time.perf_counter() - t, 2)
        warm["llm_first_call_load_ms"] = llm.calls[-1].get("load_ms")
    llm.reset()

    def answer(q: dict) -> dict:
        if args.mode == "agentic":
            from app.agent.loop import run_agent

            return run_agent(q["question"], domain="all", final_k=settings.final_k, llm=llm)
        from app.retrieval.service import run_search

        r = run_search(q["question"], mode=args.mode, domain="all", llm=llm)
        r["llm_calls"] = len(llm.calls)
        return r

    failed, done_now = 0, 0
    for rep in range(1, args.reps + 1):
        for q in questions:
            key = f"r{rep}|q{q['id']}"
            if ck.has(key):
                continue
            if args.max_questions and done_now >= args.max_questions:
                break
            pipeline.cache_clear()
            llm.reset()
            sampler = ResourceSampler().start()
            t0 = time.perf_counter()
            try:
                resp = answer(q)
            except Exception as exc:  # noqa: BLE001
                res = sampler.stop()
                failed += 1
                ck.unit(key, status="error", rep=rep, question_id=q["id"],
                        error=f"{exc.__class__.__name__}: {str(exc)[:300]}",
                        ms=round((time.perf_counter() - t0) * 1000, 1), llm_calls=llm.calls,
                        resources=res)
                print(f"[{args.label}] {key} ERROR {exc.__class__.__name__}", flush=True)
                continue
            total_ms = (time.perf_counter() - t0) * 1000
            res = sampler.stop()
            ranked = list(resp.get("ranked_chunk_ids") or [])[:RANKED_KEEP]
            docs, seen = [], set()
            for c in ranked:
                d = doc_of.get(c)
                if d is not None and d not in seen:
                    seen.add(d)
                    docs.append(d)
            m = metrics.retrieval_metrics(ranked, docs, q["gold_chunk_ids"], q["gold_doc_ids"])
            if q["qtype"] == "unanswerable":
                m["refusal_correct"] = float(metrics.refusal_correct(resp["answer"]))
            sources = [s["chunk_id"] for s in resp.get("sources", [])]
            cited = [doc_of.get(c["chunk_id"]) for c in resp.get("citations", [])]
            raw = next((c.get("raw_citations") for c in reversed(llm.calls)
                        if c["role"] in ("answer", "overview")), None)
            if hasattr(metrics, "citation_metrics"):
                am = metrics.citation_metrics(resp["answer"], raw, len(sources), cited,
                                              q["gold_doc_ids"])
            else:  # baseline code without the metric: computed later from stored fields
                am = None
            ck.unit(
                key, rep=rep, question_id=q["id"], qtype=q["qtype"], domain=q["domain"],
                total_ms=round(total_ms, 1), app_timings=resp.get("timings", {}),
                app_llm_calls=resp.get("llm_calls"), rounds=resp.get("rounds"),
                llm_calls=llm.calls, resources=res,
                trace_steps=[{"step": s.get("step"), "ms": s.get("duration_ms")}
                             for s in resp.get("trace", [])],
                subquestions=next((s.get("output", {}).get("subquestions")
                                   for s in resp.get("trace", []) if s.get("step") == "analyze"),
                                  None),
                answer=resp["answer"], citations=resp.get("citations", []),
                source_chunk_ids=sources, source_doc_ids=[doc_of.get(c) for c in sources],
                ranked_chunk_ids=ranked, metrics=m, answer_metrics=am, raw_citations=raw,
                n_sources=len(sources),
            )
            done_now += 1
            llm_ms = sum(c.get("ms", 0) for c in llm.calls)
            print(f"[{args.label}] {key} {total_ms/1000:.1f}s llm={llm_ms/1000:.1f}s "
                  f"calls={len(llm.calls)}", flush=True)

    units = ck.units()
    rep1 = [u for u in units if u["rep"] == 1]
    qkeys = ["doc_recall@5", "chunk_recall@5", "doc_mrr@10", "doc_ndcg@10", "chunk_ndcg@10",
             "refusal_correct"]
    calls_per_q = [len(u["llm_calls"]) for u in units]
    by_role: dict[str, list[float]] = {}
    tokens = {"prompt": [], "completion": [], "tok_s": [], "prefill_tok_s": []}
    for u in units:
        for c in u["llm_calls"]:
            by_role.setdefault(c["role"], []).append(c.get("ms", 0))
            if c.get("completion_tokens") and c.get("eval_ms"):
                tokens["completion"].append(c["completion_tokens"])
                tokens["tok_s"].append(round(c["completion_tokens"] / (c["eval_ms"] / 1000), 2))
            if c.get("prompt_tokens") and c.get("prompt_eval_ms"):
                tokens["prompt"].append(c["prompt_tokens"])
                tokens["prefill_tok_s"].append(
                    round(c["prompt_tokens"] / (c["prompt_eval_ms"] / 1000), 2))
    errors = sum(1 for line in ck.path.read_text().splitlines()
                 if '"status": "error"' in line)
    out = {
        "label": args.label, "status": "RUN" if len(units) == planned
        else f"PARTIAL ({len(units)}/{planned})",
        "planned": planned, "completed": len(units), "error_records": errors,
        "config": config, "model": model_meta, "warmup": warm,
        "header": ck.headers[-1],
        "latency_total_ms": summary([u["total_ms"] for u in units]),
        "per_rep_total_p50_ms": {r: summary([u["total_ms"] for u in units if u["rep"] == r]).get("p50")
                                 for r in range(1, args.reps + 1)},
        "llm_total_ms_per_q": summary([sum(c.get("ms", 0) for c in u["llm_calls"]) for u in units]),
        "llm_calls_per_q": {"mean": mean_or_none(calls_per_q),
                            "p95": summary(calls_per_q).get("p95")},
        "llm_ms_by_role": {r: summary(v) for r, v in by_role.items()},
        "tokens": {k: summary(v) for k, v in tokens.items()},
        "resources": {
            "peak_rss_mb_max": max((u["resources"]["ollama_peak_rss_mb"] for u in units), default=None),
            "cpu_pct_mean": mean_or_none([u["resources"]["ollama_cpu_pct_avg"] for u in units]),
        },
        "quality_rep1": {k: mean_or_none([u["metrics"].get(k) for u in rep1]) for k in qkeys},
        "n_rep1": len(rep1),
    }
    ck.footer(planned, failed=failed)
    (CHECKPOINT_DIR / f"eval_{args.label}.summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("label", "status", "latency_total_ms",
                                          "llm_calls_per_q", "quality_rep1")}, indent=2))


if __name__ == "__main__":
    main()
