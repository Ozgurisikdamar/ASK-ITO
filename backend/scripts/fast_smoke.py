"""Fast RAG smoke run with the real local LLM, through the streaming endpoint code.

  python scripts/fast_smoke.py --label smoke --mode overview
  python scripts/fast_smoke.py --label smoke_search --mode search

Runs `app.api.stream.produce` in-process (exactly what GET /api/search/stream
serves) for a fixed set of representative queries, one at a time, and records
per query: stage timings (embedding, BM25, pgvector, RRF, rerank, context,
time to first token, generation, total), the number of LLM calls by role, the
answer, citations and the cited sources (title, chunk id, URL, snippet) so the
grounding can be checked by hand. Answer caches are bypassed. Results are
appended to eval/checkpoints/smoke_<label>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUERIES = [
    ("topic", "quantum computing"),
    ("broad", "popular AI fields"),
    ("question", "What is machine learning?"),
    ("exact_entity", "Willow processor 105 physical qubits"),
    ("unsupported", "How do you make a sourdough starter at home?"),
    ("multi_chunk", "What did Hopfield and Hinton each contribute to the 2024 Nobel Prize in Physics?"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", default="overview", choices=["overview", "search"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    os.environ["WARMUP_ON_START"] = "false"
    os.environ["PREWARM_FEATURED"] = "false"
    os.environ["ANSWER_CACHE_ENABLED"] = "false"
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")
    for kv in args.set:
        k, _, v = kv.partition("=")
        os.environ[k.strip().upper()] = v

    import httpx

    from app.api import stream as stream_api
    from app.config import settings
    from app.index import embed
    from app.retrieval import pipeline, reranker
    from app.retrieval.store import store
    from benchkit import CHECKPOINT_DIR, Checkpoint, ollama_info

    ck = Checkpoint(CHECKPOINT_DIR / f"smoke_{args.label}.jsonl")
    config = {"mode": args.mode, "model": settings.model_for("answer"),
              "fast_rerank": settings.fast_rerank, "reranker": settings.reranker_model,
              "rerank_pool": settings.rerank_pool, "rerank_max_length": settings.rerank_max_length,
              "fast_top_k": settings.fast_top_k,
              "fast_min_evidence_score": settings.fast_min_evidence_score,
              "context_chunks": settings.overview_context_chunks,
              "context_chars": settings.overview_context_chars,
              "max_tokens": settings.overview_max_tokens, "think": settings.llm_think,
              "num_ctx": settings.llm_num_ctx, "temperature": 0.0}
    ck.header(benchmark="fast_smoke", label=args.label, config=config,
              model=ollama_info(settings.llm_base_url, settings.model_for("answer"))
              if args.mode == "overview" else {}, planned_units=len(QUERIES))

    warm = {}
    t = time.perf_counter()
    store.ensure_loaded()
    embed.warmup()
    if pipeline.uses_reranker(args.mode):
        reranker.warmup()
    warm["retrieval_models_s"] = round(time.perf_counter() - t, 2)
    if args.mode == "overview":
        t = time.perf_counter()
        httpx.post(f"{settings.llm_base_url}/api/generate",
                   json={"model": settings.model_for("answer"), "prompt": "",
                         "keep_alive": settings.llm_keep_alive}, timeout=3600).raise_for_status()
        warm["model_load_s"] = round(time.perf_counter() - t, 2)
    print("warm-up", warm, flush=True)

    for kind, q in QUERIES:
        key = f"{kind}"
        if ck.has(key):
            continue
        pipeline.cache_clear()
        t0 = time.perf_counter()
        events = list(stream_api.produce(q, args.mode, "all", threading.Event()))
        wall = round((time.perf_counter() - t0) * 1000, 1)
        kinds = [e for e, _ in events]
        done = events[-1][1] if kinds and kinds[-1] == "done" else {}
        sources = next((d["sources"] for e, d in events if e == "sources"), [])
        if kinds[-1] == "error":
            ck.unit(key, status="error", query=q, error=events[-1][1])
            continue
        by_n = {s.get("n"): s for s in sources}
        cited = [{"n": c["n"], "chunk_id": c["chunk_id"],
                  "title": by_n.get(c["n"], {}).get("doc_title"),
                  "section": by_n.get(c["n"], {}).get("section"),
                  "url": by_n.get(c["n"], {}).get("url")} for c in done.get("citations", [])]
        llm = done.get("metrics", {}).get("llm", {})
        ck.unit(key, kind=kind, query=q, wall_ms=wall, timings=done.get("timings"),
                events=kinds.count("token"), llm_calls=llm.get("calls"),
                llm_by_role=llm.get("by_role"), prompt_tokens=llm.get("prompt_tokens"),
                completion_tokens=llm.get("completion_tokens"),
                answer=done.get("answer"), abstained=done.get("abstained"),
                citations=cited, models=done.get("metrics", {}).get("models"),
                sources=[{"n": s.get("n"), "chunk_id": s["chunk_id"], "title": s["doc_title"],
                          "section": s.get("section"), "url": s["url"],
                          "snippet": s.get("snippet")} for s in sources])
        print(f"[{args.label}] {key} total={wall/1000:.1f}s "
              f"ttft={(done.get('timings') or {}).get('first_token')} calls={llm.get('calls')}",
              flush=True)
    ck.footer(len(QUERIES), warmup=warm)
    print(json.dumps({"label": args.label, "completed": len(ck.units())}))


if __name__ == "__main__":
    main()
