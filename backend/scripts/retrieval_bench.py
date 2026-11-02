"""Retrieval / reranker benchmark (no LLM): latency per stage + ranking quality.

  python scripts/retrieval_bench.py --label A_hybrid_rerank --mode hybrid_rerank \
      --reranker BAAI/bge-reranker-base --pool 30 --max-length 512 --ef-search 40 --reps 3

One configuration per process (so model load time is measured cold, and
configurations never share RAM/CPU). Results are appended to a resumable JSONL
checkpoint (eval/checkpoints/retrieval_<label>.jsonl); re-running the same
command skips completed (rep, question) units. A summary JSON is written next to
it. Quality metrics are computed exactly as the eval runner does (top-30 ranked
chunks, chunk->doc dedupe, eval.metrics.retrieval_metrics).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RANKED_KEEP = 30  # same convention as the eval runner


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", default="hybrid_rerank",
                    choices=["bm25", "vector", "hybrid", "hybrid_rerank", "search", "overview"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="any Settings env var, e.g. SEARCH_RERANK=true, FAST_TOP_K=20")
    ap.add_argument("--reranker")
    ap.add_argument("--pool", type=int)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--ef-search", type=int)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--questions", help="subset name from eval/subsets.json (default: all)")
    ap.add_argument("--rep-questions", help="subset for repetitions 2..N (default: same "
                    "questions); quality is deterministic, so extra reps only add latency samples")
    return ap.parse_args()


def main() -> None:
    args = parse()
    # config overrides must be in the environment before app.config is imported
    for env, val in (("RERANKER_MODEL", args.reranker), ("RERANK_POOL", args.pool),
                     ("RERANK_MAX_LENGTH", args.max_length), ("HNSW_EF_SEARCH", args.ef_search)):
        if val is not None:
            os.environ[env] = str(val)
    os.environ.setdefault("LLM_MODE", "mock")
    os.environ["WARMUP_ON_START"] = "false"
    overrides = {}
    for kv in args.set:
        key, _, val = kv.partition("=")
        os.environ[key.strip().upper()] = val
        overrides[key.strip().upper()] = val

    import psycopg

    from app.config import settings
    from app.eval import metrics
    from app.index import embed
    from app.retrieval import pipeline
    from app.retrieval.store import store
    from benchkit import CHECKPOINT_DIR, Checkpoint, mean_or_none, summary
    from subsets import load_questions

    ck = Checkpoint(CHECKPOINT_DIR / f"retrieval_{args.label}.jsonl")
    with psycopg.connect(settings.database_url) as conn:
        questions = load_questions(conn, args.questions)
        rep_questions = load_questions(conn, args.rep_questions) if args.rep_questions \
            else questions
    config = {
        "mode": args.mode, "reranker_model": settings.reranker_model,
        "rerank_pool": settings.rerank_pool, "rerank_max_length": settings.rerank_max_length,
        "rerank_batch_size": settings.rerank_batch_size, "hnsw_ef_search": settings.hnsw_ef_search,
        "torch_threads": settings.torch_threads, "final_k": settings.final_k,
        "embedding_model": settings.embedding_model,
        "uses_reranker": pipeline.uses_reranker(args.mode),
        "channel_k": pipeline.channel_k(args.mode), "overrides": overrides,
    }
    planned = len(questions) + (args.reps - 1) * len(rep_questions)
    ck.header(benchmark="retrieval", label=args.label, config=config,
              questions=args.questions or "all", n_questions=len(questions),
              rep_questions=args.rep_questions or args.questions or "all",
              reps=args.reps, planned_units=planned)

    # cold loads, measured separately from warm inference
    loads = {}
    t = time.perf_counter()
    store.ensure_loaded()
    loads["chunk_store_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    embed.warmup()
    loads["embedder_s"] = round(time.perf_counter() - t, 2)
    if pipeline.uses_reranker(args.mode):
        from app.retrieval import reranker

        t = time.perf_counter()
        reranker._model()
        loads["reranker_load_s"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        reranker.warmup()
        loads["reranker_first_call_s"] = round(time.perf_counter() - t, 2)
    pipeline.run_retrieval("warm up query", args.mode, domain="all", use_cache=False)

    doc_of = {cid: m.doc_id for cid, m in store._chunks.items()}
    failed = 0
    for rep in range(1, args.reps + 1):
        for q in (questions if rep == 1 else rep_questions):
            key = f"r{rep}|q{q['id']}"
            if ck.has(key):
                continue
            try:
                t0 = time.perf_counter()
                res = pipeline.run_retrieval(q["question"], args.mode, domain="all",
                                             final_k=settings.final_k, use_cache=False)
                total_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # noqa: BLE001
                failed += 1
                ck.unit(key, status="error", error=f"{exc.__class__.__name__}: {exc}")
                continue
            ranked = [c for c, _ in res.ranked][:RANKED_KEEP]
            docs, seen = [], set()
            for c in ranked:
                d = doc_of.get(c)
                if d is not None and d not in seen:
                    seen.add(d)
                    docs.append(d)
            m = metrics.retrieval_metrics(ranked, docs, q["gold_chunk_ids"], q["gold_doc_ids"])
            n_ce = res.counts.get("reranked", 0) if hasattr(res, "counts") else 0
            ck.unit(key, rep=rep, question_id=q["id"], qtype=q["qtype"],
                    total_ms=round(total_ms, 1), stages=res.timings,
                    candidates=len(res.ranked), metrics=m,
                    top_scores=[round(sc, 5) for _, sc in res.ranked[:min(n_ce, 5)]],
                    gold_in_top5_docs=bool(set(docs[:5]) & q["gold_doc_ids"]))

    units = ck.units()
    rep1 = [u for u in units if u["rep"] == 1]
    keys = ["doc_recall@5", "doc_recall@10", "chunk_recall@5", "doc_mrr@10",
            "doc_ndcg@10", "chunk_mrr@10", "chunk_ndcg@10"]
    quality = {k: mean_or_none([u["metrics"].get(k) for u in rep1]) for k in keys}
    # quality must be identical across repetitions (deterministic pipeline)
    det = all(
        u["metrics"] == next(r["metrics"] for r in rep1 if r["question_id"] == u["question_id"])
        for u in units if any(r["question_id"] == u["question_id"] for r in rep1)
    )
    lat = {"total_ms": summary([u["total_ms"] for u in units])}
    for stage in ("bm25", "vector", "rrf", "rerank"):
        vals = [u["stages"][stage] for u in units if stage in u["stages"]]
        if vals:
            lat[f"{stage}_ms"] = summary(vals)
    per_rep_p50 = {r: summary([u["total_ms"] for u in units if u["rep"] == r]).get("p50")
                   for r in range(1, args.reps + 1)}
    out = {
        "label": args.label, "config": config, "header": ck.headers[-1],
        "cold_loads": loads, "planned": planned, "completed": len(units),
        "failed_this_run": failed,
        "status": "RUN" if len(units) == planned else f"PARTIAL ({len(units)}/{planned})",
        "quality_rep1": quality, "deterministic_across_reps": det,
        "latency": lat, "per_rep_total_p50": per_rep_p50,
    }
    ck.footer(planned, failed=failed, cold_loads=loads)
    (CHECKPOINT_DIR / f"retrieval_{args.label}.summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("label", "status", "cold_loads", "quality_rep1",
                                          "deterministic_across_reps")}, indent=2))
    print(json.dumps(out["latency"], indent=2))


if __name__ == "__main__":
    main()
