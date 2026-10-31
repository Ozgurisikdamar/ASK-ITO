"""Latency + quality benchmark for the retrieval path (and optionally the LLM).

  python scripts/bench.py                         # current config, eval questions
  python scripts/bench.py --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
  python scripts/bench.py --llm                   # also time-to-first-token / total

Reports warm p50/p95 per stage (candidates, rerank, total) and, against the
reviewed eval set, doc/chunk nDCG@10 and Recall@5 of the reranked list.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", help="override RERANKER_MODEL")
    ap.add_argument("--max-length", type=int, help="override RERANK_MAX_LENGTH")
    ap.add_argument("--pool", type=int, help="override RERANK_POOL")
    ap.add_argument("--limit", type=int, default=0, help="max questions (0 = all)")
    ap.add_argument("--llm", action="store_true", help="also time the streamed overview answer")
    args = ap.parse_args()
    if args.reranker:
        settings.reranker_model = args.reranker
    if args.max_length:
        settings.rerank_max_length = args.max_length
    if args.pool:
        settings.rerank_pool = args.pool

    from app.db.connection import get_connection
    from app.eval import metrics
    from app.index import embed
    from app.retrieval import pipeline, reranker
    from app.retrieval.store import store

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, question, gold_chunk_ids, gold_doc_ids FROM eval_questions ORDER BY id"
        ).fetchall()
    if args.limit:
        rows = rows[: args.limit]

    t0 = time.perf_counter()
    store.ensure_loaded()
    embed.warmup()
    reranker.warmup()
    warm_s = time.perf_counter() - t0

    cand_ms, rerank_ms, total_ms = [], [], []
    per_q = []
    for qid, question, gold_c, gold_d in rows:
        t = time.perf_counter()
        cand = pipeline.candidates(question, "hybrid")
        c_ms = (time.perf_counter() - t) * 1000
        ranked, step = pipeline.rerank_stage(question, cand.fused, settings.final_k)
        r_ms = step.duration_ms
        cand_ms.append(c_ms)
        rerank_ms.append(r_ms)
        total_ms.append(c_ms + r_ms)
        ranked_ids = [c for c, _ in ranked[:20]]
        docs, seen = [], set()
        for c in ranked_ids:
            d = store.get(c).doc_id
            if d not in seen:
                seen.add(d)
                docs.append(d)
        per_q.append(metrics.retrieval_metrics(ranked_ids, docs, set(gold_c or []), set(gold_d or [])))

    agg = metrics.aggregate(per_q)
    report = {
        "reranker": settings.reranker_model,
        "rerank_pool": settings.rerank_pool,
        "rerank_max_length": settings.rerank_max_length,
        "questions": len(rows),
        "warmup_s": round(warm_s, 1),
        "candidates_ms": {"p50": round(pct(cand_ms, .5)), "p95": round(pct(cand_ms, .95))},
        "rerank_ms": {"p50": round(pct(rerank_ms, .5)), "p95": round(pct(rerank_ms, .95))},
        "results_total_ms": {"p50": round(pct(total_ms, .5)), "p95": round(pct(total_ms, .95))},
        "doc_ndcg@10": round(agg.get("doc_ndcg@10", 0), 3),
        "doc_recall@5": round(agg.get("doc_recall@5", 0), 3),
        "chunk_ndcg@10": round(agg.get("chunk_ndcg@10", 0), 3),
    }

    if args.llm:
        from app.agent.answer import stream_answer
        from app.retrieval.service import prepare_sources

        ttft, total = [], []
        for _qid, question, _gc, _gd in rows[:5]:
            sources = prepare_sources(question, "overview")
            t = time.perf_counter()
            first = None
            for kind, _payload in stream_answer(question, sources, mode="overview"):
                if kind == "token" and first is None:
                    first = time.perf_counter() - t
            total.append(time.perf_counter() - t)
            ttft.append(first or total[-1])
        report["llm"] = {
            "model": settings.model_for("answer"),
            "ttft_s_median": round(statistics.median(ttft), 1),
            "total_s_median": round(statistics.median(total), 1),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
