"""Retrieval + behaviour metrics (spec §8.2). Deterministic; no LLM needed."""
from __future__ import annotations

import math
from statistics import median


def recall_at_k(retrieved: list[int], gold: set[int], k: int) -> float | None:
    if not gold:
        return None
    hit = len(set(retrieved[:k]) & gold)
    return hit / len(gold)


def mrr_at_k(retrieved: list[int], gold: set[int], k: int) -> float | None:
    if not gold:
        return None
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[int], gold: set[int], k: int) -> float | None:
    if not gold:
        return None
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, cid in enumerate(retrieved[:k]) if cid in gold
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(
    retrieved_chunks: list[int],
    retrieved_docs: list[int],
    gold_chunks: set[int],
    gold_docs: set[int],
) -> dict:
    out: dict[str, float | None] = {}
    for k in (5, 10):
        out[f"chunk_recall@{k}"] = recall_at_k(retrieved_chunks, gold_chunks, k)
        out[f"doc_recall@{k}"] = recall_at_k(retrieved_docs, gold_docs, k)
    out["chunk_mrr@10"] = mrr_at_k(retrieved_chunks, gold_chunks, 10)
    out["doc_mrr@10"] = mrr_at_k(retrieved_docs, gold_docs, 10)
    out["chunk_ndcg@10"] = ndcg_at_k(retrieved_chunks, gold_chunks, 10)
    out["doc_ndcg@10"] = ndcg_at_k(retrieved_docs, gold_docs, 10)
    return out


def _clean(values: list) -> list[float]:
    return [v for v in values if isinstance(v, (int, float))]


def aggregate(per_question: list[dict]) -> dict:
    """Mean of each metric across questions, ignoring None (non-applicable)."""
    keys: set[str] = set()
    for row in per_question:
        keys.update(row.keys())
    agg = {}
    for key in keys:
        vals = _clean([row.get(key) for row in per_question])
        if vals:
            agg[key] = sum(vals) / len(vals)
    return agg


def percentiles(latencies: list[int]) -> dict:
    if not latencies:
        return {"p50": None, "p95": None}
    s = sorted(latencies)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"p50": int(median(s)), "p95": int(p95)}


def refusal_correct(answer: str) -> bool:
    return "could not find" in (answer or "").lower()
