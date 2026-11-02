"""Retrieval + behaviour metrics (spec §8.2). Deterministic; no LLM needed."""
from __future__ import annotations

import math
import re
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


# --------------------------------------------------------------------------
# Deterministic answer/citation metrics (no LLM). Definitions:
#   raw_citation_validity  share of [n] markers in the model's raw output that
#                          point to an existing source (1..n_sources); the
#                          invalid ones are stripped by validate_citations.
#   citation_coverage      share of answer sentences carrying >= 1 citation.
#   uncited_sentence_rate  1 - citation_coverage (input to the hallucination
#                          audit: uncited sentences are unsupported by design).
#   citation_gold_precision share of cited sources whose document is a gold
#                          document of the question (None without gold docs).
# Not-found answers have no sentences to cite and get None for coverage.
_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE_MARK = re.compile(r"\[(\d+)\]")


def answer_sentences(answer: str) -> list[str]:
    return [s for s in _SENT.split((answer or "").strip()) if len(s.strip(" []0-9.")) > 3]


def citation_metrics(answer: str, raw_citations: list[int] | None, n_sources: int,
                     cited_doc_ids: list[int | None], gold_doc_ids: set[int]) -> dict:
    not_found = refusal_correct(answer)
    out: dict[str, float | bool | None] = {"not_found": not_found}
    raw = raw_citations or []
    out["raw_citation_validity"] = (
        sum(1 for n in raw if 1 <= n <= n_sources) / len(raw) if raw else None)
    sents = [] if not_found else answer_sentences(answer)
    if sents:
        cited = sum(1 for s in sents if _CITE_MARK.search(s))
        out["citation_coverage"] = cited / len(sents)
        out["uncited_sentence_rate"] = 1 - out["citation_coverage"]
    else:
        out["citation_coverage"] = out["uncited_sentence_rate"] = None
    docs = [d for d in cited_doc_ids if d is not None]
    out["citation_gold_precision"] = (
        sum(1 for d in docs if d in gold_doc_ids) / len(docs) if docs and gold_doc_ids else None)
    return out
