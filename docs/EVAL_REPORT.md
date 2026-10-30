# Evaluation Report

Generated 2026-10-30T17:42:10+00:00

Test set: 46 questions. Runs: bm25, vector.

## Ablation

| metric | bm25 | vector |
|---|---|---|
| doc_recall@5 | 0.693 | **0.750** |
| chunk_recall@5 | 0.250 | **0.386** |
| doc_mrr@10 | **0.621** | 0.617 |
| doc_ndcg@10 | **0.642** | 0.628 |
| faithfulness | — | — |
| answer_relevancy | — | — |
| context_precision | — | — |
| context_recall | — | — |

## Operational & behaviour

| mode | latency p50 (ms) | latency p95 (ms) | avg LLM calls | unanswerable refusal |
|---|---|---|---|---|
| bm25 | 30 | 35 | 1.000 | 0.000 |
| vector | 103 | 601 | 1.000 | 0.000 |

> RAGAS note: generation metrics require a working LLM judge. Status per run:

> - bm25: LLM judge unavailable (mock mode)
> - vector: LLM judge unavailable (mock mode)

