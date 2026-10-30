# Evaluation Report

Generated 2026-10-30T18:03:49+00:00

Test set: 46 questions. Runs: closed_book, bm25, vector, hybrid, hybrid_rerank, hybrid_rerank_rewrite, agentic.

## Ablation

| metric | closed_book | bm25 | vector | hybrid | hybrid_rerank | hybrid_rerank_rewrite | agentic |
|---|---|---|---|---|---|---|---|
| doc_recall@5 | 0.000 | 0.693 | 0.750 | 0.716 | **0.795** | **0.795** | **0.795** |
| chunk_recall@5 | 0.000 | 0.250 | 0.386 | 0.318 | **0.432** | **0.432** | 0.398 |
| doc_mrr@10 | 0.000 | 0.621 | 0.617 | 0.657 | **0.741** | **0.741** | **0.741** |
| doc_ndcg@10 | 0.000 | 0.642 | 0.628 | 0.676 | **0.747** | **0.747** | 0.728 |
| faithfulness | — | — | — | — | — | — | — |
| answer_relevancy | — | — | — | — | — | — | — |
| context_precision | — | — | — | — | — | — | — |
| context_recall | — | — | — | — | — | — | — |

## Operational & behaviour

| mode | latency p50 (ms) | latency p95 (ms) | avg LLM calls | unanswerable refusal |
|---|---|---|---|---|
| closed_book | 0 | 0 | 0.000 | 0.000 |
| bm25 | 29 | 33 | 1.000 | 0.000 |
| vector | 92 | 126 | 1.000 | 0.000 |
| hybrid | 94 | 122 | 1.000 | 0.000 |
| hybrid_rerank | 8235 | 13521 | 1.000 | 0.000 |
| hybrid_rerank_rewrite | 7987 | 8898 | 1.000 | 0.000 |
| agentic | 10152 | 11688 | 6.000 | 0.000 |

> RAGAS note: generation metrics require a working LLM judge. Status per run:

> - closed_book: LLM judge unavailable (mock mode)
> - bm25: LLM judge unavailable (mock mode)
> - vector: LLM judge unavailable (mock mode)
> - hybrid: LLM judge unavailable (mock mode)
> - hybrid_rerank: LLM judge unavailable (mock mode)
> - hybrid_rerank_rewrite: LLM judge unavailable (mock mode)
> - agentic: LLM judge unavailable (mock mode)

