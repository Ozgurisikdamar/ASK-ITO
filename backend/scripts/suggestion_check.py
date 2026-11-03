"""Check homepage suggestions against the real retrieval + evidence path (no LLM).

  python scripts/suggestion_check.py [--out PATH]

For each candidate question: run the Answer retrieval (hybrid + bge reranker),
apply the evidence cut-off and report how many passages pass, the top reranker
score and the top documents. A suggestion is "supported" only if passages pass
the cut-off AND the top documents are on topic (judged by reading the printed
titles and passages — article-title matches alone are not enough).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CANDIDATES = [
    "How is the 2025 Physics prize related to quantum computers?",
    "What is the attention mechanism in Transformers?",
    "When is retrieval-augmented generation better than fine-tuning?",
    "What is agentic RAG?",
    "How do AI agents use tools to complete tasks?",
    "What is the difference between large and small language models?",
    "How do neural networks learn with backpropagation?",
    "Why do quantum computers need error correction?",
    "What is a qubit and how is it different from a classical bit?",
    "What did Hopfield and Hinton contribute to machine learning?",
    "How did AlphaFold change protein structure prediction?",
    "What is quantum supremacy and has it been demonstrated?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("LLM_MODE", "mock")
    os.environ["WARMUP_ON_START"] = "false"

    from app.config import settings
    from app.retrieval import hydrate, pipeline, service

    rows = []
    for q in CANDIDATES:
        res = pipeline.run_retrieval(q, "overview", domain="all",
                                     final_k=service.context_k("overview"), use_cache=False)
        ranked = res.ranked or res.final
        n_rr = res.counts.get("reranked", 0)
        passing = service.evidence_filter(ranked, "overview", n_rr)
        sources = service.context_sources(ranked, "overview", n_rr)
        infos = hydrate.fetch_chunks([s.chunk_id for s in sources])
        row = {
            "question": q,
            "top_score": round(float(ranked[0][1]), 4) if ranked else None,
            "passing": len(passing),
            "threshold": settings.fast_min_evidence_score,
            "sources": [{"title": infos[s.chunk_id].doc_title,
                         "section": infos[s.chunk_id].section,
                         "score": round(float(s.score), 4),
                         "text": hydrate.body_without_prefix(infos[s.chunk_id].text)[:220]}
                        for s in sources if s.chunk_id in infos],
        }
        rows.append(row)
        print(f"\n{q}\n  top={row['top_score']} passing={row['passing']}")
        for s in row["sources"]:
            print(f"   - {s['score']:.3f} {s['title']} > {s['section']}: {s['text'][:140]}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
