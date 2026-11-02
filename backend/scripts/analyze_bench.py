"""Summarise benchmark checkpoints into comparable tables (same method for every run).

  python scripts/analyze_bench.py eval B0_qwen3-8b_agentic C0_...   # end-to-end runs
  python scripts/analyze_bench.py retrieval A_hybrid A_hybrid_rerank ...
  python scripts/analyze_bench.py --json out.json eval ...

Reads eval/checkpoints/*.jsonl (never the in-run summaries), recomputes the
deterministic citation metrics with app.eval.metrics for every run (also for
runs made with the unchanged baseline code), and prints markdown tables with
the sample size and RUN / PARTIAL status of every row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fmt(v, pct=False, nd=3):
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load(kind: str, label: str):
    from benchkit import CHECKPOINT_DIR, Checkpoint

    ck = Checkpoint(CHECKPOINT_DIR / f"{kind}_{label}.jsonl")
    header = ck.headers[-1] if ck.headers else {}
    errors = sum(1 for line in ck.path.read_text().splitlines()
                 if '"kind": "unit"' in line and '"status": "error"' in line) \
        if ck.path.exists() else 0
    return header, ck.units(), errors


def eval_row(label: str, conn) -> dict:
    from app.eval import metrics
    from benchkit import mean_or_none, summary
    from subsets import load_probes, load_questions

    header, units, errors = load("eval", label)
    qset = header.get("questions", "all")
    qs = {q["id"]: q for q in (load_probes() if qset == "probes"
                               else load_questions(conn, qset))}
    # only the latest run's question set counts (a label may be resumed with a
    # smaller subset; units of other questions stay in the file but are ignored)
    units = [u for u in units if u["question_id"] in qs]
    planned = header.get("planned_units")
    rep1 = [u for u in units if u["rep"] == 1]
    cites = []
    for u in rep1:
        q = qs.get(u["question_id"])
        if q is None:
            continue
        cites.append(metrics.citation_metrics(
            u["answer"], u.get("raw_citations"), u.get("n_sources") or len(u["source_chunk_ids"]),
            _cited_docs(u), q["gold_doc_ids"]))
    llm_ms = [sum(c.get("ms", 0) for c in u["llm_calls"]) for u in units]
    calls = [len(u["llm_calls"]) for u in units]
    gen = [c["completion_tokens"] / (c["eval_ms"] / 1000) for u in units for c in u["llm_calls"]
           if c.get("completion_tokens") and c.get("eval_ms")]
    pre = [c["prompt_tokens"] / (c["prompt_eval_ms"] / 1000) for u in units
           for c in u["llm_calls"] if c.get("prompt_tokens") and c.get("prompt_eval_ms")]
    roles: dict[str, list[float]] = {}
    for u in units:
        for c in u["llm_calls"]:
            roles.setdefault(c["role"], []).append(c.get("ms", 0))
    retr = [sum(s.get("ms") or 0 for s in u.get("trace_steps", [])
                if s.get("step") in ("search", "final_rerank")) for u in units]
    if header.get("config", {}).get("mode") != "agentic":
        retr = [sum(v for k, v in (u.get("app_timings") or {}).items()
                    if k in ("bm25", "vector", "rrf", "rerank")) for u in units]
    lat = summary([u["total_ms"] for u in units])
    model = header.get("model", {})
    return {
        "label": label, "mode": header.get("config", {}).get("mode"),
        "model": model.get("model"), "quant": model.get("quantization"),
        "params": model.get("parameter_size"), "questions": qset,
        "planned": planned, "completed": len(units), "errors": errors,
        "status": "RUN" if planned and len(units) >= planned else
        f"PARTIAL ({len(units)}/{planned})",
        "started_at": header.get("started_at"), "git_sha": header.get("git_sha"),
        "config": header.get("config"),
        "latency_ms": lat,
        "latency_by_rep_p50": {r: summary([u["total_ms"] for u in units if u["rep"] == r])["p50"]
                               for r in sorted({u["rep"] for u in units})},
        "llm_ms": summary(llm_ms), "retrieval_ms": summary(retr),
        "llm_calls_mean": mean_or_none(calls), "llm_calls_p95": summary(calls).get("p95"),
        "llm_ms_by_role_p50": {r: summary(v)["p50"] for r, v in sorted(roles.items())},
        "gen_tok_s_median": summary(gen).get("p50"), "prefill_tok_s_median": summary(pre).get("p50"),
        "peak_rss_mb": max((u["resources"]["ollama_peak_rss_mb"] for u in units), default=None),
        "cpu_pct_mean": mean_or_none([u["resources"]["ollama_cpu_pct_avg"] for u in units]),
        "n_rep1": len(rep1),
        "quality": {k: mean_or_none([u["metrics"].get(k) for u in rep1]) for k in
                    ("doc_recall@5", "chunk_recall@5", "doc_mrr@10", "doc_ndcg@10")},
        "refusal": mean_or_none([u["metrics"].get("refusal_correct") for u in rep1]),
        "n_unanswerable": sum(1 for u in rep1 if u["metrics"].get("refusal_correct") is not None),
        "citations": {k: mean_or_none([c[k] for c in cites]) for k in
                      ("raw_citation_validity", "citation_coverage", "uncited_sentence_rate",
                       "citation_gold_precision")},
        "not_found_rate": mean_or_none([1.0 if c["not_found"] else 0.0 for c in cites]),
        "context_overflows": sum(1 for u in units for c in u["llm_calls"]
                                 if c.get("context_overflow")),
        "cold_calls": sum(1 for u in units for c in u["llm_calls"]
                          if (c.get("load_ms") or 0) > 1000),
    }


def _cited_docs(u: dict) -> list[int | None]:
    by_chunk = dict(zip(u["source_chunk_ids"], u.get("source_doc_ids") or []))
    return [by_chunk.get(c["chunk_id"]) for c in u.get("citations", [])]


def retrieval_row(label: str) -> dict:
    from benchkit import mean_or_none, summary

    header, units, errors = load("retrieval", label)
    rep1 = [u for u in units if u["rep"] == 1]
    planned = header.get("planned_units")
    out = {
        "label": label, "config": header.get("config"), "planned": planned,
        "completed": len(units), "errors": errors,
        "status": "RUN" if planned and len(units) >= planned else
        f"PARTIAL ({len(units)}/{planned})",
        "n_rep1": len(rep1), "started_at": header.get("started_at"),
        "latency_ms": summary([u["total_ms"] for u in units]),
        "rerank_ms": summary([u["stages"]["rerank"] for u in units if "rerank" in u["stages"]]),
        **{k: mean_or_none([u["metrics"].get(k) for u in rep1]) for k in
           ("doc_recall@5", "chunk_recall@5", "doc_mrr@10", "doc_ndcg@10", "chunk_ndcg@10")},
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("kind", choices=["eval", "retrieval"])
    ap.add_argument("labels", nargs="+")
    args = ap.parse_args()
    rows = []
    if args.kind == "eval":
        import psycopg

        from app.config import settings

        with psycopg.connect(settings.database_url) as conn:
            rows = [eval_row(lb, conn) for lb in args.labels]
        print("| run | status | n | p50 s | p95 s | LLM s p50 | calls mean | gen tok/s | "
              "doc R@5 | MRR@10 | cite cov | cite gold | not-found | refusal |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['label']} | {r['status']} | {r['completed']} | "
                  f"{_fmt(r['latency_ms'].get('p50', 0) / 1000, nd=1)} | "
                  f"{_fmt(r['latency_ms'].get('p95', 0) / 1000, nd=1)} | "
                  f"{_fmt((r['llm_ms'].get('p50') or 0) / 1000, nd=1)} | "
                  f"{_fmt(r['llm_calls_mean'], nd=2)} | {_fmt(r['gen_tok_s_median'], nd=1)} | "
                  f"{_fmt(r['quality']['doc_recall@5'])} | {_fmt(r['quality']['doc_mrr@10'])} | "
                  f"{_fmt(r['citations']['citation_coverage'])} | "
                  f"{_fmt(r['citations']['citation_gold_precision'])} | "
                  f"{_fmt(r['not_found_rate'])} | {_fmt(r['refusal'])} (n={r['n_unanswerable']}) |")
    else:
        rows = [retrieval_row(lb) for lb in args.labels]
        print("| run | status | n | p50 ms | p95 ms | rerank p50 ms | doc R@5 | chunk R@5 | "
              "MRR@10 | nDCG@10 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['label']} | {r['status']} | {r['completed']} | "
                  f"{_fmt(r['latency_ms'].get('p50'), nd=0)} | {_fmt(r['latency_ms'].get('p95'), nd=0)} | "
                  f"{_fmt(r['rerank_ms'].get('p50'), nd=0)} | {_fmt(r['doc_recall@5'])} | "
                  f"{_fmt(r['chunk_recall@5'])} | {_fmt(r['doc_mrr@10'])} | {_fmt(r['doc_ndcg@10'])} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
