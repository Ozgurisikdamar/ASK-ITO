"""Evidence cut-off calibration report (no LLM).

  python scripts/evidence_calibration.py [--thresholds 0.1,0.3,0.5]

For each question in eval/evidence_calibration.json runs the Answer-mode
retrieval (hybrid + reranker) and records the top reranker score and whether
the evidence cut-off would let the question reach the LLM. Prints false-reject
(answerable blocked) and pass-through rates per group and threshold, and writes
eval/checkpoints/evidence_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default="0.05,0.1,0.3,0.5,0.7")
    args = ap.parse_args()
    os.environ.setdefault("LLM_MODE", "mock")
    os.environ["WARMUP_ON_START"] = "false"
    from app.config import settings
    from app.retrieval import pipeline
    from benchkit import CHECKPOINT_DIR, hardware, now

    data = json.loads((Path(__file__).resolve().parents[1] / "eval" /
                       "evidence_calibration.json").read_text())
    groups = {k: v for k, v in data.items() if isinstance(v, list)}
    rows = []
    for group, qs in groups.items():
        for q in qs:
            res = pipeline.run_retrieval(q, "overview", domain="all", use_cache=False)
            n = res.counts.get("reranked", 0)
            top = res.ranked[0][1] if n and res.ranked else None
            rows.append({"group": group, "question": q, "top_score": top})
            print(f"{group:24s} {top if top is None else round(top, 4)}  {q[:70]}", flush=True)
    table = {}
    for t in [float(x) for x in args.thresholds.split(",")]:
        table[t] = {g: {"n": len(qs), "pass_to_llm": sum(
            1 for r in rows if r["group"] == g and r["top_score"] is not None
            and r["top_score"] >= t)} for g, qs in groups.items()}
    out = {"recorded_at": now(), "hardware": hardware(), "reranker": settings.reranker_model,
           "rerank_pool": settings.rerank_pool, "rerank_max_length": settings.rerank_max_length,
           "configured_threshold": settings.fast_min_evidence_score, "rows": rows,
           "by_threshold": table}
    (CHECKPOINT_DIR / "evidence_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
