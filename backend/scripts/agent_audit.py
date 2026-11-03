"""Agentic (AI Mode) LLM-call audit on the five smoke questions (no benchmark).

  python scripts/agent_audit.py --model qwen3.5:9b-q8_0 [--passes 2]

Runs the real agent (real retrieval, reranker and local model) and records per
question: LLM calls by role (analyze / rewrite / sufficiency / answer), rounds,
searches, which deterministic guards fired (duplicate sub-question, query
already searched, no new passages), agent tool-cache hits and total time.
Pass 1 starts with an empty tool cache; pass 2 repeats the same questions in the
same process, so analyse / rewrite / sufficiency can come from the tool cache
(in-process L1; Redis is not used here) while the final answer is always
generated. Output: eval/checkpoints/quant_smoke/agent_audit_<model>.json
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

from quant_smoke import OUT, QUESTIONS  # noqa: E402

GUARD_STEPS = ("subquestion_skipped", "search_skipped", "sufficiency_stopped",
               "sufficiency_skipped", "rewrite_skipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()
    os.environ.update({"WARMUP_ON_START": "false", "PREWARM_FEATURED": "false",
                       "ANSWER_CACHE_ENABLED": "false", "REDIS_ENABLED": "false",
                       "AGENT_TOOL_CACHE": "true", "LLM_DEFAULT_MODEL": args.model})
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")

    from app.agent import tool_cache
    from app.agent.loop import run_agent
    from app.config import settings
    from app.llm.metrics import recording
    from app.retrieval import pipeline

    tool_cache.clear()
    rows = []
    for p in range(1, args.passes + 1):
        for q in QUESTIONS:
            pipeline.cache_clear()
            t0 = time.perf_counter()
            with recording() as rec:
                res = run_agent(q, domain="all")
            m = res["metrics"]
            steps = [s["step"] for s in res["trace"]]
            row = {
                "pass": p, "question": q, "total_s": round(time.perf_counter() - t0, 1),
                "llm_calls": res["llm_calls"],
                "by_role": {r: v["calls"] for r, v in m["llm"]["by_role"].items()},
                "llm_ms_by_role": {r: v["ms"] for r, v in m["llm"]["by_role"].items()},
                "prompt_tokens": m["llm"].get("prompt_tokens"),
                "completion_tokens": m["llm"].get("completion_tokens"),
                "rounds": res["rounds"], "searches": steps.count("search"),
                "qtype": m["agent"].get("qtype"), "subquestions": m["agent"].get("subquestions"),
                "guards": {g: steps.count(g) for g in GUARD_STEPS if g in steps},
                "tool_cache_hits": rec.notes.get("tool_cache_hits", {}),
                "answer": res["answer"], "citations": [c["n"] for c in res["citations"]],
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("pass", "question", "total_s", "by_role",
                                                  "guards", "tool_cache_hits")}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"model": args.model, "max_rounds": settings.agent_max_rounds,
           "max_llm_calls": settings.agent_max_llm_calls,
           "rewrite_policy": settings.agent_rewrite_policy,
           "adaptive_sufficiency": settings.agent_adaptive_sufficiency, "rows": rows}
    path = OUT / f"agent_audit_{args.model.replace(':', '_')}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
