"""Agentic (AI Mode) LLM-call audit on four fixed queries (no benchmark).

  # current code: pass 1 = empty tool cache, pass 2 = same queries again
  python scripts/agent_audit.py --model qwen3.5:9b-q8_0 --label current --passes 2
  # baseline: the same script against an older checkout of the app
  python scripts/agent_audit.py --model qwen3.5:9b-q8_0 --label baseline \
      --app-root /path/to/checkout-at-4433ebd/backend --passes 1

Runs the real agent (real retrieval, reranker and local model) on one simple
factual, one multi-hop, one comparison and one near-domain unsupported query
and records per run: LLM calls by role (analyze / rewrite / sufficiency /
answer), searches and reranker passes, rounds, deterministic guard hits,
agent tool-cache hits, the answer, citations and latency. With --app-root the
`app` package is imported from that checkout, so the baseline is the old code
running against the same database, index and model. Output:
eval/checkpoints/quant_smoke/agent_audit_<label>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

QUERIES = [
    ("simple_fact", "What is a transmon qubit?"),
    ("multi_hop", "What did Hopfield and Hinton each contribute to the 2024 Nobel Prize "
                  "in Physics?"),
    ("comparison", "How do superconducting qubits compare with trapped-ion qubits?"),
    ("near_domain_unsupported", "Which organisation received the 2024 Nobel Peace Prize?"),
]
GUARD_STEPS = ("subquestion_skipped", "search_skipped", "sufficiency_stopped",
               "sufficiency_skipped", "rewrite_skipped")
OUT = Path(os.environ.get("ASKITO_CHECKPOINT_DIR",
                          Path(__file__).resolve().parents[1] / "eval" / "checkpoints")) \
    / "quant_smoke"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--app-root", help="backend directory whose `app` package to run")
    args = ap.parse_args()
    root = Path(args.app_root).resolve() if args.app_root else \
        Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)  # .env / data paths resolve inside that checkout
    os.environ.update({"WARMUP_ON_START": "false", "PREWARM_FEATURED": "false",
                       "ANSWER_CACHE_ENABLED": "false", "REDIS_ENABLED": "false",
                       "AGENT_TOOL_CACHE": "true", "LLM_DEFAULT_MODEL": args.model})
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")

    import subprocess

    from app.agent.loop import run_agent
    from app.config import settings
    from app.llm.metrics import recording
    from app.retrieval import pipeline

    try:
        from app.agent import tool_cache
        tool_cache.clear()
        has_tool_cache = True
    except ImportError:  # baseline code predates the tool cache
        has_tool_cache = False
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                      text=True).strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"

    rows = []
    for p in range(1, args.passes + 1):
        for kind, q in QUERIES:
            pipeline.cache_clear()
            t0 = time.perf_counter()
            try:
                with recording() as rec:
                    res = run_agent(q, domain="all")
            except Exception as exc:  # noqa: BLE001
                rows.append({"pass": p, "kind": kind, "question": q,
                             "error": f"{exc.__class__.__name__}: {str(exc)[:200]}",
                             "total_s": round(time.perf_counter() - t0, 1)})
                print(json.dumps(rows[-1]), flush=True)
                continue
            m = res["metrics"]
            steps = [s["step"] for s in res["trace"]]
            row = {
                "pass": p, "kind": kind, "question": q,
                "total_s": round(time.perf_counter() - t0, 1),
                "llm_calls": res["llm_calls"],
                "by_role": {r: v["calls"] for r, v in m["llm"]["by_role"].items()},
                "llm_s_by_role": {r: round(v["ms"] / 1000, 1)
                                  for r, v in m["llm"]["by_role"].items()},
                "prompt_tokens": m["llm"].get("prompt_tokens"),
                "completion_tokens": m["llm"].get("completion_tokens"),
                "rounds": res["rounds"], "searches": steps.count("search"),
                "rerank_passes": steps.count("rerank") + steps.count("final_rerank"),
                "qtype": m["agent"].get("qtype"),
                "subquestions": m["agent"].get("subquestions"),
                "guards": {g: steps.count(g) for g in GUARD_STEPS if g in steps},
                "tool_cache_hits": rec.notes.get("tool_cache_hits", {}),
                "answer": res["answer"], "citations": [c["n"] for c in res["citations"]],
                "abstained": res["answer"].strip().startswith("I could not find"),
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("pass", "kind", "total_s", "by_role",
                                                  "searches", "rounds", "guards",
                                                  "tool_cache_hits")}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"label": args.label, "code_sha": sha, "app_root": str(root), "model": args.model,
           "tool_cache_available": has_tool_cache,
           "max_rounds": settings.agent_max_rounds,
           "max_llm_calls": settings.agent_max_llm_calls,
           "timeout_s": settings.agent_timeout_s,
           "rewrite_policy": settings.agent_rewrite_policy,
           "adaptive_sufficiency": settings.agent_adaptive_sufficiency, "rows": rows,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = OUT / f"agent_audit_{args.label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
