"""Offline claim-level citation support over stored benchmark answers.

  python scripts/claim_support_pass.py --labels QS_qwen3.5-9b --judge qwen3:8b

Reads eval/checkpoints/eval_<label>.jsonl, rebuilds the passages each answer was
generated from (source order = citation numbers; each body capped at
OVERVIEW_CONTEXT_CHARS, as in build_context for the overview prompt) and asks a
local judge model whether each cited passage supports each claim
(app.eval.claim_support). One record per answer is appended to
eval/checkpoints/claims_<judge>.jsonl (resumable). Never part of the serving path.
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
    ap.add_argument("--labels", required=True)
    ap.add_argument("--judge", required=True, help="local judge model (Ollama tag)")
    ap.add_argument("--context-chars", type=int, default=0,
                    help="passage cap (default: OVERVIEW_CONTEXT_CHARS)")
    args = ap.parse_args()
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")
    os.environ["WARMUP_ON_START"] = "false"

    from app.config import settings
    from app.eval.claim_support import evaluate_answer, llm_judge
    from app.llm.client import make_client
    from app.retrieval.hydrate import body_without_prefix
    from app.retrieval.store import store
    from benchkit import CHECKPOINT_DIR, Checkpoint, mean_or_none, ollama_info

    chars = args.context_chars or settings.overview_context_chars
    store.ensure_loaded()
    client = make_client(settings.llm_mode, settings.llm_base_url, settings.llm_api_key,
                         args.judge)
    judge = llm_judge(client, args.judge)
    safe = args.judge.replace(":", "_").replace("/", "_")
    ck = Checkpoint(CHECKPOINT_DIR / f"claims_{safe}.jsonl")
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    work = []
    for label in labels:
        for u in Checkpoint(CHECKPOINT_DIR / f"eval_{label}.jsonl").units():
            work.append((label, u))
    ck.header(benchmark="claim_support", judge=ollama_info(settings.llm_base_url, args.judge),
              labels=labels, planned_units=len(work), context_chars=chars)
    for label, u in work:
        key = f"{label}|{u['key']}"
        if ck.has(key):
            continue
        metas = store.many(u["source_chunk_ids"])
        passages = {i: body_without_prefix(metas[c].text)[:chars]
                    for i, c in enumerate(u["source_chunk_ids"], start=1) if c in metas}
        try:
            res = evaluate_answer(u["answer"], passages, judge)
        except Exception as exc:  # noqa: BLE001
            ck.unit(key, status="error", error=f"{exc.__class__.__name__}: {str(exc)[:200]}")
            continue
        ck.unit(key, label=label, question_id=u["question_id"], rep=u.get("rep"), **res)
        print(f"[claims] {key} n={res['n_claims']} support={res.get('claim_support_rate')}",
              flush=True)
    rows = ck.units()
    out = {}
    for label in labels:
        rs = [r for r in rows if r["label"] == label and not r.get("not_found")]
        allc = [c for r in rs for c in r["claims"]]
        n = len(allc)
        out[label] = {
            "answers_judged": len(rs), "claims": n,
            "claim_support_rate": sum(c["status"] == "supported" for c in allc) / n if n else None,
            "partial_support_rate": sum(c["status"] == "partial" for c in allc) / n if n else None,
            "unsupported_claim_rate": sum(c["status"] == "unsupported" for c in allc) / n
            if n else None,
            "uncited_claim_rate": sum(c["status"] == "uncited" for c in allc) / n if n else None,
            "irrelevant_citation_rate_mean": mean_or_none(
                [r.get("irrelevant_citation_rate") for r in rs]),
            "not_found_answers": sum(1 for r in rows if r["label"] == label and r.get("not_found")),
        }
    ck.footer(len(work))
    (CHECKPOINT_DIR / f"claims_{safe}.summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
