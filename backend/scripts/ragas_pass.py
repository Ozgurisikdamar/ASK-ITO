"""RAGAS pass over stored benchmark answers, with one fixed local judge.

  python scripts/ragas_pass.py --labels B_qwen3-8b_hybrid_rerank,C_... \
      --judge qwen3:14b --questions lat12

Reads eval/checkpoints/eval_<label>.jsonl (repetition 1), rebuilds for each
answered question the exact context the generator saw (each source passage
body capped at ANSWER_CONTEXT_CHARS, as in build_context) and scores
faithfulness, answer_relevancy, context_precision and context_recall with the
judge model through the project's LLM client (Ollama native API, thinking off,
JSON mode, one request at a time). Every (label, question) score is appended to
eval/checkpoints/ragas_<judge>.jsonl as soon as it is computed, so an
interrupted pass resumes where it stopped. Generation and judging never run at
the same time (run this after the generation benchmarks).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="comma-separated eval_bench labels")
    ap.add_argument("--judge", required=True, help="local judge model (Ollama tag)")
    ap.add_argument("--questions", default="lat12", help="subset of questions to judge")
    args = ap.parse_args()

    os.environ["LLM_JUDGE_MODEL"] = args.judge
    os.environ.setdefault("LLM_MODE", "ollama")
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434")
    os.environ["WARMUP_ON_START"] = "false"

    import psycopg
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    from app.config import settings
    from app.eval import ragas_eval
    from app.eval.metrics import refusal_correct
    from app.retrieval.hydrate import body_without_prefix
    from app.retrieval.store import store
    from benchkit import CHECKPOINT_DIR, Checkpoint, mean_or_none, ollama_info
    from subsets import load_questions

    store.ensure_loaded()
    with psycopg.connect(settings.database_url) as conn:
        questions = {q["id"]: q for q in load_questions(conn, args.questions)}
    safe = args.judge.replace(":", "_").replace("/", "_")
    ck = Checkpoint(CHECKPOINT_DIR / f"ragas_{safe}.jsonl")

    work = []
    for label in [x.strip() for x in args.labels.split(",") if x.strip()]:
        src = Checkpoint(CHECKPOINT_DIR / f"eval_{label}.jsonl")
        for u in src.units():
            q = questions.get(u["question_id"])
            if u.get("rep") != 1 or q is None or not q["reference"]:
                continue
            work.append((label, q, u))
    judge_meta = ollama_info(settings.llm_base_url, args.judge)
    ck.header(benchmark="ragas", judge=judge_meta, labels=args.labels,
              questions=args.questions, planned_units=len(work),
              judge_settings={"think": settings.llm_think, "num_ctx": settings.llm_num_ctx,
                              "max_tokens": settings.llm_judge_max_tokens,
                              "workers": 1, "timeout_s": settings.ragas_timeout_s},
              context_chars=settings.answer_context_chars)

    client, model = ragas_eval.judge_client()
    ok, msg = client.ping()
    if not ok:
        raise SystemExit(f"judge not available: {msg}")
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))

    for label, q, u in work:
        key = f"{label}|{q['id']}"
        if ck.has(key):
            continue
        metas = store.many(u["source_chunk_ids"])
        contexts = [body_without_prefix(metas[c].text)[: settings.answer_context_chars]
                    for c in u["source_chunk_ids"] if c in metas]
        judge = ragas_eval.make_judge_llm(client, model)
        ds = Dataset.from_list([{"question": q["question"], "answer": u["answer"],
                                 "contexts": contexts or [""], "reference": q["reference"]}])
        t0 = time.perf_counter()
        try:
            res = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision,
                                        context_recall],
                           llm=judge, embeddings=emb, run_config=ragas_eval.run_config(),
                           raise_exceptions=False, show_progress=False)
            row = res.to_pandas().iloc[0]
            scores = {}
            for m in METRICS:
                v = row.get(m)
                scores[m] = None if v is None or (isinstance(v, float) and math.isnan(v)) \
                    else float(v)
        except Exception as exc:  # noqa: BLE001
            ck.unit(key, status="error", label=label, question_id=q["id"],
                    error=f"{exc.__class__.__name__}: {str(exc)[:300]}")
            print(f"[ragas] {key} ERROR {exc.__class__.__name__}", flush=True)
            continue
        ck.unit(key, label=label, question_id=q["id"], qtype=q["qtype"],
                not_found=refusal_correct(u["answer"]), scores=scores,
                judge_calls=len(judge.calls),
                judge_ms=round((time.perf_counter() - t0) * 1000, 1),
                judge_tokens={"prompt": sum(c.get("prompt_tokens") or 0 for c in judge.calls),
                              "completion": sum(c.get("completion_tokens") or 0
                                                for c in judge.calls)},
                missing=[m for m, v in scores.items() if v is None])
        print(f"[ragas] {key} {json.dumps(scores)} {len(judge.calls)} calls", flush=True)

    out = {"judge": judge_meta, "header": ck.headers[-1], "by_label": {}}
    for label in sorted({w[0] for w in work}):
        rows = [r for r in ck.units() if r["label"] == label]
        planned = sum(1 for w in work if w[0] == label)
        out["by_label"][label] = {
            "planned": planned, "completed": len(rows),
            "status": "RUN" if len(rows) == planned else f"PARTIAL ({len(rows)}/{planned})",
            **{m: mean_or_none([r["scores"].get(m) for r in rows]) for m in METRICS},
            "missing_values": {m: sum(1 for r in rows if r["scores"].get(m) is None)
                               for m in METRICS},
            "judge_ms_mean": mean_or_none([r["judge_ms"] for r in rows]),
        }
    ck.footer(len(work))
    (CHECKPOINT_DIR / f"ragas_{safe}.summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["by_label"], indent=2))


if __name__ == "__main__":
    main()
