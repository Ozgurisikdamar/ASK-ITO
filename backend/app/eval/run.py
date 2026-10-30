"""Evaluation runner (spec §8.3).

Runs one or more pipeline configurations over the reviewed test set, stores
per-question results, aggregates retrieval + behaviour + operational metrics,
computes RAGAS generation metrics, and writes docs/EVAL_REPORT.md.

  python -m app.eval.run --configs all
  python -m app.eval.run --configs hybrid,agentic

Resumable: already-answered (run_id, question_id) pairs are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone

import psycopg


def _finite(obj):
    """Replace NaN/Inf floats with None (Postgres JSONB rejects them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj

from app.config import settings
from app.eval import metrics, ragas_eval, report
from app.retrieval import pipeline

ALL_MODES = pipeline.RETRIEVAL_MODES + ["agentic"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _config_json(mode: str) -> dict:
    return {
        "mode": mode,
        "final_k": settings.final_k,
        "bm25": {"k1": settings.bm25_k1, "b": settings.bm25_b},
        "rrf_k": settings.rrf_k,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "models": {
            "agent": settings.model_for("agent"),
            "answer": settings.model_for("answer"),
            "judge": settings.model_for("judge"),
        },
        "git_commit": _git_commit(),
    }


def _questions(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, question, reference, gold_chunk_ids, gold_doc_ids, qtype, domain "
        "FROM eval_questions ORDER BY id"
    ).fetchall()
    return [
        {"id": r[0], "question": r[1], "reference": r[2],
         "gold_chunk_ids": set(r[3] or []), "gold_doc_ids": set(r[4] or []),
         "qtype": r[5], "domain": r[6]}
        for r in rows
    ]


def _docs_of(conn, chunk_ids: list[int]) -> list[int]:
    if not chunk_ids:
        return []
    m = {int(r[0]): int(r[1]) for r in conn.execute(
        "SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (chunk_ids,)).fetchall()}
    seen, out = set(), []
    for c in chunk_ids:
        d = m.get(c)
        if d is not None and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _contexts(conn, chunk_ids: list[int]) -> list[str]:
    if not chunk_ids:
        return []
    m = {int(r[0]): r[1] for r in conn.execute(
        "SELECT id, text FROM chunks WHERE id = ANY(%s)", (chunk_ids[:5],)).fetchall()}
    return [m[c].split("\n\n", 1)[-1] for c in chunk_ids[:5] if c in m]


def _answer_for(mode: str, q: dict) -> dict:
    if mode == "agentic":
        from app.agent.loop import run_agent

        r = run_agent(q["question"], domain="all", final_k=settings.final_k)
        r.setdefault("ranked_chunk_ids", [c["chunk_id"] for c in r.get("sources", [])])
        return r
    from app.retrieval.service import run_search

    r = run_search(q["question"], mode=mode, domain="all", final_k=settings.final_k)
    r["llm_calls"] = 0 if mode == "closed_book" else 1
    return r


def run_config(conn, mode: str, questions: list[dict]) -> int:
    row = conn.execute(
        "INSERT INTO eval_runs (config_name, config_json, started_at, status) "
        "VALUES (%s,%s,%s,'running') RETURNING id",
        (mode, json.dumps(_config_json(mode)), datetime.now(timezone.utc)),
    ).fetchone()
    run_id = int(row[0])
    conn.commit()

    for q in questions:
        t0 = time.perf_counter()
        resp = _answer_for(mode, q)
        latency = int((time.perf_counter() - t0) * 1000)
        ranked = resp.get("ranked_chunk_ids", [])
        m = metrics.retrieval_metrics(
            ranked, _docs_of(conn, ranked), q["gold_chunk_ids"], q["gold_doc_ids"])
        m["refusal_correct"] = (
            float(metrics.refusal_correct(resp["answer"])) if q["qtype"] == "unanswerable" else None
        )
        conn.execute(
            """INSERT INTO eval_results
               (run_id, question_id, answer, retrieved_chunk_ids, metrics, trace, latency_ms, llm_calls)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (run_id, question_id) DO NOTHING""",
            (run_id, q["id"], resp["answer"], ranked, json.dumps(_finite(m)),
             json.dumps(resp.get("trace", [])), latency, resp.get("llm_calls", 0)),
        )
        conn.commit()

    summary = _summarize(conn, run_id, mode, questions)
    conn.execute(
        "UPDATE eval_runs SET status='done', finished_at=%s, summary=%s WHERE id=%s",
        (datetime.now(timezone.utc), json.dumps(_finite(summary)), run_id),
    )
    conn.commit()
    return run_id


def _summarize(conn, run_id: int, mode: str, questions: list[dict]) -> dict:
    rows = conn.execute(
        "SELECT question_id, answer, retrieved_chunk_ids, metrics, latency_ms, llm_calls "
        "FROM eval_results WHERE run_id=%s", (run_id,)).fetchall()
    per_q = [r[3] for r in rows]
    retrieval = metrics.aggregate(per_q)
    latencies = [r[4] for r in rows]
    qmap = {q["id"]: q for q in questions}

    # RAGAS over answerable questions
    samples = []
    for r in rows:
        q = qmap.get(r[0])
        if q and q["reference"]:
            samples.append({
                "question": q["question"], "answer": r[1],
                "contexts": _contexts(conn, r[2] or []), "reference": q["reference"],
            })
    ragas = ragas_eval.compute(samples)

    refusals = [r[3].get("refusal_correct") for r in rows if r[3].get("refusal_correct") is not None]
    return {
        "mode": mode,
        "n_questions": len(rows),
        "retrieval": retrieval,
        "ragas": ragas,
        "behaviour": {
            "unanswerable_refusal_rate": (sum(refusals) / len(refusals)) if refusals else None,
        },
        "operational": {
            "latency_ms": metrics.percentiles(latencies),
            "avg_llm_calls": (sum(r[5] for r in rows) / len(rows)) if rows else 0,
        },
    }


def run(configs: list[str]) -> dict:
    modes = ALL_MODES if configs == ["all"] else configs
    bad = [m for m in modes if m not in ALL_MODES]
    if bad:
        raise ValueError(f"unknown configs: {bad}")
    results = {}
    with psycopg.connect(settings.database_url) as conn:
        questions = _questions(conn)
        if not questions:
            raise RuntimeError("no eval_questions — run make testset-seed + testset-load")
        for mode in modes:
            print(f"[eval] running config: {mode} over {len(questions)} questions ...", flush=True)
            results[mode] = run_config(conn, mode, questions)
    report.generate(list(results.values()))
    return {"runs": results, "report": str(report.REPORT_PATH)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all", help="all | comma-separated modes")
    args = ap.parse_args()
    try:
        out = run([c.strip() for c in args.configs.split(",")])
    except Exception as exc:  # noqa: BLE001
        print(f"eval failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
