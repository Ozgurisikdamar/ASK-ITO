"""Evaluation runner (spec §8.3).

Runs one or more pipeline configurations over the reviewed test set, stores
per-question results, aggregates retrieval + behaviour + operational metrics,
computes RAGAS generation metrics, and writes docs/EVAL_REPORT.md.

  python -m app.eval.run --configs all
  python -m app.eval.run --configs hybrid,agentic
  python -m app.eval.run --configs bm25,hybrid_rerank --retrieval-only

--retrieval-only scores the ranked passages without generating answers (no LLM
call, except the query rewrite of hybrid_rerank_rewrite); --no-ragas skips the
RAGAS judge. The test set is resolved against the loaded corpus first
(app.eval.prepare). A failing configuration is stored with status "error" and
its message.
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

from app.config import settings
from app.eval import metrics, ragas_eval, report
from app.retrieval import pipeline


def _finite(obj):
    """Replace NaN/Inf floats with None (Postgres JSONB rejects them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj

# "search" produces no answer, so it is not an evaluation configuration
ALL_MODES = [m for m in pipeline.RETRIEVAL_MODES if m != "search"] + ["agentic"]
# configurations whose result is the answer itself: they need answers
ANSWER_ONLY_MODES = ("closed_book", "agentic")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _config_json(mode: str, answers: bool = True, ragas: bool = True,
                 source: str = "cli") -> dict:
    return {
        "mode": mode,
        "answers": answers,
        "ragas": ragas and answers,
        "source": source,
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
    r["llm_calls"] = r["metrics"]["llm"]["calls"]  # measured (0 when no sources)
    return r


def _retrieve_for(mode: str, q: dict) -> dict:
    """Ranked passages only (the same retrieval run_search does), no answer."""
    from app.llm.metrics import recording

    t0 = time.perf_counter()
    with recording() as rec:  # hybrid_rerank_rewrite still rewrites with the LLM
        result = pipeline.run_retrieval(q["question"], mode, domain="all",
                                        final_k=settings.final_k, use_cache=False)
    llm = rec.summary()
    return {
        "answer": "", "ranked_chunk_ids": [c for c, _ in result.ranked],
        "trace": result.trace_dicts(), "llm_calls": llm["calls"],
        "metrics": {"llm": llm, "retrieval_ms": int((time.perf_counter() - t0) * 1000)},
    }


def error_text(exc: BaseException) -> str:
    """What the evaluation page shows for a failed configuration: the actionable
    message for a local-model problem, otherwise the exception class and a short
    message (this is a developer page; the full traceback is in the server log)."""
    code = getattr(exc, "code", None)
    if code:
        from app.api.errors import MESSAGES

        if code in MESSAGES:
            return MESSAGES[code]
    return f"{exc.__class__.__name__}: {str(exc)[:200]}".rstrip(": ")


def create_run(conn, mode: str, *, answers: bool = True, ragas: bool = True,
               status: str = "running", source: str = "cli") -> int:
    row = conn.execute(
        "INSERT INTO eval_runs (config_name, config_json, started_at, status) "
        "VALUES (%s,%s,%s,%s) RETURNING id",
        (mode, json.dumps(_config_json(mode, answers, ragas, source)),
         datetime.now(timezone.utc), status),
    ).fetchone()
    conn.commit()
    return int(row[0])


def mark_error(conn, run_id: int, message: str, progress: dict | None = None) -> None:
    conn.rollback()
    conn.execute(
        "UPDATE eval_runs SET status='error', finished_at=%s, summary=%s WHERE id=%s",
        (datetime.now(timezone.utc), json.dumps({"error": message, "progress": progress}),
         run_id),
    )
    conn.commit()


def _ops(resp: dict) -> dict:
    """Operational metrics stored per question (no prompt/answer text)."""
    met = resp.get("metrics") or {}
    llm = met.get("llm") or {}
    return {
        "llm_ms": llm.get("total_ms"), "llm_calls": llm.get("calls"),
        "llm_by_role": llm.get("by_role"), "prompt_tokens": llm.get("prompt_tokens"),
        "completion_tokens": llm.get("completion_tokens"), "llm_errors": llm.get("errors"),
        "context_overflows": llm.get("context_overflows"),
        "retrieval_ms": met.get("retrieval_ms"), "rounds": resp.get("rounds"),
    }


def run_config(conn, mode: str, questions: list[dict], *, answers: bool = True,
               ragas: bool = True, run_id: int | None = None) -> int:
    """Run one configuration over `questions` and store the results. `run_id` is
    an already created (queued) row; it is created here otherwise. On failure
    the row gets status "error" with the message, then the exception propagates."""
    if run_id is None:
        run_id = create_run(conn, mode, answers=answers, ragas=ragas)
    progress = {"done": 0, "total": len(questions)}
    conn.execute("UPDATE eval_runs SET status='running', started_at=%s, summary=%s WHERE id=%s",
                 (datetime.now(timezone.utc), json.dumps({"progress": progress}), run_id))
    conn.commit()

    try:
        for q in questions:
            pipeline.cache_clear()  # every question is a real (uncached) retrieval
            t0 = time.perf_counter()
            resp = _answer_for(mode, q) if answers else _retrieve_for(mode, q)
            latency = int((time.perf_counter() - t0) * 1000)
            ranked = resp.get("ranked_chunk_ids", [])
            m = metrics.retrieval_metrics(
                ranked, _docs_of(conn, ranked), q["gold_chunk_ids"], q["gold_doc_ids"])
            m["refusal_correct"] = (
                float(metrics.refusal_correct(resp["answer"]))
                if answers and q["qtype"] == "unanswerable" else None
            )
            m["ops"] = _ops(resp)
            conn.execute(
                """INSERT INTO eval_results
                   (run_id, question_id, answer, retrieved_chunk_ids, metrics, trace,
                    latency_ms, llm_calls)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id, question_id) DO NOTHING""",
                (run_id, q["id"], resp["answer"], ranked, json.dumps(_finite(m)),
                 json.dumps(resp.get("trace", [])), latency, resp.get("llm_calls", 0)),
            )
            progress["done"] += 1
            conn.execute("UPDATE eval_runs SET summary=%s WHERE id=%s",
                         (json.dumps({"progress": progress}), run_id))
            conn.commit()

        summary = _summarize(conn, run_id, mode, questions, ragas=ragas and answers)
        summary["progress"] = progress
        conn.execute(
            "UPDATE eval_runs SET status='done', finished_at=%s, summary=%s WHERE id=%s",
            (datetime.now(timezone.utc), json.dumps(_finite(summary)), run_id),
        )
        conn.commit()
    except Exception as exc:
        mark_error(conn, run_id, error_text(exc), progress)
        raise
    return run_id


def _summarize(conn, run_id: int, mode: str, questions: list[dict],
               ragas: bool = True) -> dict:
    rows = conn.execute(
        "SELECT question_id, answer, retrieved_chunk_ids, metrics, latency_ms, llm_calls "
        "FROM eval_results WHERE run_id=%s", (run_id,)).fetchall()
    per_q = [r[3] for r in rows]
    retrieval = metrics.aggregate(per_q)
    latencies = [r[4] for r in rows]
    qmap = {q["id"]: q for q in questions}

    # RAGAS over answerable questions
    samples = []
    for r in rows if ragas else ():
        q = qmap.get(r[0])
        if q and q["reference"]:
            samples.append({
                "question": q["question"], "answer": r[1],
                "contexts": _contexts(conn, r[2] or []), "reference": q["reference"],
            })
    ragas_scores = (ragas_eval.compute(samples) if ragas
                    else ragas_eval._empty("not run (answers or RAGAS were not requested)"))

    refusals = [r[3].get("refusal_correct") for r in rows if r[3].get("refusal_correct") is not None]
    return {
        "mode": mode,
        "n_questions": len(rows),
        "retrieval": retrieval,
        "ragas": ragas_scores,
        "behaviour": {
            "unanswerable_refusal_rate": (sum(refusals) / len(refusals)) if refusals else None,
        },
        "operational": {
            "latency_ms": metrics.percentiles(latencies),
            "avg_llm_calls": (sum(r[5] for r in rows) / len(rows)) if rows else 0,
            "llm_ms": metrics.percentiles(
                [int(r[3]["ops"]["llm_ms"]) for r in rows
                 if (r[3].get("ops") or {}).get("llm_ms") is not None]),
            "p95_llm_calls": metrics.percentiles([r[5] for r in rows]).get("p95"),
        },
    }


class NoQuestions(RuntimeError):
    """The test set has no question usable with the loaded corpus."""


def check_modes(modes: list[str], answers: bool) -> None:
    bad = [m for m in modes if m not in ALL_MODES]
    if bad:
        raise ValueError(f"unknown configs: {bad}")
    if not answers and (needs := [m for m in modes if m in ANSWER_ONLY_MODES]):
        raise ValueError(f"these configurations need generated answers: {', '.join(needs)}")


def _norm_q(text: str) -> str:
    return " ".join((text or "").split()).lower()


def load_questions(conn, subset: str | None = None) -> tuple[list[dict], dict]:
    """The test set resolved against this corpus, in the shape run_config uses.
    `subset` (a name in eval/subsets.json) runs only that stratified slice."""
    from app.eval import prepare, testset

    prep = prepare.prepare(conn)
    if not prep["questions"]:
        why = ("eval/testset.jsonl is empty (run make testset-seed)" if not prep["total"]
               else "none of its gold articles are in this corpus")
        raise NoQuestions(f"No evaluation questions: {why}.")
    ids = {q["id"] for q in prep["questions"]}
    questions = [q for q in _questions(conn) if q["id"] in ids]
    if subset:
        wanted = testset.subset_questions(subset)
        if wanted is None:
            raise ValueError(f"unknown subset: {subset} (known: {', '.join(testset.subsets())})")
        want = {_norm_q(w) for w in wanted}
        questions = [q for q in questions if _norm_q(q["question"]) in want]
        if not questions:
            raise NoQuestions(f"No evaluation questions: subset '{subset}' has none in this corpus.")
    return questions, prep


def run(configs: list[str], answers: bool = True, ragas: bool = True,
        subset: str | None = None) -> dict:
    modes = ALL_MODES if configs == ["all"] else configs
    check_modes(modes, answers)
    results = {}
    with psycopg.connect(settings.database_url) as conn:
        questions, prep = load_questions(conn, subset)
        if prep["skipped"]:
            print(f"[eval] skipped {len(prep['skipped'])} questions: "
                  f"{prep['skipped'][:5]}", flush=True)
        for mode in modes:
            print(f"[eval] running config: {mode} over {len(questions)} questions ...", flush=True)
            results[mode] = run_config(conn, mode, questions, answers=answers, ragas=ragas)
    report.generate(list(results.values()))
    return {"runs": results, "report": str(report.REPORT_PATH)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all", help="all | comma-separated modes")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="score the ranked passages only; no answers, no RAGAS")
    ap.add_argument("--no-ragas", action="store_true", help="skip the RAGAS judge")
    ap.add_argument("--subset", default=None,
                    help="run only a named subset from eval/subsets.json (e.g. lat12, sub20)")
    args = ap.parse_args()
    try:
        out = run([c.strip() for c in args.configs.split(",")],
                  answers=not args.retrieval_only, ragas=not args.no_ragas,
                  subset=args.subset)
    except Exception as exc:  # noqa: BLE001
        print(f"eval failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
