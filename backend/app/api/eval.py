"""Evaluation API. M7: test-set review endpoints (spec §8.1 step 3)."""
from __future__ import annotations

import psycopg
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.eval import testset
from app.retrieval.hydrate import body_without_prefix

router = APIRouter()


def _conn():
    return psycopg.connect(settings.database_url)


@router.get("/eval/runs")
def list_runs():
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, config_name, status, started_at, finished_at, summary "
            "FROM eval_runs ORDER BY id DESC"
        ).fetchall()
    out = []
    for r in rows:
        summary = r[5] or {}
        out.append({
            "id": r[0], "config_name": r[1], "status": r[2],
            "started_at": r[3].isoformat() if r[3] else None,
            "finished_at": r[4].isoformat() if r[4] else None,
            "retrieval": summary.get("retrieval", {}),
            "ragas": summary.get("ragas", {}),
            "operational": summary.get("operational", {}),
            "behaviour": summary.get("behaviour", {}),
            "n_questions": summary.get("n_questions"),
        })
    return {"runs": out}


@router.get("/eval/runs/{run_id}")
def run_detail(run_id: int):
    with _conn() as conn:
        run = conn.execute(
            "SELECT id, config_name, status, config_json, summary FROM eval_runs WHERE id=%s",
            (run_id,),
        ).fetchone()
        if not run:
            return JSONResponse(status_code=404, content={"error": "unknown_run"})
        rows = conn.execute(
            """SELECT r.question_id, q.question, q.qtype, q.domain, q.reference,
                      q.gold_chunk_ids, q.gold_doc_ids,
                      r.answer, r.retrieved_chunk_ids, r.metrics, r.latency_ms, r.llm_calls, r.trace
               FROM eval_results r JOIN eval_questions q ON q.id = r.question_id
               WHERE r.run_id=%s ORDER BY r.question_id""",
            (run_id,),
        ).fetchall()
    results = [
        {"question_id": r[0], "question": r[1], "qtype": r[2], "domain": r[3],
         "reference": r[4], "gold_chunk_ids": r[5] or [], "gold_doc_ids": r[6] or [],
         "answer": r[7], "retrieved_chunk_ids": r[8] or [], "metrics": r[9] or {},
         "latency_ms": r[10], "llm_calls": r[11], "trace": r[12] or []}
        for r in rows
    ]
    return {"id": run[0], "config_name": run[1], "status": run[2],
            "config_json": run[3] or {}, "summary": run[4] or {}, "results": results}


@router.get("/eval/compare")
def compare(a: int, b: int):
    metric = "doc_ndcg@10"
    with _conn() as conn:
        def fetch(rid):
            return {r[0]: (r[1] or {}) for r in conn.execute(
                "SELECT question_id, metrics FROM eval_results WHERE run_id=%s", (rid,)).fetchall()}
        ma, mb = fetch(a), fetch(b)
        qinfo = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT id, qtype, domain FROM eval_questions").fetchall()}
    rows = []
    for qid in sorted(set(ma) & set(mb)):
        va, vb = ma[qid].get(metric), mb[qid].get(metric)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            rows.append({"question_id": qid, "qtype": qinfo.get(qid, ("", ""))[0],
                         "a": va, "b": vb, "delta": round(vb - va, 4)})
    rows.sort(key=lambda x: x["delta"])
    return {"a": a, "b": b, "metric": metric,
            "improved": [r for r in rows if r["delta"] > 0][-10:][::-1],
            "regressed": [r for r in rows if r["delta"] < 0][:10]}


class RunRequest(BaseModel):
    configs: list[str]


@router.post("/eval/runs")
def start_run(req: RunRequest, background: BackgroundTasks):
    from app.eval import run as eval_run

    modes = eval_run.ALL_MODES if req.configs == ["all"] else req.configs
    bad = [m for m in modes if m not in eval_run.ALL_MODES]
    if bad:
        return JSONResponse(status_code=400, content={"error": "bad_configs", "detail": bad})
    background.add_task(eval_run.run, modes)
    return {"status": "started", "configs": modes}


def _gold_previews(chunk_ids: list[int]) -> list[dict]:
    if not chunk_ids:
        return []
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            """SELECT c.id, d.title, c.section, c.text
               FROM chunks c JOIN documents d ON d.id = c.doc_id
               WHERE c.id = ANY(%s)""",
            (chunk_ids,),
        ).fetchall()
    return [
        {"chunk_id": int(r[0]), "doc_title": r[1], "section": r[2],
         "snippet": body_without_prefix(r[3])[:260]}
        for r in rows
    ]


@router.get("/eval/review")
def get_review():
    drafts = testset.read_jsonl(testset.DRAFT)
    state = testset.read_json(testset.REVIEW_STATE, {})
    accepted = {q["id"]: q for q in testset.read_jsonl(testset.TESTSET)}
    items = []
    for d in drafts:
        qid = d["id"]
        status = state.get(qid, "pending")
        current = accepted.get(qid, d)
        items.append({
            **current,
            "status": status,
            "gold_chunks": _gold_previews(current.get("gold_chunk_ids") or []),
        })
    summary = {
        "total": len(drafts),
        "accepted": sum(1 for s in state.values() if s == "accepted"),
        "rejected": sum(1 for s in state.values() if s == "rejected"),
        "pending": len(drafts) - len(state),
    }
    return {"items": items, "summary": summary}


class ReviewAction(BaseModel):
    id: str
    action: str  # accept | reject
    question: str | None = None
    reference: str | None = None


@router.post("/eval/review")
def post_review(action: ReviewAction):
    drafts = {q["id"]: q for q in testset.read_jsonl(testset.DRAFT)}
    if action.id not in drafts:
        return JSONResponse(status_code=404, content={"error": "unknown_id"})
    if action.action not in ("accept", "reject"):
        return JSONResponse(status_code=400, content={"error": "bad_action"})

    state = testset.read_json(testset.REVIEW_STATE, {})
    accepted = {q["id"]: q for q in testset.read_jsonl(testset.TESTSET)}

    if action.action == "reject":
        state[action.id] = "rejected"
        accepted.pop(action.id, None)
    else:
        q = dict(drafts[action.id])
        if action.question is not None:
            q["question"] = action.question
        if action.reference is not None:
            q["reference"] = action.reference or None
        q.pop("status", None)
        q.pop("gold_chunks", None)
        q.pop("source", None)
        errs = testset.validate(q)
        if errs:
            return JSONResponse(status_code=400, content={"error": "invalid", "detail": errs})
        accepted[action.id] = q
        state[action.id] = "accepted"

    testset.write_json(testset.REVIEW_STATE, state)
    testset.write_jsonl(testset.TESTSET, accepted.values())
    return {"id": action.id, "status": state[action.id],
            "accepted_total": sum(1 for s in state.values() if s == "accepted")}
