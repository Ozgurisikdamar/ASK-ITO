"""Evaluation API. M7: test-set review endpoints (spec §8.1 step 3)."""
from __future__ import annotations

import psycopg
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.eval import testset
from app.retrieval.hydrate import body_without_prefix

router = APIRouter()


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
