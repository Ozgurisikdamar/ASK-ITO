"""/api/search route (spec §9)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval import pipeline
from app.retrieval.service import run_search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "agentic"
    domain: str = "all"
    final_k: int | None = None


@router.post("/search")
def search(req: SearchRequest):
    if req.mode == "agentic":
        # The agentic loop is implemented in M5.
        return JSONResponse(
            status_code=501,
            content={"error": "not_implemented",
                     "detail": "agentic mode is implemented in M5; use a retrieval mode"},
        )
    if req.mode not in pipeline.RETRIEVAL_MODES:
        return JSONResponse(
            status_code=400,
            content={"error": "bad_mode",
                     "detail": f"unknown mode '{req.mode}'; valid: {pipeline.RETRIEVAL_MODES}"},
        )
    if req.domain not in ("all", "nobel", "quantum", "ai"):
        return JSONResponse(
            status_code=400,
            content={"error": "bad_domain", "detail": "domain must be all|nobel|quantum|ai"},
        )
    try:
        return run_search(req.query, req.mode, req.domain, req.final_k or settings.final_k)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500, content={"error": "search_failed", "detail": str(exc)}
        )
