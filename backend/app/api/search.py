"""/api/search route (spec §9)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval import pipeline
from app.retrieval.service import run_search

router = APIRouter()

VALID_MODES = pipeline.RETRIEVAL_MODES + ["agentic"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: str = Field(default_factory=lambda: settings.default_mode)
    domain: str = "all"
    final_k: int | None = Field(default=None, ge=1, le=20)


@router.post("/search")
def search(req: SearchRequest):
    if req.mode not in VALID_MODES:
        return JSONResponse(
            status_code=400,
            content={"error": "bad_mode",
                     "detail": f"unknown mode '{req.mode}'; valid: {VALID_MODES}"},
        )
    if req.domain not in ("all", "nobel", "quantum", "ai"):
        return JSONResponse(
            status_code=400,
            content={"error": "bad_domain", "detail": "domain must be all|nobel|quantum|ai"},
        )
    try:
        if req.mode == "agentic":
            from app.agent.loop import run_agent

            return run_agent(req.query, req.domain, req.final_k or settings.final_k)
        return run_search(req.query, req.mode, req.domain, req.final_k)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=500, content={"error": "search_failed", "detail": str(exc)}
        )
