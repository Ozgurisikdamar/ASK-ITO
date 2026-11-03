"""/api/search route (spec §9)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.errors import generation_limiter, llm_error, needs_generation, safe_error
from app.llm.client import LLMError, get_llm_client
from app.config import settings
from app.retrieval import pipeline
from app.retrieval.service import run_search

router = APIRouter()

VALID_MODES = pipeline.RETRIEVAL_MODES + ["agentic", "ai"]


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
    holds_slot = needs_generation(req.mode)
    if holds_slot and not generation_limiter.try_acquire():
        return JSONResponse(status_code=429, content=safe_error("busy"))
    mode, route_info = req.mode, None
    if mode == "ai":  # AI Mode: one grounded answer, or the agent when needed
        from app.agent.router import route

        path, reason = route(req.query)
        mode, route_info = ("agentic" if path == "agent" else "overview"), \
            {"route": path, "route_reason": reason}
    try:
        if holds_slot and req.mode != "search":
            client = get_llm_client()  # load once in the background (see stream.model_ready)
            if hasattr(client, "ensure_loaded"):
                for role in ("agent", "answer") if mode == "agentic" else ("answer",):
                    client.ensure_loaded(settings.model_for(role))
        if mode == "agentic":
            from app.agent.loop import run_agent

            res = run_agent(req.query, req.domain, req.final_k or settings.final_k)
        else:
            res = run_search(req.query, mode, req.domain, req.final_k)
        return {**res, **route_info} if route_info else res
    except LLMError as exc:
        status = 504 if exc.code in ("model_timeout", "model_load_timeout") else 503
        return JSONResponse(status_code=status, content=llm_error(exc))
    except Exception as exc:  # noqa: BLE001  (details are logged, not returned)
        return JSONResponse(status_code=500, content=safe_error("search_failed", exc))
    finally:
        if holds_slot:
            generation_limiter.release()
