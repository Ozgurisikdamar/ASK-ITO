"""Search suggestions: featured queries (pre-warmed, answered instantly)."""
from __future__ import annotations

from fastapi import APIRouter

from app.warmup import FEATURED_QUERIES, state

router = APIRouter()


@router.get("/suggest/featured")
def featured():
    return {"queries": FEATURED_QUERIES, "prewarm": state["prewarm"]}
