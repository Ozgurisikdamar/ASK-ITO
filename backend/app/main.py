"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import eval as eval_api, health, search
from app.config import settings

app = FastAPI(title="Ask ITO", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(eval_api.router, prefix="/api", tags=["eval"])


@app.get("/")
def root() -> dict:
    return {"name": "Ask ITO", "version": "0.1.0", "docs": "/docs"}
