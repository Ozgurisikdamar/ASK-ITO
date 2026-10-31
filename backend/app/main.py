"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import eval as eval_api, health, search, stream, suggest
from app.config import settings
from app.db.connection import close_pool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.warmup import start_background

    start_background()  # load models/indexes + pre-warm featured answers, off the request path
    yield
    close_pool()


app = FastAPI(title="Ask ITO", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(stream.router, prefix="/api", tags=["search"])
app.include_router(suggest.router, prefix="/api", tags=["suggest"])
app.include_router(eval_api.router, prefix="/api", tags=["eval"])


@app.get("/")
def root() -> dict:
    return {"name": "Ask ITO", "version": "0.2.0", "docs": "/docs"}
