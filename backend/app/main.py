"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import eval as eval_api, health, search, stream, suggest
from app.config import settings
from app.db.connection import close_pool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import threading

    from app.eval.jobs import mark_interrupted
    from app.warmup import start_background

    start_background()  # load models/indexes + pre-warm featured answers, off the request path
    # evaluation runs cut short by a restart are marked as such (not "running" forever)
    threading.Thread(target=mark_interrupted, name="eval-cleanup", daemon=True).start()
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

@app.exception_handler(Exception)
async def unhandled_error(_request: Request, exc: Exception):
    """Last-resort handler: generic message + error id; details only in the log."""
    from app.api.errors import safe_error

    return JSONResponse(status_code=500, content=safe_error("internal_error", exc))


app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(stream.router, prefix="/api", tags=["search"])
app.include_router(suggest.router, prefix="/api", tags=["suggest"])
app.include_router(eval_api.router, prefix="/api", tags=["eval"])


@app.get("/")
def root() -> dict:
    return {"name": "Ask ITO", "version": "0.2.0", "docs": "/docs"}
