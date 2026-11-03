"""Safe API errors and the answer-generation concurrency limit.

Clients get a stable error code, a generic message and an error id; the
exception text (which may contain URLs, paths or other internals) is only
written to the server log under that id.
"""
from __future__ import annotations

import logging
import threading
import uuid

from app.config import settings

log = logging.getLogger("askito.api")

MESSAGES = {
    "search_failed": "The search could not be completed. Please try again.",
    "internal_error": "Something went wrong on the server. Please try again.",
    "busy": "Another answer is being generated. Try again in a moment, or use Search.",
}


def safe_error(code: str, exc: BaseException | None = None) -> dict:
    """Public error payload; logs the details server-side with an error id."""
    error_id = uuid.uuid4().hex[:12]
    if exc is not None:
        log.error("error_id=%s code=%s %s: %s", error_id, code, exc.__class__.__name__, exc,
                  exc_info=exc)
    return {"error": code, "message": MESSAGES.get(code, MESSAGES["internal_error"]),
            "detail": MESSAGES.get(code, MESSAGES["internal_error"]), "error_id": error_id}


class GenerationLimiter:
    """At most MAX_CONCURRENT_GENERATIONS requests may run an LLM at once (one
    local model serves one request at a time; extra requests would only queue
    for minutes). Non-blocking: callers get False and report "busy"."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= max(1, settings.max_concurrent_generations):
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        return self._active


generation_limiter = GenerationLimiter()


def needs_generation(mode: str) -> bool:
    """Every mode except "search" may call the LLM."""
    return mode != "search"
