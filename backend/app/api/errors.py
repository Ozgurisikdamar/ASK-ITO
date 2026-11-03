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
    "busy": "Another answer is still being generated. Try again in a moment, or use Search.",
    # local model problems (app.llm.client.LLMError.code): say what to do
    "model_memory": "The local model could not be loaded: not enough memory. The default "
                    "Q8_0 model needs about 12 GB for Ollama alone (Docker Desktop: Settings "
                    "→ Resources → Memory). Search still works.",
    "model_not_found": "The local model is not installed. Run ./askito model, then try "
                       "again. Search still works.",
    "model_unreachable": "The local model server (Ollama) is not reachable. Start it and "
                         "try again. Search still works.",
    "model_timeout": "The local model did not answer in time. Please try again.",
    "model_load_timeout": "Loading the local model took too long. Check the memory and disk "
                          "available to Ollama, then try again. Search still works.",
    "generation_failed": "The answer could not be generated. Please try again.",
}
LLM_CODES = ("model_memory", "model_not_found", "model_unreachable", "model_timeout",
             "model_load_timeout", "generation_failed")


def safe_error(code: str, exc: BaseException | None = None) -> dict:
    """Public error payload; logs the details server-side with an error id."""
    error_id = uuid.uuid4().hex[:12]
    if exc is not None:
        log.error("error_id=%s code=%s %s: %s", error_id, code, exc.__class__.__name__, exc,
                  exc_info=exc)
    return {"error": code, "message": MESSAGES.get(code, MESSAGES["internal_error"]),
            "detail": MESSAGES.get(code, MESSAGES["internal_error"]), "error_id": error_id}


def llm_error(exc: BaseException, stage: str = "generation") -> dict:
    """Public payload for a failed LLM call: its classified code and the matching
    actionable message (unknown failures fall back to search_failed)."""
    code = getattr(exc, "code", None)
    payload = safe_error(code if code in LLM_CODES else "search_failed", exc)
    payload["stage"] = stage
    return payload


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
