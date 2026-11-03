"""Request ids and log lines.

Every HTTP request gets an id: the caller's X-Request-ID when it is a short,
safe token, otherwise a new one. It is sent back in the X-Request-ID response
header, kept in a context variable (worker threads started with a copied
context keep it), written into every askito log line as rid=..., and used as
the `rid` of the search stream's events, so what the user saw, the response
header and the server log can be matched.

Log lines carry timings, counts and error classes only, never question or
answer text (see app.llm.metrics.log_request).
"""
from __future__ import annotations

import contextvars
import logging
import re
import uuid

request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id",
                                                                        default=None)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_FORMAT = "%(asctime)s %(levelname)s %(name)s rid=%(rid)s %(message)s"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def current_id() -> str:
    """The id of the request being handled, or a new one outside a request."""
    return request_id.get() or new_id()


class RequestIdMiddleware:
    """Pure ASGI (does not buffer streaming responses)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = next((v.decode("latin-1") for k, v in scope.get("headers", [])
                         if k == b"x-request-id"), "")
        rid = incoming if _SAFE_ID.match(incoming) else new_id()
        token = request_id.set(rid)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                headers = [h for h in message.get("headers", []) if h[0] != b"x-request-id"]
                message = {**message, "headers": headers + [(b"x-request-id", rid.encode())]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id.reset(token)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = request_id.get() or "-"
        return True


def setup_logging(level: str = "INFO") -> None:
    """Print the askito.* loggers (INFO by default) with the request id. Without
    a handler Python shows only warnings, so the per-request timing lines and
    ingestion results were not visible. Idempotent; records still propagate."""
    logger = logging.getLogger("askito")
    if any(getattr(h, "_askito", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler._askito = True  # type: ignore[attr-defined]
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
