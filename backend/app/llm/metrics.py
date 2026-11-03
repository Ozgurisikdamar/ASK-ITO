"""Per-request LLM call accounting (observability only; never changes outputs).

Usage:
    with recording() as rec:          # one search request
        with llm_role("analyze"):
            llm.chat(...)             # an instrumented client records the call
    rec.summary()

`instrumented(client)` wraps any LLM client: every chat/stream_chat call is
timed and, when a recorder is active, appended to it with the current role,
model, token counts reported by the provider (Ollama: prompt_eval_count /
eval_count; otherwise marked "unknown"), load time (cold start), time to first
token for streams, and error/timeout flags. Prompt and output text are
never recorded.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("askito.metrics")

_recorder: contextvars.ContextVar["CallRecorder | None"] = contextvars.ContextVar(
    "askito_recorder", default=None)
_role: contextvars.ContextVar[str] = contextvars.ContextVar("askito_llm_role", default="other")

COLD_LOAD_MS = 1000.0  # load_duration above this means the model was (re)loaded


@dataclass
class CallRecorder:
    calls: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def add(self, rec: dict) -> None:
        self.calls.append(rec)

    def note(self, key: str, value) -> None:
        """Non-LLM facts about the request (e.g. context trimming counts)."""
        self.notes[key] = value

    def summary(self) -> dict:
        by_role: dict[str, dict] = {}
        for c in self.calls:
            r = by_role.setdefault(c["role"], {"calls": 0, "ms": 0.0})
            r["calls"] += 1
            r["ms"] = round(r["ms"] + c.get("ms", 0.0), 1)
        known = [c for c in self.calls if c.get("prompt_tokens") is not None]
        return {
            "calls": len(self.calls),
            "total_ms": round(sum(c.get("ms", 0.0) for c in self.calls), 1),
            "by_role": by_role,
            "models": sorted({c["model"] for c in self.calls if c.get("model")}),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in known) if known else None,
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in known)
            if known else None,
            "tokens_reported": len(known) == len(self.calls),
            "errors": sum(1 for c in self.calls if c.get("error")),
            "timeouts": sum(1 for c in self.calls if c.get("timeout")),
            "context_overflows": sum(1 for c in self.calls if c.get("context_overflow")),
            "cold_loads": sum(1 for c in self.calls if c.get("cold")),
            "notes": self.notes,
            "detail": self.calls,
        }


@contextmanager
def recording():
    rec = CallRecorder()
    token = _recorder.set(rec)
    try:
        yield rec
    finally:
        _recorder.reset(token)


@contextmanager
def llm_role(role: str):
    token = _role.set(role)
    try:
        yield
    finally:
        _role.reset(token)


def current_recorder() -> CallRecorder | None:
    return _recorder.get()


def _usage_fields(usage: dict) -> dict:
    out = {"tokens": "reported" if usage.get("prompt_tokens") is not None else "unknown"}
    for k in ("prompt_tokens", "completion_tokens", "prompt_eval_ms", "eval_ms", "load_ms",
              "done_reason", "context_overflow"):
        if k in usage:
            out[k] = usage[k]
    if usage.get("model"):
        out["model"] = usage["model"]
    out["cold"] = (usage.get("load_ms") or 0) > COLD_LOAD_MS
    return out


class InstrumentedLLM:
    """Transparent wrapper: same interface and outputs as the wrapped client."""

    askito_instrumented = True

    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name):  # ping, warmup, last_usage, capabilities, ...
        return getattr(self.inner, name)

    def _base(self, kw: dict) -> dict:
        return {"role": _role.get(), "model": kw.get("model") or "",
                "json_mode": bool(kw.get("json_mode")), "max_tokens": kw.get("max_tokens")}

    def _finish(self, rec: dict, t0: float, exc: BaseException | None = None) -> None:
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        usage = getattr(self.inner, "last_usage", None) or {}
        if usage and exc is None:
            rec.update(_usage_fields(usage))
        else:
            rec.setdefault("tokens", "unknown")
        if exc is not None:
            rec["error"] = exc.__class__.__name__
            rec["timeout"] = isinstance(exc, httpx.TimeoutException) or \
                getattr(exc, "code", None) in ("model_timeout", "model_load_timeout")
            if getattr(exc, "code", None):
                rec["error_code"] = exc.code
        recorder = _recorder.get()
        if recorder is not None:
            recorder.add(rec)

    def chat(self, messages, **kw):
        rec, t0 = self._base(kw), time.perf_counter()
        if hasattr(self.inner, "last_usage"):
            self.inner.last_usage = {}
        try:
            out = self.inner.chat(messages, **kw)
        except Exception as exc:
            self._finish(rec, t0, exc)
            raise
        self._finish(rec, t0)
        return out

    def stream_chat(self, messages, **kw):
        rec, t0 = self._base(kw), time.perf_counter()
        if hasattr(self.inner, "last_usage"):
            self.inner.last_usage = {}
        try:
            for piece in self.inner.stream_chat(messages, **kw):
                if "ttft_ms" not in rec:
                    rec["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                yield piece
        except Exception as exc:
            self._finish(rec, t0, exc)
            raise
        self._finish(rec, t0)


def instrumented(client):
    """Wrap a client once (wrappers that already delegate to an instrumented
    client set `askito_instrumented = True` and are returned as-is)."""
    if client is None or getattr(type(client), "askito_instrumented", False):
        return client
    return InstrumentedLLM(client)


def log_request(kind: str, metrics: dict) -> None:
    """One structured log line per request: timings and counts, no content."""
    llm = metrics.get("llm", {})
    slim = {k: v for k, v in metrics.items() if k != "llm"}
    slim["llm"] = {k: v for k, v in llm.items() if k != "detail"}
    log.info("%s %s", kind, json.dumps(slim, default=str))
