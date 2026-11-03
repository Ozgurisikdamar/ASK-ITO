"""Which model weights actually serve a role: tag, weight quantization, digest.

The model tag is configured in one place (LLM_DEFAULT_MODEL / LLM_<ROLE>_MODEL);
this module asks the runtime what that tag resolves to, so caches and reports
never confuse e.g. qwen3.5:9b (Q4_K_M) with qwen3.5:9b-q8_0 (Q8_0). Weight
quantization is unrelated to the KV-cache type (OLLAMA_KV_CACHE_TYPE).
"""
from __future__ import annotations

import threading
import time

from app.config import settings

_TTL_S = 60.0
_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_lock = threading.Lock()


def clear() -> None:
    with _lock:
        _cache.clear()


def served_model(role: str = "answer") -> dict:
    """{"name", "quantization", "digest", ...}; unknown fields are None."""
    name = settings.model_for(role)
    key = (settings.llm_mode, name)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _TTL_S:
            return dict(hit[1])
    ident = {"name": name, "quantization": None, "digest": None}
    if settings.llm_mode == "ollama":
        from app.llm.client import get_llm_client

        fn = getattr(get_llm_client(), "model_identity", None)
        if fn is not None:
            try:
                ident = {**ident, **fn(name)}
            except Exception:  # noqa: BLE001  (runtime unreachable: identity unknown)
                pass
    with _lock:
        _cache[key] = (now, ident)
    return dict(ident)


def model_tag(role: str = "answer") -> str:
    """Cache-key form: mode:name:quantization:digest-prefix ("?" when unknown,
    which never matches an entry written while the identity was known)."""
    ident = served_model(role)
    return ":".join([settings.llm_mode, ident["name"], ident.get("quantization") or "?",
                     (ident.get("digest") or "?")[:12]])
