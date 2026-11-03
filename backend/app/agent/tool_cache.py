"""Cache for the agent's deterministic LLM tool calls (analyse, rewrite, sufficiency).

These calls run at temperature 0 with think off and return small JSON objects,
so the same model + full prompt + generation limits gives the same result. The
key is built from all of those (the full prompt text includes the template, the
question and, for sufficiency, the retrieved context), so a change to any of
them is a miss. Only successfully parsed results are stored; failures are
retried next time. Nothing is cached when thinking is on.

The values are tool outputs, not conversation state: nothing per-user or
per-session is stored, so there is no cross-session leakage. L1 is an
in-process LRU; L2 is the optional local Redis (kind "tool", REDIS_TTL_TOOL_S).
Hits are counted per role on the request's CallRecorder (notes.tool_cache_hits).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

from app.config import settings

TOOL_CACHE_VERSION = "1"
_L1_SIZE = 256
_l1: OrderedDict[str, object] = OrderedDict()
_lock = threading.Lock()


def clear() -> None:
    with _lock:
        _l1.clear()


def enabled() -> bool:
    return settings.agent_tool_cache and not settings.llm_think


def tool_key(role: str, model_role: str, prompt: str, max_tokens: int) -> str:
    from app.llm.identity import model_tag
    from app.redis_cache import hash_key

    return hash_key((TOOL_CACHE_VERSION, role, model_tag(model_role), settings.llm_num_ctx,
                     0.0, max_tokens, prompt))


def _note_hit(role: str, tier: str) -> None:
    from app.llm.metrics import current_recorder

    rec = current_recorder()
    if rec is None:
        return
    hits = rec.notes.setdefault("tool_cache_hits", {})
    hits[role] = hits.get(role, 0) + 1
    tiers = rec.notes.setdefault("tool_cache_tiers", {})
    tiers[tier] = tiers.get(tier, 0) + 1


def get_or_call(role: str, model_role: str, prompt: str, max_tokens: int,
                compute: Callable[[], object | None]):
    """Return (value, cached) where value is compute()'s JSON-able result."""
    if not enabled():
        return compute(), False
    key = tool_key(role, model_role, prompt, max_tokens)
    with _lock:
        if key in _l1:
            _l1.move_to_end(key)
            _note_hit(role, "l1")
            return _l1[key], True
    from app.redis_cache import cache

    value = cache.get_json("tool", key) if cache.enabled else None
    if value is not None:
        _put_l1(key, value)
        _note_hit(role, "l2")
        return value, True
    value = compute()
    if value is not None:
        _put_l1(key, value)
        if cache.enabled:
            cache.set_json("tool", key, value)
    return value, False


def _put_l1(key: str, value) -> None:
    with _lock:
        _l1[key] = value
        _l1.move_to_end(key)
        while len(_l1) > _L1_SIZE:
            _l1.popitem(last=False)
