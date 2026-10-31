"""Disk cache for LLM calls during eval (spec §7). Keyed by model + messages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.config import settings
from app.llm.client import LLMClient, get_llm_client

CACHE_DIR = Path(__file__).resolve().parents[2] / "eval" / ".llm_cache"


class CachedLLM:
    def __init__(self, llm: LLMClient | None = None, enabled: bool | None = None) -> None:
        self._llm = llm or get_llm_client()
        self.enabled = settings.llm_cache_enabled if enabled is None else enabled
        if self.enabled:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _key(self, messages, kw) -> str:
        blob = json.dumps(
            {"m": messages, "kw": {k: v for k, v in kw.items() if k != "cancel"},
             "model": kw.get("model") or settings.model_for("answer")},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages, **kw) -> str:
        if not self.enabled:
            return self._llm.chat(messages, **kw)
        path = CACHE_DIR / f"{self._key(messages, kw)}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        resp = self._llm.chat(messages, **kw)
        path.write_text(resp, encoding="utf-8")
        return resp

    def stream_chat(self, messages, **kw):
        yield self.chat(messages, **{k: v for k, v in kw.items() if k != "cancel"})

    def ping(self):
        return self._llm.ping()
