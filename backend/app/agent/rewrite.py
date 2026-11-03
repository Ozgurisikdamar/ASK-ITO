"""LLM query rewriting (spec §5.1 hybrid_rerank_rewrite, §6 step 2)."""
from __future__ import annotations

import json
import logging
import re

from app.agent.prompts import load
from app.llm.client import LLMClient, get_llm_client

log = logging.getLogger("askito.rewrite")


def rewrite_query(query: str, llm: LLMClient | None = None) -> str:
    from app.config import settings
    from app.llm.metrics import instrumented, llm_role

    from app.agent import tool_cache

    llm = instrumented(llm or get_llm_client())
    prompt = load("rewrite").replace("{question}", query)

    def compute() -> str | None:
        with llm_role("rewrite"):
            raw = llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0,
                model=settings.model_for("agent"), json_mode=True, max_tokens=120,
            )
        return _parse(raw)

    try:
        rewritten, _hit = tool_cache.get_or_call("rewrite", "agent", prompt, 120, compute)
        return rewritten or query
    except Exception as exc:  # noqa: BLE001  (rewriting is optional: search the original)
        log.warning("query rewrite failed (%s: %s); searching the original query",
                    exc.__class__.__name__, str(exc)[:200])
        return query


def _parse(raw: str) -> str | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    q = obj.get("query")
    return q.strip() if isinstance(q, str) and q.strip() else None
