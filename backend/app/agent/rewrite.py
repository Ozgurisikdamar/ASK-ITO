"""LLM query rewriting (spec §5.1 hybrid_rerank_rewrite, §6 step 2)."""
from __future__ import annotations

import json
import re

from app.agent.prompts import load
from app.llm.client import LLMClient, get_llm_client


def rewrite_query(query: str, llm: LLMClient | None = None) -> str:
    llm = llm or get_llm_client()
    prompt = load("rewrite").replace("{question}", query)
    try:
        from app.config import settings

        raw = llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0,
            model=settings.model_for("agent"), json_mode=True, max_tokens=120,
        )
        rewritten = _parse(raw)
        return rewritten or query
    except Exception:  # noqa: BLE001
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
