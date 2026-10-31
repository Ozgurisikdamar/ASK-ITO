"""Pydantic schemas for the agent's structured LLM outputs (spec §6)."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.llm.client import LLMClient

_JSON = re.compile(r"\{.*\}", re.DOTALL)


class Analysis(BaseModel):
    qtype: Literal["simple", "multi_hop", "comparison"] = "simple"
    subquestions: list[str] = Field(default_factory=list)


class Sufficiency(BaseModel):
    sufficient: bool
    missing: str = ""
    next_query: str | None = None


def extract_json(raw: str) -> dict | None:
    m = _JSON.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def call_json(
    llm: LLMClient, prompt: str, model_cls: type[BaseModel], retries: int = 1,
    role: str = "agent", max_tokens: int = 400,
):
    """Call the LLM (JSON mode, role model) and validate the output. Returns a
    model instance or None. Retries once on invalid JSON (spec §6 step 1)."""
    from app.config import settings

    for _ in range(retries + 1):
        raw = llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0,
            model=settings.model_for(role), json_mode=True, max_tokens=max_tokens,
        )
        obj = extract_json(raw)
        if obj is not None:
            try:
                return model_cls(**obj)
            except ValidationError:
                continue
    return None
