"""Persistent answer cache (table `answer_cache`, migration 0003)."""
from __future__ import annotations

import hashlib
import json

from app.config import settings
from app.db.connection import get_connection
from app.index.bm25_index import index_version


def _model_tag() -> str:
    return f"{settings.llm_mode}:{settings.model_for('answer')}"


def make_key(query: str, mode: str, domain: str | None) -> str:
    parts = [
        " ".join(query.lower().split()), mode, domain or "all", _model_tag(),
        settings.prompt_version, index_version(),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def get(query: str, mode: str, domain: str | None) -> dict | None:
    if not settings.answer_cache_enabled:
        return None
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT answer, citations, chunk_ids FROM answer_cache WHERE key = %s",
                (make_key(query, mode, domain),),
            ).fetchone()
    except Exception:  # noqa: BLE001  (table missing before migrate: just miss)
        return None
    if not row:
        return None
    return {"answer": row[0], "citations": row[1], "chunk_ids": list(row[2])}


def put(query: str, mode: str, domain: str | None, answer: str,
        citations: list[dict], chunk_ids: list[int]) -> None:
    if not settings.answer_cache_enabled:
        return
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO answer_cache
                     (key, query, mode, domain, model, answer, citations, chunk_ids)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (key) DO UPDATE SET
                     answer = EXCLUDED.answer, citations = EXCLUDED.citations,
                     chunk_ids = EXCLUDED.chunk_ids, created_at = now()""",
                (make_key(query, mode, domain), query, mode, domain or "all", _model_tag(),
                 answer, json.dumps(citations), chunk_ids),
            )
    except Exception:  # noqa: BLE001
        return
