"""Persistent answer cache (table `answer_cache`, migration 0003)."""
from __future__ import annotations

import hashlib
import json

from app.config import settings
from app.db.connection import get_connection


def _model_tag() -> str:
    return f"{settings.llm_mode}:{settings.model_for('answer')}"


def answer_fingerprint(mode: str) -> tuple:
    """Every setting that changes the answer for a given query: the model and its
    generation settings, the prompt version, the evidence cut-off, the context
    budget, and the retrieval fingerprint (which includes the index version)."""
    from app.retrieval import pipeline, service

    return (
        _model_tag(), settings.llm_think, settings.llm_num_ctx, settings.prompt_version,
        settings.overview_max_tokens, settings.answer_max_tokens,
        settings.overview_context_chunks, settings.overview_context_chars,
        settings.answer_context_chars, settings.final_k,
        service.evidence_cutoff_active(mode), settings.fast_min_evidence_score,
        settings.evidence_score_model,
        pipeline.retrieval_fingerprint(mode),
    )


def make_key(query: str, mode: str, domain: str | None) -> str:
    parts = [" ".join(query.lower().split()), mode, domain or "all",
             repr(answer_fingerprint(mode))]
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
