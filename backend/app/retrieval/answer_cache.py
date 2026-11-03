"""Persistent answer cache (table `answer_cache`, migration 0003), with an
optional Redis front cache (REDIS_ENABLED + REDIS_ANSWER_FRONT_CACHE).

PostgreSQL stays the durable store; Redis only saves the round trip and is
filled on write and on a PG hit. Both use the same key, which covers the
model tag (name + quantization/digest), generation settings, prompt version,
evidence cut-off and the retrieval fingerprint (index version included).
"""
from __future__ import annotations

import hashlib
import json

from app.config import settings
from app.db.connection import get_connection


def _model_tag() -> str:
    """Model name + weight quantization + manifest digest (app.llm.identity)."""
    from app.llm.identity import model_tag

    return model_tag("answer")


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


def _sources_match(stored: list[int], source_ids: list[int] | None) -> bool:
    """A cached answer cites passages by number, so it is only valid for the
    exact same ordered sources (None = caller does not know them yet)."""
    return source_ids is None or [int(c) for c in stored] == [int(c) for c in source_ids]


def get(query: str, mode: str, domain: str | None,
        source_ids: list[int] | None = None) -> dict | None:
    """Redis front cache (when enabled), then PostgreSQL. The hit carries
    `cache_kind` = "redis" | "pg" for telemetry; a hit whose stored sources
    differ from `source_ids` is treated as a miss."""
    if not settings.answer_cache_enabled:
        return None
    key = make_key(query, mode, domain)
    from app.redis_cache import cache

    if cache.enabled and settings.redis_answer_front_cache:
        hit = cache.get_json("ans", key)
        if isinstance(hit, dict) and "answer" in hit and \
                _sources_match(hit.get("chunk_ids") or [], source_ids):
            return {"answer": hit["answer"], "citations": hit.get("citations") or [],
                    "chunk_ids": list(hit.get("chunk_ids") or []), "cache_kind": "redis"}
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT answer, citations, chunk_ids FROM answer_cache WHERE key = %s",
                (key,),
            ).fetchone()
    except Exception:  # noqa: BLE001  (table missing before migrate: just miss)
        return None
    if not row or not _sources_match(list(row[2]), source_ids):
        return None
    out = {"answer": row[0], "citations": row[1], "chunk_ids": list(row[2]),
           "cache_kind": "pg"}
    _front_put(key, out)
    return out


def _front_put(key: str, entry: dict) -> None:
    from app.redis_cache import cache

    if cache.enabled and settings.redis_answer_front_cache:
        cache.set_json("ans", key, {"answer": entry["answer"],
                                    "citations": entry["citations"],
                                    "chunk_ids": entry["chunk_ids"]})


def put(query: str, mode: str, domain: str | None, answer: str,
        citations: list[dict], chunk_ids: list[int]) -> None:
    if not settings.answer_cache_enabled:
        return
    key = make_key(query, mode, domain)
    _front_put(key, {"answer": answer, "citations": citations, "chunk_ids": list(chunk_ids)})
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO answer_cache
                     (key, query, mode, domain, model, answer, citations, chunk_ids)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (key) DO UPDATE SET
                     answer = EXCLUDED.answer, citations = EXCLUDED.citations,
                     chunk_ids = EXCLUDED.chunk_ids, created_at = now()""",
                (key, query, mode, domain or "all", _model_tag(),
                 answer, json.dumps(citations), chunk_ids),
            )
    except Exception:  # noqa: BLE001
        return
