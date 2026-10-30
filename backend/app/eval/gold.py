"""Resolve gold document/chunk ids from the DB for test-set questions."""
from __future__ import annotations

import psycopg

from app.config import settings


def resolve_doc_ids(titles: list[str]) -> list[int]:
    """Map document titles to ids (exact first, then case-insensitive)."""
    ids: list[int] = []
    with psycopg.connect(settings.database_url) as conn:
        for title in titles:
            row = conn.execute(
                "SELECT id FROM documents WHERE title = %s", (title,)
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT id FROM documents WHERE lower(title) = lower(%s) LIMIT 1", (title,)
                ).fetchone()
            if row:
                ids.append(int(row[0]))
    return ids


def best_chunks(doc_ids: list[int], query: str, k: int = 2) -> list[int]:
    """Top-k chunk ids within the given docs by BM25-ish full-text rank."""
    if not doc_ids or not query.strip():
        return []
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            """
            SELECT id FROM chunks
            WHERE doc_id = ANY(%s) AND tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (doc_ids, query, query, k),
        ).fetchall()
        if not rows:
            # fallback: first chunks of the docs
            rows = conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ANY(%s) ORDER BY doc_id, ord LIMIT %s",
                (doc_ids, k),
            ).fetchall()
    return [int(r[0]) for r in rows]
