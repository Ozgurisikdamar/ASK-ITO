"""Fetch chunk/document details for a set of chunk ids (order preserved)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg

from app.config import settings


@dataclass
class ChunkInfo:
    chunk_id: int
    doc_id: int
    doc_title: str
    section: str | None
    url: str
    domain: str
    text: str


def fetch_chunks(chunk_ids: list[int]) -> dict[int, ChunkInfo]:
    if not chunk_ids:
        return {}
    sql = """
        SELECT c.id, c.doc_id, d.title, c.section, d.url, c.domain, c.text
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE c.id = ANY(%s)
    """
    out: dict[int, ChunkInfo] = {}
    with psycopg.connect(settings.database_url) as conn:
        for r in conn.execute(sql, (chunk_ids,)).fetchall():
            out[int(r[0])] = ChunkInfo(int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], r[6])
    return out


def body_without_prefix(text: str) -> str:
    """Chunk text is stored as "{title} > {section}\\n\\n{body}"; return body."""
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text


def make_snippet(text: str, query: str, max_len: int = 240) -> str:
    body = body_without_prefix(text)
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    low = body.lower()
    pos = -1
    for t in terms:
        pos = low.find(t)
        if pos != -1:
            break
    if pos == -1:
        return body[:max_len].strip() + ("…" if len(body) > max_len else "")
    start = max(0, pos - max_len // 3)
    snippet = body[start : start + max_len].strip()
    return ("…" if start > 0 else "") + snippet + ("…" if start + max_len < len(body) else "")
