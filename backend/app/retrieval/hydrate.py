"""Chunk/document details for result display, plus snippet helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.store import store


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
    """Chunk + document details from the in-memory store (no DB round trip)."""
    if not chunk_ids:
        return {}
    store.ensure_loaded()
    return {
        cid: ChunkInfo(m.chunk_id, m.doc_id, m.doc_title, m.section, m.url, m.domain, m.text)
        for cid, m in store.many(chunk_ids).items()
    }


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
