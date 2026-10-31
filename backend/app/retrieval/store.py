"""In-memory chunk/document metadata (read-mostly; the corpus is a frozen snapshot).

Loaded once at startup (and again whenever `make index` rebuilds the index), so
domain filtering, document-diversity lookups, reranker passage texts and result
hydration never touch the database. A search then needs a single DB round trip:
the pgvector query.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from app.db.connection import get_connection
from app.index.bm25_index import index_version


@dataclass(slots=True, frozen=True)
class ChunkMeta:
    chunk_id: int
    doc_id: int
    domain: str
    section: str | None
    text: str
    doc_title: str
    url: str


class ChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[int, ChunkMeta] = {}
        self._doc_chunks: dict[int, list[int]] = {}
        self._version: str | None = None
        self._lock = threading.Lock()

    # -- loading ---------------------------------------------------------
    def ensure_loaded(self) -> None:
        version = index_version()
        if self._version == version and self._chunks:
            return
        with self._lock:
            if self._version == version and self._chunks:
                return
            self._load(version)

    def _load(self, version: str) -> None:
        chunks: dict[int, ChunkMeta] = {}
        doc_chunks: dict[int, list[int]] = {}
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT c.id, c.doc_id, c.domain, c.section, c.text, d.title, d.url
                   FROM chunks c JOIN documents d ON d.id = c.doc_id
                   ORDER BY c.doc_id, c.ord"""
            ).fetchall()
        for r in rows:
            meta = ChunkMeta(int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], r[6])
            chunks[meta.chunk_id] = meta
            doc_chunks.setdefault(meta.doc_id, []).append(meta.chunk_id)
        self._chunks, self._doc_chunks, self._version = chunks, doc_chunks, version

    # -- lookups ---------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def version(self) -> str | None:
        return self._version

    def get(self, chunk_id: int) -> ChunkMeta | None:
        return self._chunks.get(chunk_id)

    def many(self, chunk_ids: list[int]) -> dict[int, ChunkMeta]:
        out = {}
        for cid in chunk_ids:
            meta = self._chunks.get(cid)
            if meta is not None:
                out[cid] = meta
        return out

    def doc_chunk_ids(self, doc_id: int) -> list[int]:
        return self._doc_chunks.get(doc_id, [])

    def filter_domain(
        self, pairs: list[tuple[int, float]], domain: str | None
    ) -> list[tuple[int, float]]:
        if domain in (None, "all"):
            return pairs
        return [(c, s) for c, s in pairs if (m := self._chunks.get(c)) and m.domain == domain]


store = ChunkStore()
