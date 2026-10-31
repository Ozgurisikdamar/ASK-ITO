"""Vector store behind an interface (spec §1.1) so the backend is swappable.

Only pgvector is implemented. Cosine distance (`<=>`) is used; similarity is
reported as 1 - distance.
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.db.connection import get_connection


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class VectorStore(Protocol):
    def search(
        self, query_embedding: list[float], k: int, domain: str | None = None
    ) -> list[tuple[int, float]]:
        """Return [(chunk_id, similarity)] best-first."""
        ...


class PgVectorStore:
    def search(
        self, query_embedding: list[float], k: int, domain: str | None = None
    ) -> list[tuple[int, float]]:
        vec = to_pgvector(query_embedding)
        where = "" if domain in (None, "all") else "WHERE c.domain = %(domain)s"
        sql = f"""
            SELECT c.id, 1 - (c.embedding <=> %(vec)s::vector) AS score
            FROM chunks c
            {where}
            ORDER BY c.embedding <=> %(vec)s::vector
            LIMIT %(k)s
        """
        params = {"vec": vec, "k": k, "domain": domain}
        # HNSW returns at most ef_search rows (default 40); raise it so top-k is honoured.
        ef = max(settings.hnsw_ef_search, k)
        with get_connection() as conn, conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
            rows = conn.execute(sql, params).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]
