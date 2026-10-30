"""Core retrieval primitives (spec §5 steps 1-3).

BM25 and dense vector search, fused with Reciprocal Rank Fusion. The full
pipeline (reranker, modes, document diversity, API shape) is built in M4; this
module provides the building blocks and lets M3 demonstrate that all three
retrieval modes return results.
"""
from __future__ import annotations

from app.config import settings
from app.index import bm25_index
from app.index.embed import embed_query
from app.retrieval.vectorstore import PgVectorStore

_vector_store = PgVectorStore()
_bm25: bm25_index.Bm25Searcher | None = None


def _bm25_searcher() -> bm25_index.Bm25Searcher:
    global _bm25
    if _bm25 is None:
        _bm25 = bm25_index.Bm25Searcher()
    return _bm25


def bm25_search(query: str, k: int | None = None) -> list[tuple[int, float]]:
    return _bm25_searcher().search(query, k or settings.bm25_top_k)


def vector_search(
    query: str, k: int | None = None, domain: str | None = None
) -> list[tuple[int, float]]:
    emb = embed_query(query)
    return _vector_store.search(emb, k or settings.vector_top_k, domain=domain)


def rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]], k: int | None = None
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Returns [(chunk_id, rrf_score)] best-first."""
    k = k or settings.rrf_k
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (cid, _score) in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str, k: int | None = None, domain: str | None = None
) -> list[tuple[int, float]]:
    bm = bm25_search(query, settings.bm25_top_k)
    if domain not in (None, "all"):
        bm = _filter_domain(bm, domain)
    vec = vector_search(query, settings.vector_top_k, domain=domain)
    fused = rrf_fuse([bm, vec])
    return fused[: (k or settings.final_k)]


def _filter_domain(pairs: list[tuple[int, float]], domain: str) -> list[tuple[int, float]]:
    if not pairs:
        return pairs
    import psycopg

    ids = [cid for cid, _ in pairs]
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE id = ANY(%s) AND domain = %s", (ids, domain)
        ).fetchall()
    keep = {int(r[0]) for r in rows}
    return [(cid, s) for cid, s in pairs if cid in keep]
