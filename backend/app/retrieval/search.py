"""Core retrieval primitives (spec §5 steps 1-3).

BM25 and dense vector search, fused with Reciprocal Rank Fusion. The domain
filter is applied before fusion on both sides: BM25 draws from a wider pool
and is filtered in memory, the vector query filters in SQL.
"""
from __future__ import annotations

from app.config import settings
from app.index import bm25_index
from app.index.embed import embed_query
from app.retrieval.store import store
from app.retrieval.vectorstore import PgVectorStore

_vector_store = PgVectorStore()


def bm25_search(
    query: str, k: int | None = None, domain: str | None = None
) -> list[tuple[int, float]]:
    k = k or settings.bm25_top_k
    if domain in (None, "all"):
        return bm25_index.get_searcher().search(query, k)
    store.ensure_loaded()
    wide = bm25_index.get_searcher().search(query, max(settings.bm25_pool_k, k))
    return store.filter_domain(wide, domain)[:k]


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
    bm = bm25_search(query, settings.bm25_top_k, domain=domain)
    vec = vector_search(query, settings.vector_top_k, domain=domain)
    return rrf_fuse([bm, vec])[: (k or settings.final_k)]
