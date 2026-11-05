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
    query: str, k: int | None = None, domain: str | None = None,
    embedding: list[float] | None = None,
) -> list[tuple[int, float]]:
    """pgvector search; pass a precomputed `embedding` to avoid embedding twice."""
    emb = embedding if embedding is not None else embed_query(query)
    return _vector_store.search(emb, k or settings.vector_top_k, domain=domain)


def rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]], k: int | None = None,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Returns [(chunk_id, rrf_score)] best-first.
    `weights` scales each input list's contribution (default 1.0 each)."""
    k = k or settings.rrf_k
    scores: dict[int, float] = {}
    for i, ranked in enumerate(ranked_lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, (cid, _score) in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _rrf_weights() -> list[float]:
    return [settings.rrf_bm25_weight, settings.rrf_vector_weight]


def hybrid_search(
    query: str, k: int | None = None, domain: str | None = None
) -> list[tuple[int, float]]:
    bm = bm25_search(query, settings.bm25_top_k, domain=domain)
    vec = vector_search(query, settings.vector_top_k, domain=domain)
    return rrf_fuse([bm, vec], weights=_rrf_weights())[: (k or settings.final_k)]
