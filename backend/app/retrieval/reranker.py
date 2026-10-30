"""Cross-encoder reranker (spec §5 step 4, mandatory component).

Scores (query, passage) pairs with a cross-encoder (bge-reranker-base by
default). This is a required stage for the hybrid_rerank* and agentic modes; if
the model cannot be loaded the error propagates (never silently skipped), and
/api/health reports the reranker separately.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, device="cpu")


def rerank(query: str, passages: list[tuple[int, str]]) -> list[tuple[int, float]]:
    """passages: [(chunk_id, text)] -> [(chunk_id, score)] sorted best-first."""
    if not passages:
        return []
    pairs = [(query, text) for _, text in passages]
    scores = _model().predict(pairs, show_progress_bar=False)
    ranked = sorted(
        ((cid, float(s)) for (cid, _), s in zip(passages, scores)),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def available() -> bool:
    try:
        _model()
        return True
    except Exception:  # noqa: BLE001
        return False
