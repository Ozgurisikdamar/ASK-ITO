"""Cross-encoder reranker (spec §5 step 4, mandatory component).

Scores (query, passage) pairs with a cross-encoder (model from config). This is
a required stage for the hybrid_rerank*, overview and agentic modes; if the
model cannot be loaded the error propagates (never silently skipped), and
/api/health reports the load state.

CPU cost scales with pairs x tokens, so pairs are truncated to
`rerank_max_length` tokens and only the top `rerank_pool` candidates are scored.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.ml import configure_torch, inference_lock

_state = {"loaded": False, "error": None}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    configure_torch()
    try:
        model = CrossEncoder(
            settings.reranker_model, device="cpu", max_length=settings.rerank_max_length
        )
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{exc.__class__.__name__}: {exc}"
        raise
    _state["loaded"] = True
    return model


def rerank(query: str, passages: list[tuple[int, str]]) -> list[tuple[int, float]]:
    """passages: [(chunk_id, text)] -> [(chunk_id, score)] sorted best-first."""
    if not passages:
        return []
    model = _model()
    pairs = [(query, text) for _, text in passages]
    with inference_lock:
        scores = model.predict(
            pairs, batch_size=settings.rerank_batch_size, show_progress_bar=False
        )
    return sorted(
        ((cid, float(s)) for (cid, _), s in zip(passages, scores)),
        key=lambda x: x[1],
        reverse=True,
    )


def warmup() -> None:
    rerank("warm up", [(0, "warm up passage")])


def status() -> dict:
    return {"model": settings.reranker_model, "loaded": _state["loaded"], "error": _state["error"]}


def available() -> bool:
    try:
        _model()
        return True
    except Exception:  # noqa: BLE001
        return False
