"""Deterministic reranker test: sorting/id-mapping with a stubbed model."""
from app.retrieval import reranker


class _StubModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs, show_progress_bar=False, **kwargs):
        return self._scores


def test_rerank_sorts_by_score_and_keeps_ids(monkeypatch):
    # chunk 101 -> 0.2, 202 -> 0.9, 303 -> 0.5  =>  order 202, 303, 101
    monkeypatch.setattr(reranker, "_model", lambda: _StubModel([0.2, 0.9, 0.5]))
    passages = [(101, "a"), (202, "b"), (303, "c")]
    out = reranker.rerank("q", passages)
    assert [cid for cid, _ in out] == [202, 303, 101]
    assert out[0][1] == 0.9


def test_rerank_empty():
    assert reranker.rerank("q", []) == []
