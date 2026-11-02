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


def test_cross_encoder_loaded_once_per_process(monkeypatch):
    """The cross-encoder is a process-wide singleton: never re-loaded per request."""
    import sentence_transformers

    built = []

    class FakeCE:
        def __init__(self, name, device=None, max_length=None):
            built.append((name, device, max_length))

        def predict(self, pairs, **kwargs):
            return [float(len(b)) for _, b in pairs]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCE)
    reranker._model.cache_clear()
    try:
        for _ in range(3):
            out = reranker.rerank("q", [(1, "aa"), (2, "aaaa")])
            assert [c for c, _ in out] == [2, 1]
        assert len(built) == 1
        name, device, max_length = built[0]
        assert name == reranker.settings.reranker_model and device == "cpu"
        assert max_length == reranker.settings.rerank_max_length
        assert reranker.status()["loaded"] is True
    finally:
        reranker._model.cache_clear()
