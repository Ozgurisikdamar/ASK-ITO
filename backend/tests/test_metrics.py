from app.eval import metrics


def test_recall_at_k():
    assert metrics.recall_at_k([10, 20, 30, 40, 50], {30, 99}, 5) == 0.5
    assert metrics.recall_at_k([10, 20], {10, 20}, 2) == 1.0
    assert metrics.recall_at_k([1, 2, 3], {9}, 3) == 0.0
    assert metrics.recall_at_k([1], set(), 5) is None  # no gold -> N/A


def test_mrr_at_k():
    assert metrics.mrr_at_k([10, 20, 30], {30}, 10) == 1 / 3
    assert metrics.mrr_at_k([30, 10], {30}, 10) == 1.0
    assert metrics.mrr_at_k([1, 2], {9}, 10) == 0.0


def test_ndcg_at_k():
    assert abs(metrics.ndcg_at_k([1, 2, 3], {1}, 10) - 1.0) < 1e-9
    # relevant item lower down -> < 1
    assert metrics.ndcg_at_k([2, 1], {1}, 10) < 1.0
    assert metrics.ndcg_at_k([1], set(), 10) is None


def test_retrieval_metrics_shape():
    m = metrics.retrieval_metrics([1, 2, 3], [11, 12], {1}, {11})
    for k in ("chunk_recall@5", "doc_recall@5", "chunk_mrr@10", "doc_ndcg@10"):
        assert k in m


def test_aggregate_ignores_none():
    agg = metrics.aggregate([{"a": 1.0, "b": None}, {"a": 0.0, "b": 0.5}])
    assert agg["a"] == 0.5
    assert agg["b"] == 0.5


def test_percentiles():
    p = metrics.percentiles([10, 20, 30, 40, 100])
    assert p["p50"] == 30 and p["p95"] >= 40


def test_refusal_correct():
    assert metrics.refusal_correct("I could not find this in the sources.") is True
    assert metrics.refusal_correct("Here is the answer.") is False
