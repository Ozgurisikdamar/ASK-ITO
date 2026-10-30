from app.retrieval.search import rrf_fuse


def test_rrf_rewards_agreement_and_rank():
    # chunk 1 is top of both lists -> highest; chunk 2 appears in both lower;
    # chunk 3 only once.
    bm25 = [(1, 9.0), (2, 8.0), (3, 7.0)]
    vec = [(1, 0.9), (2, 0.8), (4, 0.7)]
    fused = rrf_fuse([bm25, vec], k=60)
    order = [cid for cid, _ in fused]
    assert order[0] == 1  # agreed top rank
    assert order.index(2) < order.index(3)  # 2 in both beats 3 in one
    assert set(order) == {1, 2, 3, 4}


def test_rrf_uses_rank_not_score():
    # scores are tiny but rank is what matters
    a = [(10, 0.01), (20, 0.001)]
    b = [(20, 100.0), (10, 50.0)]
    fused = dict(rrf_fuse([a, b], k=60))
    # both appear at ranks 1 and 2 across the two lists -> equal rrf score
    assert abs(fused[10] - fused[20]) < 1e-9
