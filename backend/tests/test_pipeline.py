"""Integration tests for the retrieval pipeline and /api/search.

Skipped automatically when the index/DB is not built (so `make test` works on a
fresh checkout). Uses the mock LLM (set in conftest) so no model/network is
needed for answer generation; retrieval and reranking use the real models.
"""
import psycopg
import pytest

from app.config import settings
from app.index import bm25_index
from app.retrieval import pipeline
from app.retrieval.service import run_search


def _index_ready() -> bool:
    if not bm25_index.exists():
        return False
    try:
        with psycopg.connect(settings.database_url) as c:
            return c.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _index_ready(), reason="index/DB not built (run make ingest + make index)"
)

Q = "2024 Nobel Prize physics neural networks Hopfield Hinton"
SCHEMA_KEYS = {"mode", "answer", "citations", "sources", "trace", "rewritten_query", "timings"}


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "hybrid_rerank"])
def test_retrieval_modes_return_schema(mode):
    res = run_search(Q, mode=mode, domain="all", final_k=5)
    assert SCHEMA_KEYS <= set(res)
    assert res["mode"] == mode
    assert 1 <= len(res["sources"]) <= 5
    for s in res["sources"]:
        assert {"chunk_id", "doc_title", "section", "url", "snippet", "domain", "score"} <= set(s)
    assert isinstance(res["answer"], str) and res["answer"]


def test_closed_book_has_no_sources():
    res = run_search(Q, mode="closed_book")
    assert res["sources"] == []
    assert res["trace"] == []
    assert isinstance(res["answer"], str) and res["answer"]


def test_hybrid_rerank_trace_has_all_four_steps():
    res = run_search(Q, mode="hybrid_rerank", final_k=5)
    steps = [s["step"] for s in res["trace"]]
    for expected in ("bm25", "vector", "rrf", "rerank"):
        assert expected in steps, f"missing trace step {expected}: {steps}"


def test_document_diversity_cap():
    res = run_search(Q, mode="hybrid_rerank", final_k=5)
    from collections import Counter
    with psycopg.connect(settings.database_url) as c:
        ids = [s["chunk_id"] for s in res["sources"]]
        docs = [r[0] for r in c.execute(
            "SELECT doc_id FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()]
    assert max(Counter(docs).values()) <= settings.max_chunks_per_doc


def test_domain_filter():
    res = run_search("deep learning transformers", mode="hybrid", domain="ai", final_k=5)
    assert all(s["domain"] == "ai" for s in res["sources"])


def test_rewrite_mode_has_rewrite_step():
    res = run_search(Q, mode="hybrid_rerank_rewrite", final_k=5)
    steps = [s["step"] for s in res["trace"]]
    assert steps[0] == "rewrite"
    for expected in ("bm25", "vector", "rrf", "rerank"):
        assert expected in steps
