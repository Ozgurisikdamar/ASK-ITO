"""Fast RAG path: query routing, LLM-call budget, hybrid channels, citations,
abstention, and no RAGAS in the serving path.

Unit tests need nothing; integration tests use the real index + reranker and a
scripted/mocked LLM (skipped when the index is absent)."""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from app.config import settings
from app.retrieval import pipeline, service
from tests.test_agent import _index_ready

BACKEND = Path(__file__).resolve().parents[1]


# -- unit ----------------------------------------------------------------------

def test_default_mode_is_fast_answer_and_agentic_is_opt_in():
    assert settings.default_mode == "overview"
    assert "agentic" not in pipeline.RETRIEVAL_MODES  # never part of the fast pipeline
    from app.api.search import SearchRequest

    assert SearchRequest(query="x").mode == "overview"


def test_reranker_switch_only_affects_fast_modes(monkeypatch):
    monkeypatch.setattr(settings, "fast_rerank", True)
    assert pipeline.uses_reranker("overview") and pipeline.uses_reranker("search")
    monkeypatch.setattr(settings, "fast_rerank", False)
    assert not pipeline.uses_reranker("overview") and not pipeline.uses_reranker("search")
    assert pipeline.uses_reranker("hybrid_rerank")  # evaluation modes keep their reranker
    assert not pipeline.uses_reranker("hybrid")


def test_fast_top_k_overrides_channel_pool_for_fast_modes_only(monkeypatch):
    monkeypatch.setattr(settings, "fast_top_k", 15)
    assert pipeline.channel_k("overview") == (15, 15)
    assert pipeline.channel_k("hybrid") == (settings.bm25_top_k, settings.vector_top_k)
    monkeypatch.setattr(settings, "fast_top_k", 0)
    assert pipeline.channel_k("search") == (settings.bm25_top_k, settings.vector_top_k)


def test_evidence_filter_uses_reranked_scores_only(monkeypatch):
    monkeypatch.setattr(settings, "fast_rerank", True)
    ranked = [(1, 0.9), (2, 0.4), (3, 0.2), (4, 0.016)]  # last one: RRF tail score
    monkeypatch.setattr(settings, "fast_min_evidence_score", 0.3)
    assert service.evidence_filter(ranked, "overview", n_reranked=3) == [(1, 0.9), (2, 0.4)]
    monkeypatch.setattr(settings, "fast_min_evidence_score", 0.01)
    assert (4, 0.016) not in service.evidence_filter(ranked, "overview", n_reranked=3)
    monkeypatch.setattr(settings, "fast_min_evidence_score", 0.0)  # off
    assert service.evidence_filter(ranked, "overview", 3) == ranked
    monkeypatch.setattr(settings, "fast_min_evidence_score", 0.3)
    assert service.evidence_filter(ranked, "hybrid_rerank", 3) == ranked  # not a fast mode
    monkeypatch.setattr(settings, "fast_rerank", False)  # no CE scores -> no cut-off
    assert service.evidence_filter(ranked, "overview", 0) == ranked


def test_search_mode_is_not_an_eval_configuration():
    from app.eval.run import ALL_MODES

    assert "search" not in ALL_MODES and "overview" in ALL_MODES


def test_ragas_is_not_imported_by_the_serving_path():
    code = (
        "import os, sys; os.environ['LLM_MODE']='mock'; os.environ['WARMUP_ON_START']='false';"
        "os.environ['PREWARM_FEATURED']='false';"
        "import app.main, app.api.search, app.api.stream, app.retrieval.service;"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('ragas', 'datasets') or m == 'app.eval.ragas_eval')))"
    )
    out = subprocess.run([sys.executable, "-c", "import json;" + code], cwd=BACKEND,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


# -- integration ---------------------------------------------------------------

needs_index = pytest.mark.skipif(not _index_ready(), reason="index/DB not built")


class CitingLLM:
    """Scripted answer generator: cites [1], [2] and a non-existent [9]."""

    def __init__(self):
        self.calls = 0

    def ping(self):
        return True, "scripted"

    def chat(self, messages, **kw):
        self.calls += 1
        return "Transmons are superconducting qubits [1]. They reduce charge noise [2]. " \
               "Invented claim [9]."

    def stream_chat(self, messages, **kw):
        self.calls += 1
        yield from ["Transmons are superconducting qubits [1]. ", "They reduce noise [2]."]


@needs_index
def test_fast_answer_makes_exactly_one_llm_call_and_cites_real_sources():
    pipeline.cache_clear()
    llm = CitingLLM()
    res = service.run_search("transmon qubit", mode="overview", llm=llm)
    m = res["metrics"]
    assert llm.calls == 1 and m["llm"]["calls"] == 1
    assert list(m["llm"]["by_role"]) == ["overview"]  # no analysis/rewrite/sufficiency call
    source_ids = {s["chunk_id"] for s in res["sources"]}
    assert res["citations"] and all(c["chunk_id"] in source_ids for c in res["citations"])
    assert "[9]" not in res["answer"]  # fabricated citation stripped
    for s in res["sources"]:
        assert s["doc_title"] and s["url"] and s["chunk_id"]
    assert {"embedding", "bm25", "vector", "rrf"} <= set(m["retrieval_ms"])
    assert m["models"]["llm"] == settings.model_for("answer")


@needs_index
def test_search_mode_makes_no_llm_call():
    pipeline.cache_clear()
    llm = CitingLLM()
    res = service.run_search("quantum computing", mode="search", llm=llm)
    assert llm.calls == 0 and res["metrics"]["llm"]["calls"] == 0
    assert res["answer"] == "" and res["citations"] == []
    assert res["results"] and res["sources"]
    assert res["metrics"]["models"]["llm"] is None


@needs_index
def test_hybrid_uses_both_channels_fused_with_rrf():
    from app.retrieval.search import bm25_search, rrf_fuse, vector_search

    q = "Nobel Prize in Physics 2024 neural networks"
    cand = pipeline.candidates(q, "overview")
    steps = [s.step for s in cand.trace]
    assert steps[:3] == ["bm25", "vector", "rrf"]
    k_bm, k_vec = pipeline.channel_k("overview")
    bm, vec = bm25_search(q, k_bm), vector_search(q, k_vec)
    assert cand.fused == rrf_fuse([bm, vec])
    fused_ids = {c for c, _ in cand.fused}
    assert fused_ids == {c for c, _ in bm} | {c for c, _ in vec}


@needs_index
def test_abstains_without_llm_call_when_no_passage_reaches_the_threshold(monkeypatch):
    monkeypatch.setattr(settings, "fast_rerank", True)
    monkeypatch.setattr(settings, "fast_min_evidence_score", 1.01)  # unreachable
    pipeline.cache_clear()
    llm = CitingLLM()
    res = service.run_search("How do you bake sourdough bread?", mode="overview", llm=llm)
    assert llm.calls == 0
    assert "could not find" in res["answer"].lower() and res["citations"] == []
    assert res["metrics"]["evidence"]["abstained"] is True
    assert res["results"]  # ranked search results are still returned


@needs_index
def test_stream_search_mode_events_and_no_tokens(monkeypatch):
    from app.api import stream as stream_api

    pipeline.cache_clear()
    events = list(stream_api.produce("quantum computing", "search", "all", threading.Event()))
    kinds = [e for e, _ in events]
    assert "token" not in kinds and kinds[0] == "results" and kinds[-1] == "done"
    done = events[-1][1]
    assert done["answer"] is None and done["metrics"]["llm"]["calls"] == 0
    assert "context" in done["timings"]


@needs_index
def test_stream_fast_answer_reports_stage_timings_and_models(monkeypatch):
    from app.api import stream as stream_api
    from app.retrieval import answer_cache

    monkeypatch.setattr(answer_cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(answer_cache, "put", lambda *a, **k: None)
    pipeline.cache_clear()
    events = list(stream_api.produce("transmon qubit", "overview", "all", threading.Event()))
    done = events[-1][1]
    t = done["timings"]
    assert {"embedding", "bm25", "vector", "rrf", "context", "first_token", "answer",
            "total"} <= set(t)
    assert done["metrics"]["llm"]["calls"] == 1
    assert done["metrics"]["models"]["reranker"] == (
        settings.reranker_model if settings.fast_rerank else None)
    assert done["reranker"] == done["metrics"]["models"]["reranker"]


@needs_index
def test_stream_abstention_reports_no_llm_model(monkeypatch):
    from app.api import stream as stream_api
    from app.retrieval import answer_cache

    monkeypatch.setattr(answer_cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(answer_cache, "put", lambda *a, **k: None)
    monkeypatch.setattr(settings, "fast_rerank", True)
    monkeypatch.setattr(settings, "fast_min_evidence_score", 1.01)
    pipeline.cache_clear()
    events = list(stream_api.produce("sourdough bread", "overview", "all", threading.Event()))
    done = events[-1][1]
    assert done["abstained"] is True and done["model"] is None
    assert done["metrics"]["llm"]["calls"] == 0
    assert "could not find" in done["answer"].lower()
