"""Evidence cut-off: a weak out-of-domain gate on the reranker score.

Off-topic questions are stopped before the LLM; answerable questions pass;
near-domain unsupported questions usually pass too (documented limitation —
they rely on the prompt's not-found instruction). The cut-off is only active
with the reranker it was calibrated for."""
import json
from pathlib import Path

import pytest

from app.config import settings
from app.retrieval import pipeline, service
from tests.test_agent import _index_ready
from tests.test_fast_rag import CitingLLM

CAL = json.loads((Path(__file__).resolve().parents[1] / "eval" /
                  "evidence_calibration.json").read_text())


def test_cutoff_disabled_for_an_uncalibrated_reranker(monkeypatch):
    monkeypatch.setattr(settings, "answer_rerank", True)
    monkeypatch.setattr(settings, "fast_rerank", None)
    monkeypatch.setattr(settings, "fast_min_evidence_score", 0.3)
    monkeypatch.setattr(settings, "reranker_model", settings.evidence_score_model)
    assert service.evidence_cutoff_active("overview")
    monkeypatch.setattr(settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert not service.evidence_cutoff_active("overview")
    ranked = [(1, -3.0), (2, -5.0)]  # MiniLM logits: would all be cut by 0.3
    assert service.evidence_filter(ranked, "overview", 2) == ranked


def test_calibration_set_has_all_groups():
    assert CAL["answerable"] and CAL["near_domain_unsupported"] and CAL["out_of_domain"]


needs_index = pytest.mark.skipif(not _index_ready(), reason="index/DB not built")


def _answer(q):
    pipeline.cache_clear()
    llm = CitingLLM()
    res = service.run_search(q, mode="overview", llm=llm)
    return llm.calls, res


@needs_index
@pytest.mark.parametrize("q", CAL["out_of_domain"][:2])
def test_out_of_domain_is_stopped_without_llm(q):
    calls, res = _answer(q)
    assert calls == 0 and res["metrics"]["evidence"]["abstained"] is True
    assert "could not find" in res["answer"].lower()


@needs_index
def test_answerable_question_reaches_the_llm():
    calls, res = _answer(CAL["answerable"][0])
    assert calls == 1 and res["metrics"]["evidence"]["abstained"] is False


@needs_index
def test_near_domain_unsupported_can_pass_the_gate():
    # Known limitation: shares vocabulary with the corpus, so the reranker scores
    # it highly; abstention is left to the model's not-found instruction.
    calls, res = _answer("Which organisation received the 2024 Nobel Peace Prize?")
    assert calls == 1 and res["metrics"]["evidence"]["abstained"] is False
