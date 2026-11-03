"""Adaptive agent switches (all off by default) and budget/deadline guardrails.

Real retrieval + reranker, scripted LLM. Skipped when the index is absent."""
import pytest

from app.config import settings
from tests.test_agent import HOPFIELD_Q, ScriptedLLM, _index_ready

pytestmark = pytest.mark.skipif(not _index_ready(), reason="index/DB not built")

SIMPLE_Q = "What is a transmon qubit?"


class SimpleLLM(ScriptedLLM):
    def chat(self, messages, temperature=0.0, model=None, **kwargs):
        if "Analyse the user's question" in messages[-1]["content"]:
            self.calls += 1
            return '{"qtype": "simple", "subquestions": []}'
        return super().chat(messages, temperature, model, **kwargs)


class InsufficientLLM(ScriptedLLM):
    """Never satisfied; asks for a different follow-up search each round."""
    FOLLOW_UPS = ["Hinton backpropagation", "Boltzmann machine", "Hopfield network memory",
                  "restricted Boltzmann machine training"]

    def chat(self, messages, temperature=0.0, model=None, **kwargs):
        if "Decide whether the collected context" in messages[-1]["content"]:
            self.calls += 1
            self.n_suff = getattr(self, "n_suff", 0) + 1
            nxt = self.FOLLOW_UPS[(self.n_suff - 1) % len(self.FOLLOW_UPS)]
            return '{"sufficient": false, "missing": "x", "next_query": "%s"}' % nxt
        return super().chat(messages, temperature, model, **kwargs)


class RepeatingLLM(ScriptedLLM):
    """Never satisfied and keeps asking for the same follow-up query."""

    def chat(self, messages, temperature=0.0, model=None, **kwargs):
        if "Decide whether the collected context" in messages[-1]["content"]:
            self.calls += 1
            return '{"sufficient": false, "missing": "x", "next_query": "Hinton backpropagation"}'
        return super().chat(messages, temperature, model, **kwargs)


def _run(q=HOPFIELD_Q, llm=None):
    from app.agent.loop import run_agent
    from app.retrieval import pipeline

    pipeline.cache_clear()
    return run_agent(q, domain="all", final_k=5, llm=llm or ScriptedLLM("A [1]."))


def _steps(res):
    return [s["step"] for s in res["trace"]]


def _roles(res):
    return res["metrics"]["llm"]["by_role"]


def test_defaults_reproduce_original_flow():
    assert settings.agent_rewrite_policy == "always"
    assert not settings.agent_adaptive_sufficiency
    assert settings.agent_final_rerank_pool == 0 and not settings.agent_reuse_rerank_scores
    res = _run()
    assert "rewrite_skipped" not in _steps(res) and "sufficiency_skipped" not in _steps(res)
    assert _roles(res)["rewrite"]["calls"] == 2  # one per sub-question
    assert res["agent_counts"]["final_rerank_reused"] == 0


def test_skip_subquestions_policy(monkeypatch):
    monkeypatch.setattr(settings, "agent_rewrite_policy", "skip_subquestions")
    res = _run()
    assert "rewrite" not in _roles(res)
    assert _steps(res).count("rewrite_skipped") == 2
    assert _steps(res).count("search") >= 2  # multi-hop still searches every sub-question


def test_skip_simple_policy_only_affects_simple_questions(monkeypatch):
    monkeypatch.setattr(settings, "agent_rewrite_policy", "skip_simple")
    simple = _run(SIMPLE_Q, SimpleLLM("A [1]."))
    assert "rewrite" not in _roles(simple) and "rewrite_skipped" in _steps(simple)
    multi = _run()
    assert _roles(multi)["rewrite"]["calls"] == 2


def test_adaptive_sufficiency_skips_llm_when_covered(monkeypatch):
    monkeypatch.setattr(settings, "agent_adaptive_sufficiency", True)
    monkeypatch.setattr(settings, "agent_coverage_min_score", -1e9)
    res = _run()
    assert "sufficiency" not in _roles(res) and "sufficiency_skipped" in _steps(res)
    monkeypatch.setattr(settings, "agent_coverage_min_score", 1e9)
    res = _run()
    assert _roles(res)["sufficiency"]["calls"] >= 1


def test_adaptive_caps_extra_rounds_for_simple_questions(monkeypatch):
    class SimpleInsufficient(InsufficientLLM, SimpleLLM):
        pass

    monkeypatch.setattr(settings, "agent_coverage_min_score", 1e9)
    base = _run(SIMPLE_Q, SimpleInsufficient("A [1]."))
    monkeypatch.setattr(settings, "agent_adaptive_sufficiency", True)
    monkeypatch.setattr(settings, "agent_max_extra_rounds_simple", 1)
    capped = _run(SIMPLE_Q, SimpleInsufficient("A [1]."))
    assert base["rounds"] == settings.agent_max_rounds
    assert capped["rounds"] == 2


def test_final_rerank_pool_and_score_reuse(monkeypatch):
    monkeypatch.setattr(settings, "agent_final_rerank_pool", 5)
    res = _run()
    assert res["agent_counts"]["final_rerank_scored"] <= 5
    assert res["agent_counts"]["union"] >= res["agent_counts"]["final_rerank_scored"]

    monkeypatch.setattr(settings, "agent_final_rerank_pool", 0)
    monkeypatch.setattr(settings, "agent_rewrite_policy", "never")
    monkeypatch.setattr(settings, "agent_reuse_rerank_scores", False)
    fresh = _run(SIMPLE_Q, SimpleLLM("A [1]."))
    monkeypatch.setattr(settings, "agent_reuse_rerank_scores", True)
    reused = _run(SIMPLE_Q, SimpleLLM("A [1]."))
    assert reused["agent_counts"]["final_rerank_reused"] > 0
    # identical (query, passage) pairs -> identical ranking and answer sources
    assert [s["chunk_id"] for s in reused["sources"]] == [s["chunk_id"] for s in fresh["sources"]]
    assert reused["ranked_chunk_ids"] == fresh["ranked_chunk_ids"]


def test_call_budget_reserves_final_answer(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_llm_calls", 2)
    res = _run()
    assert res["llm_calls"] <= 2
    assert "answer" in _roles(res) and res["answer"]


def test_deadline_still_answers(monkeypatch):
    monkeypatch.setattr(settings, "agent_timeout_s", 0)
    res = _run()
    assert "answer" in _steps(res) and res["answer"]


# -- deterministic guards ---------------------------------------------------------

def test_identical_search_queries_are_searched_once():
    class SameRewriteLLM(ScriptedLLM):
        def chat(self, messages, temperature=0.0, model=None, **kwargs):
            if "Rewrite the user's question" in messages[-1]["content"]:
                self.calls += 1
                return '{"query": "Hopfield Hinton neural networks Nobel 2024"}'
            return super().chat(messages, temperature, model, **kwargs)

    res = _run(llm=SameRewriteLLM("A [1]."))  # both sub-questions -> one query
    assert _steps(res).count("search") == 1 and "search_skipped" in _steps(res)
    assert _roles(res)["rewrite"]["calls"] == 2  # rewriting itself is unchanged


def test_repeated_follow_up_query_stops_the_sufficiency_loop(monkeypatch):
    monkeypatch.setattr(settings, "agent_max_rounds", 5)
    monkeypatch.setattr(settings, "agent_max_llm_calls", 20)
    res = _run(llm=RepeatingLLM("A [1]."))
    stops = [s for s in res["trace"] if s["step"] == "sufficiency_stopped"]
    assert stops and "already searched" in stops[0]["output"]["reason"]
    # one follow-up search, then the repeat is detected: 2 sufficiency calls, not 4
    assert _roles(res)["sufficiency"]["calls"] == 2 and res["rounds"] == 2
    assert "answer" in _roles(res) and res["answer"]


def test_duplicate_subquestions_are_dropped():
    class DupLLM(ScriptedLLM):
        def chat(self, messages, temperature=0.0, model=None, **kwargs):
            if "Analyse the user's question" in messages[-1]["content"]:
                self.calls += 1
                return ('{"qtype": "multi_hop", "subquestions": ["What did Hopfield do?", '
                        '"what did  hopfield do?"]}')
            return super().chat(messages, temperature, model, **kwargs)

    res = _run(llm=DupLLM("A [1]."))
    assert "subquestion_skipped" in _steps(res)
    assert _roles(res)["rewrite"]["calls"] == 1
