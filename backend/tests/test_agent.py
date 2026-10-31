"""Agentic loop integration tests with a scripted LLM (no model/network for the
LLM side; retrieval + reranker are real). Skipped when the index is absent."""
import psycopg
import pytest

from app.config import settings
from app.index import bm25_index

HOPFIELD_Q = "What did Hopfield and Hinton each contribute to the 2024 Physics Nobel?"


def _index_ready() -> bool:
    if not bm25_index.exists():
        return False
    try:
        with psycopg.connect(settings.database_url) as c:
            return c.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _index_ready(), reason="index/DB not built")


class ScriptedLLM:
    """Returns controlled outputs based on which prompt is being asked."""

    def __init__(self, answer_text: str):
        self.answer_text = answer_text
        self.calls = 0

    def ping(self):
        return True, "scripted"

    def chat(self, messages, temperature=0.0, model=None, **kwargs):
        self.calls += 1
        p = messages[-1]["content"]
        if "Analyse the user's question" in p:
            return ('{"qtype": "multi_hop", "subquestions": '
                    '["What did John Hopfield contribute?", "What did Geoffrey Hinton contribute?"]}')
        if "Rewrite the user's question" in p:
            return '{"query": "Hopfield Hinton neural networks Nobel 2024"}'
        if "Decide whether the collected context" in p:
            return '{"sufficient": true, "missing": "", "next_query": null}'
        if "question-answering assistant" in p:
            return self.answer_text
        return "ok"


def _run(answer_text):
    from app.agent.loop import run_agent

    return run_agent(HOPFIELD_Q, domain="all", final_k=5, llm=ScriptedLLM(answer_text))


def test_multihop_produces_multiple_queries_and_pipeline_steps():
    res = _run("Hopfield built associative memory [1]. Hinton worked on Boltzmann machines [2].")
    steps = [s["step"] for s in res["trace"]]
    assert steps.count("search") >= 2           # one search per sub-question
    for expected in ("analyze", "bm25", "vector", "rrf", "rerank", "answer"):
        assert expected in steps, f"missing {expected}: {steps}"
    assert res["mode"] == "agentic"
    assert res["llm_calls"] <= settings.agent_max_llm_calls


def test_citation_validation_strips_invalid():
    res = _run("Valid claim [1]. Bogus source [9][0].")
    ns = [c["n"] for c in res["citations"]]
    assert 9 not in ns and 0 not in ns
    assert "[9]" not in res["answer"] and "[0]" not in res["answer"]
    assert all(1 <= c["n"] <= len(res["sources"]) for c in res["citations"])


def test_unanswerable_says_not_found():
    res = _run("I could not find this in the sources.")
    assert "could not find" in res["answer"].lower()
    assert res["citations"] == []
