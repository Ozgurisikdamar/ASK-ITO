"""Agent tool cache: deterministic analyse / rewrite / sufficiency results are
reused only for an identical model + prompt + limits key."""
import pytest

from app.agent import tool_cache
from app.agent.schemas import Analysis, Sufficiency, call_json
from app.config import settings
from app.llm.metrics import recording


class JsonLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return self.reply


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(settings, "agent_tool_cache", True)
    monkeypatch.setattr(settings, "llm_think", False)
    monkeypatch.setattr(settings, "redis_enabled", False)
    tool_cache.clear()
    yield
    tool_cache.clear()


def test_identical_prompt_is_served_from_cache():
    llm = JsonLLM('{"qtype": "simple", "subquestions": []}')
    with recording() as rec:
        a = call_json(llm, "PROMPT q1", Analysis, cache_role="analyze")
        b = call_json(llm, "PROMPT q1", Analysis, cache_role="analyze")
    assert llm.calls == 1 and a == b and a.qtype == "simple"
    assert rec.notes["tool_cache_hits"] == {"analyze": 1}


def test_any_prompt_or_limit_change_is_a_miss():
    llm = JsonLLM('{"sufficient": true, "missing": ""}')
    call_json(llm, "context A", Sufficiency, cache_role="sufficiency")
    call_json(llm, "context B", Sufficiency, cache_role="sufficiency")
    call_json(llm, "context A", Sufficiency, cache_role="sufficiency", max_tokens=200)
    assert llm.calls == 3


def test_model_change_is_a_miss(monkeypatch):
    llm = JsonLLM('{"qtype": "simple", "subquestions": []}')
    call_json(llm, "P", Analysis, cache_role="analyze")
    monkeypatch.setattr(settings, "llm_default_model", "another-model:q8_0")
    call_json(llm, "P", Analysis, cache_role="analyze")
    assert llm.calls == 2


def test_invalid_output_is_not_cached():
    llm = JsonLLM("not json")
    assert call_json(llm, "P", Analysis, retries=0, cache_role="analyze") is None
    assert call_json(llm, "P", Analysis, retries=0, cache_role="analyze") is None
    assert llm.calls == 2


def test_off_when_thinking_or_disabled(monkeypatch):
    llm = JsonLLM('{"qtype": "simple", "subquestions": []}')
    monkeypatch.setattr(settings, "llm_think", True)
    call_json(llm, "P", Analysis, cache_role="analyze")
    call_json(llm, "P", Analysis, cache_role="analyze")
    monkeypatch.setattr(settings, "llm_think", False)
    monkeypatch.setattr(settings, "agent_tool_cache", False)
    call_json(llm, "P", Analysis, cache_role="analyze")
    assert llm.calls == 3


def test_without_cache_role_nothing_is_cached():
    llm = JsonLLM('{"qtype": "simple", "subquestions": []}')
    call_json(llm, "P", Analysis)
    call_json(llm, "P", Analysis)
    assert llm.calls == 2


def test_rewrite_is_cached():
    from app.agent.rewrite import rewrite_query

    llm = JsonLLM('{"query": "qubit definition"}')
    assert rewrite_query("what is a qubit", llm=llm) == "qubit definition"
    assert rewrite_query("what is a qubit", llm=llm) == "qubit definition"
    assert llm.calls == 1
