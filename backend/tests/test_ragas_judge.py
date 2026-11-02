"""RAGAS judge adapter: runs through the project LLM client, sequential, JSON mode."""
import asyncio

import pytest

from app.eval import ragas_eval

pytest.importorskip("ragas")


class _Client:
    def __init__(self):
        self.calls = []
        self.last_usage = {"prompt_tokens": 12, "completion_tokens": 3}

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return '{"verdict": 1}'


def _prompt(text):
    from langchain_core.prompt_values import StringPromptValue

    return StringPromptValue(text=text)


def test_judge_adapter_uses_project_client_in_json_mode():
    client = _Client()
    judge = ragas_eval.make_judge_llm(client, "judge-model")
    res = judge.generate_text(_prompt("Is this faithful?"), n=2, temperature=0.0)
    assert [g.text for g in res.generations[0]] == ['{"verdict": 1}'] * 2
    messages, kw = client.calls[0]
    assert messages == [{"role": "user", "content": "Is this faithful?"}]
    assert kw["model"] == "judge-model" and kw["json_mode"] is True
    assert kw["max_tokens"] == ragas_eval.settings.llm_judge_max_tokens
    assert len(judge.calls) == 2 and judge.calls[0]["prompt_tokens"] == 12


def test_judge_adapter_async_path():
    client = _Client()
    judge = ragas_eval.make_judge_llm(client, "m")
    res = asyncio.run(judge.agenerate_text(_prompt("x")))
    assert res.generations[0][0].text == '{"verdict": 1}'


def test_run_config_is_sequential():
    rc = ragas_eval.run_config()
    assert rc.max_workers == 1 and rc.timeout == ragas_eval.settings.ragas_timeout_s


def test_compute_reports_unavailable_in_mock_mode():
    out = ragas_eval.compute([{"question": "q", "answer": "a", "contexts": ["c"],
                               "reference": "r"}])
    assert out["faithfulness"] is None and "mock" in out["ragas_status"]
