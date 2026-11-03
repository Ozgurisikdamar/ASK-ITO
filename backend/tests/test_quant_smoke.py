"""Q4/Q8 smoke script: identity checks (no silent substitution) and identical
prompts / limits for both models. Unit tests with a fake runtime; the real run
is reported separately."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quant_smoke as qs  # noqa: E402


def test_identity_check_refuses_missing_runtime_model_and_other_quant():
    with pytest.raises(qs.SmokeError, match="not reachable"):
        qs.check_identity({}, "qwen3.5:9b-q8_0", "Q8_0")
    with pytest.raises(qs.SmokeError, match="not available"):
        qs.check_identity({"ollama_version": "0.40.12"}, "qwen3.5:9b-q8_0", "Q8_0")
    with pytest.raises(qs.SmokeError, match="refusing"):
        qs.check_identity({"ollama_version": "x", "quantization": "Q4_K_M"},
                          "qwen3.5:9b-q8_0", "Q8_0")
    qs.check_identity({"ollama_version": "x", "quantization": "Q8_0"}, "qwen3.5:9b-q8_0", "Q8_0")


def test_citation_check():
    assert qs._citation_check("A [1]. B [2][7].", 4) == {
        "cited_ids": [1, 2, 7], "citations_total": 3, "citation_id_validity": 0.667}
    assert qs._citation_check("no citations", 4)["citation_id_validity"] is None


def test_exactly_the_five_requested_questions():
    assert qs.QUESTIONS == ["quantum computing", "What is machine learning?",
                            "popular AI fields", "Willow processor 105 physical qubits",
                            "Hopfield and Hinton, 2024 Physics prize"]


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.headers = data, status, {"content-type": "json"}

    def json(self):
        return self._d


def _fake_runtime(monkeypatch, tmp_path, quant_by_model):
    import benchkit
    import httpx

    from app.llm import client as llm_client

    frozen = {"settings": {"num_ctx": 4096}, "questions": [
        {"question": "q1", "max_tokens": 320, "sources": [{"chunk_id": 5}, {"chunk_id": 9}],
         "messages": [{"role": "user", "content": "PROMPT q1 [1] [2]"}]},
        {"question": "q2", "max_tokens": 0, "sources": [], "messages": []},
    ]}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(qs, "OUT", tmp_path)
    monkeypatch.setattr(benchkit, "ollama_info", lambda base, m: {
        "ollama_version": "0.40.12", "model": m, "quantization": quant_by_model[m]})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp({"models": []}))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp({"load_duration": 0,
                                                               "parameters": "top_p 0.95"}))
    calls = []

    def fake_stream(self, messages, temperature=0.0, model=None, max_tokens=None, **kw):
        calls.append({"model": model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens})
        self.last_usage = {"model": model, "prompt_tokens": 100, "prompt_eval_ms": 5000,
                           "completion_tokens": 20, "eval_ms": 8000}
        yield "Answer [1]."

    monkeypatch.setattr(llm_client.OllamaClient, "stream_chat", fake_stream)
    return calls


def test_both_models_get_identical_messages_and_limits(monkeypatch, tmp_path):
    calls = _fake_runtime(monkeypatch, tmp_path,
                          {"qwen3.5:9b": "Q4_K_M", "qwen3.5:9b-q8_0": "Q8_0"})
    ns = type("A", (), {})
    for model, quant in (("qwen3.5:9b", "Q4_K_M"), ("qwen3.5:9b-q8_0", "Q8_0")):
        a = ns()
        a.model, a.expect_quant, a.tag = model, quant, "t"
        qs.run(a)
    assert [c["model"] for c in calls] == ["qwen3.5:9b", "qwen3.5:9b-q8_0"]
    assert calls[0]["messages"] == calls[1]["messages"]
    assert (calls[0]["temperature"], calls[0]["max_tokens"]) == \
        (calls[1]["temperature"], calls[1]["max_tokens"]) == (0.0, 320)
    q8 = json.loads((tmp_path / "run_qwen3.5_9b-q8_0_t.json").read_text())
    assert q8["status"] == "ok" and q8["rows"][0]["source_chunk_ids"] == [5, 9]
    assert q8["rows"][0]["decode_tok_s"] == 2.5 and q8["rows"][1]["abstained_before_llm"]


def test_quantization_mismatch_stops_without_generating(monkeypatch, tmp_path):
    calls = _fake_runtime(monkeypatch, tmp_path, {"qwen3.5:9b-q8_0": "Q4_K_M"})
    a = type("A", (), {})()
    a.model, a.expect_quant, a.tag = "qwen3.5:9b-q8_0", "Q8_0", "t"
    with pytest.raises(SystemExit, match="STOPPED"):
        qs.run(a)
    assert calls == []
    out = json.loads((tmp_path / "run_qwen3.5_9b-q8_0_t.json").read_text())
    assert out["status"] == "stopped" and "Q4_K_M" in out["error"]
