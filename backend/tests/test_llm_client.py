"""LLM provider adapters, exercised against an in-process mock HTTP transport."""
import json

import httpx
import pytest

from app.llm import client as llm
from app.llm.client import (
    LLMError,
    MockLLMClient,
    OllamaClient,
    OpenAICompatClient,
    ThinkFilter,
    strip_think,
)


def _wire(c, handler):
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


# -- think filtering ---------------------------------------------------------

def test_strip_think_removes_blocks_and_unterminated_tail():
    assert strip_think("<think>hidden</think> Answer [1].") == "Answer [1]."
    assert strip_think("Answer. <think>never closed") == "Answer."


def test_think_filter_handles_tags_split_across_chunks():
    f = ThinkFilter()
    chunks = ["Hel", "lo <thi", "nk>secret rea", "soning</th", "ink> world", "!"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "Hello world!"


def test_think_filter_keeps_plain_angle_brackets():
    f = ThinkFilter()
    out = f.feed("a < b and <tag>") + f.flush()
    assert out == "a < b and <tag>"


# -- Ollama native -----------------------------------------------------------

def test_ollama_chat_sends_native_body_and_strips_v1():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"message": {"content": "<think>x</think>Hi [1]"}})

    c = _wire(OllamaClient("http://ollama:11434/v1", default_model="qwen3:30b"), handler)
    out = c.chat([{"role": "user", "content": "q"}], json_mode=True, max_tokens=50)
    assert out == "Hi [1]"
    assert seen["url"] == "http://ollama:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "qwen3:30b" and body["stream"] is False
    assert body["think"] is False and body["format"] == "json"
    assert body["options"]["num_predict"] == 50 and "keep_alive" in body


def test_ollama_stream_parses_ndjson_and_hides_think():
    lines = [
        {"message": {"content": "<think>plan"}, "done": False},
        {"message": {"content": "</think>The "}, "done": False},
        {"message": {"content": "answer [1]."}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]
    payload = "\n".join(json.dumps(x) for x in lines) + "\n"
    c = _wire(OllamaClient("http://o:11434", default_model="m"),
              lambda r: httpx.Response(200, text=payload))
    assert "".join(c.stream_chat([{"role": "user", "content": "q"}])) == "The answer [1]."


def test_ollama_retries_without_think_field_when_rejected():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        if req.url.path != "/api/chat":
            return httpx.Response(404, json={"error": "unknown"})  # /api/show: caps unknown
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": "model does not support think"})
        return httpx.Response(200, json={"message": {"content": "ok"}})

    c = _wire(OllamaClient("http://o:11434", default_model="old"), handler)
    assert c.chat([{"role": "user", "content": "q"}]) == "ok"
    assert "think" in bodies[0] and "think" not in bodies[1]
    c.chat([{"role": "user", "content": "q"}])  # remembered: no third rejected call
    assert len(bodies) == 3 and "think" not in bodies[2]


def test_ollama_ping_reports_missing_model():
    c = _wire(OllamaClient("http://o:11434", default_model="qwen3:30b"),
              lambda r: httpx.Response(200, json={"models": [{"name": "other:7b"}]}))
    ok, msg = c.ping()
    assert not ok and "not pulled" in msg


def test_missing_model_is_an_error(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_default_model", "")
    monkeypatch.setattr(llm.settings, "llm_answer_model", "")
    c = OllamaClient("http://o:11434", default_model="")
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "q"}])


# -- OpenAI-compatible (Groq etc.) -------------------------------------------

def test_openai_stream_parses_sse():
    sse = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"Fast "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"answer."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    c = _wire(OpenAICompatClient("https://api.example/v1", "k", "qwen/qwen3-32b"),
              lambda r: httpx.Response(200, text=sse))
    assert "".join(c.stream_chat([{"role": "user", "content": "q"}])) == "Fast answer."


def test_openai_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "rate"})
        assert json.loads(req.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    c = _wire(OpenAICompatClient("https://api.example/v1", "k", "m"), handler)
    assert c.chat([{"role": "user", "content": "q"}], json_mode=True) == "{}"
    assert calls["n"] == 2


def test_openai_raises_after_retries_exhausted():
    c = _wire(OpenAICompatClient("https://api.example/v1", "k", "m"),
              lambda r: httpx.Response(503, text="down"))
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "q"}])


# -- no fallback ------------------------------------------------------------

def test_no_fallback_provider_exists():
    import app.config as cfg

    assert not hasattr(llm, "FallbackClient")
    assert not any(k.startswith("llm_fallback") for k in cfg.Settings.model_fields)


def test_mock_stream_respects_cancel():
    import threading

    cancel = threading.Event()
    gen = MockLLMClient().stream_chat([{"role": "user", "content": "a b c d e f"}], cancel=cancel)
    first = next(gen)
    cancel.set()
    assert first and list(gen) == []


# -- capabilities, usage accounting, context-overflow guard -------------------

def _ollama_handler(caps, chat_payload, seen):
    def handler(req):
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": caps})
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=chat_payload)
    return handler


def test_think_field_omitted_for_models_without_thinking_capability():
    seen = []
    c = _wire(OllamaClient("http://o:11434", default_model="m"),
              _ollama_handler(["completion"], {"message": {"content": "x"}}, seen))
    c.chat([{"role": "user", "content": "q"}])
    assert "think" not in seen[0]
    seen.clear()
    c2 = _wire(OllamaClient("http://o:11434", default_model="m"),
               _ollama_handler(["completion", "thinking"], {"message": {"content": "x"}}, seen))
    c2.chat([{"role": "user", "content": "q"}])
    assert seen[0]["think"] is False


def test_usage_recorded_and_context_overflow_warned(monkeypatch, caplog):
    monkeypatch.setattr(llm.settings, "llm_num_ctx", 1000)
    payload = {"message": {"content": "x"}, "prompt_eval_count": 950, "eval_count": 40,
               "prompt_eval_duration": 2_000_000_000, "eval_duration": 4_000_000_000,
               "load_duration": 1_000_000, "done_reason": "stop", "model": "m"}
    seen = []
    c = _wire(OllamaClient("http://o:11434", default_model="m"),
              _ollama_handler(["completion"], payload, seen))
    with caplog.at_level("WARNING", logger="askito.llm"):
        c.chat([{"role": "user", "content": "q"}], max_tokens=100)
    u = c.last_usage
    assert (u["prompt_tokens"], u["completion_tokens"], u["eval_ms"]) == (950, 40, 4000.0)
    assert u["context_overflow"] is True and c.context_overflows == 1
    assert "context window" in caplog.text
    payload["prompt_eval_count"] = 100
    c.chat([{"role": "user", "content": "q"}], max_tokens=100)
    assert c.last_usage["context_overflow"] is False and c.context_overflows == 1


def test_payload_options_follow_config(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_num_ctx", 8192)
    monkeypatch.setattr(llm.settings, "llm_num_thread", 3)
    monkeypatch.setattr(llm.settings, "llm_keep_alive", "10m")
    seen = []
    c = _wire(OllamaClient("http://o:11434", default_model="m"),
              _ollama_handler(None, {"message": {"content": "x"}}, seen))
    c.chat([{"role": "user", "content": "q"}], max_tokens=77, temperature=0.2)
    opts = seen[0]["options"]
    assert opts == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 77, "num_thread": 3}
    assert seen[0]["keep_alive"] == "10m"
    monkeypatch.setattr(llm.settings, "llm_num_thread", 0)
    c.chat([{"role": "user", "content": "q"}])
    assert "num_thread" not in seen[1]["options"]  # 0 = let Ollama decide


def test_timeout_surfaces_as_error():
    def handler(req):
        if req.url.path == "/api/show":
            return httpx.Response(404)
        raise httpx.ReadTimeout("slow", request=req)

    c = _wire(OllamaClient("http://o:11434", default_model="m"), handler)
    with pytest.raises(httpx.ReadTimeout):
        c.chat([{"role": "user", "content": "q"}])


def test_role_model_fallback(monkeypatch):
    s = llm.settings
    monkeypatch.setattr(s, "llm_default_model", "base")
    for attr in ("llm_agent_model", "llm_answer_model", "llm_judge_model"):
        monkeypatch.setattr(s, attr, "")
    assert {r: s.model_for(r) for r in ("agent", "answer", "judge")} == \
        {"agent": "base", "answer": "base", "judge": "base"}
    monkeypatch.setattr(s, "llm_judge_model", "judge-14b")
    assert s.model_for("judge") == "judge-14b" and s.model_for("agent") == "base"
