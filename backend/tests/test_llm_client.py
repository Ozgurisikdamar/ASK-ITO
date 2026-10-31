"""LLM provider adapters, exercised against an in-process mock HTTP transport."""
import json

import httpx
import pytest

from app.llm import client as llm
from app.llm.client import (
    FallbackClient,
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


# -- fallback ---------------------------------------------------------------

class _Broken:
    def chat(self, messages, **kw):
        raise LLMError("HTTP 429")

    def stream_chat(self, messages, **kw):
        raise LLMError("HTTP 429")
        yield  # pragma: no cover


def test_fallback_used_when_primary_fails_before_output():
    fb = FallbackClient(_Broken(), MockLLMClient())
    msgs = [{"role": "user", "content": "hello"}]
    assert fb.chat(msgs, model="primary-only").startswith("[mock]")
    assert "".join(fb.stream_chat(msgs)).startswith("[mock]")


def test_mock_stream_respects_cancel():
    import threading

    cancel = threading.Event()
    gen = MockLLMClient().stream_chat([{"role": "user", "content": "a b c d e f"}], cancel=cancel)
    first = next(gen)
    cancel.set()
    assert first and list(gen) == []
