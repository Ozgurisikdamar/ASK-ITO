"""LLM client abstraction with streaming (spec §7).

Providers (LLM_MODE):
  * "ollama": local Ollama native API (/api/chat). Sends `think` (off by default:
              reasoning tokens are slow on CPU), `keep_alive` (keeps the model in
              RAM between queries) and context/thread options. NDJSON streaming.
  * "openai": any OpenAI-compatible endpoint (e.g. Groq's free tier).
              SSE streaming; LLM_EXTRA_BODY is merged into each request body.
  * "mock":   deterministic offline stub (dev/tests); streams word by word.

All providers expose `chat()` (optionally JSON mode) and `stream_chat()`, use a
persistent HTTP client, retry 429/5xx honouring Retry-After, and strip any
`<think>...</think>` block (also across stream chunk boundaries). An optional
fallback provider takes over when the primary fails before producing output.
Model names come from config only; callers pass `settings.model_for(role)`.
"""
from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache
from typing import Iterator, Protocol

import httpx

from app.config import settings

Message = dict[str, str]
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def ping(self) -> tuple[bool, str]: ...
    def chat(
        self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None,
        json_mode: bool = False, max_tokens: int | None = None,
    ) -> str: ...
    def stream_chat(
        self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None,
        max_tokens: int | None = None, cancel: threading.Event | None = None,
    ) -> Iterator[str]: ...


def strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text or "")
    # an unterminated block (e.g. truncated output) is dropped entirely
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


class ThinkFilter:
    """Streaming filter that hides `<think>...</think>` spans split across chunks."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while self._buf:
            if self._inside:
                end = self._buf.find("</think>")
                if end == -1:
                    self._buf = self._buf[-len("</think>"):]  # keep a possible partial tag
                    return "".join(out)
                self._buf = self._buf[end + len("</think>"):].lstrip()
                self._inside = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    # hold back a trailing partial "<think" prefix
                    keep = 0
                    for i in range(1, len("<think>")):
                        if self._buf.endswith("<think>"[:i]):
                            keep = i
                    out.append(self._buf[: len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep:]
                    return "".join(out)
                out.append(self._buf[:start])
                self._buf = self._buf[start + len("<think>"):]
                self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


class _HTTPClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = httpx.Client(
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=settings.llm_connect_timeout_s),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def _retry_wait(self, resp: httpx.Response, attempt: int) -> float:
        header = resp.headers.get("Retry-After")
        try:
            return min(float(header), 30.0) if header else min(2.0 ** attempt, 30.0)
        except ValueError:
            return min(2.0 ** attempt, 30.0)

    def _post(self, url: str, body: dict) -> dict:
        for attempt in range(settings.llm_max_retries + 1):
            resp = self._http.post(url, json=body)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < settings.llm_max_retries:
                    time.sleep(self._retry_wait(resp, attempt))
                    continue
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise LLMError("retries exhausted")

    def _stream_lines(self, url: str, body: dict, cancel: threading.Event | None):
        for attempt in range(settings.llm_max_retries + 1):
            with self._http.stream("POST", url, json=body) as resp:
                if (resp.status_code == 429 or resp.status_code >= 500) and \
                        attempt < settings.llm_max_retries:
                    wait = self._retry_wait(resp, attempt)
                    resp.close()
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    resp.read()
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if cancel is not None and cancel.is_set():
                        return  # closing the response cancels generation server-side
                    if line:
                        yield line
                return
        raise LLMError("retries exhausted")


class OllamaClient(_HTTPClient):
    """Ollama native API. A trailing /v1 (OpenAI-compat path) in the URL is ignored."""

    def __init__(self, base_url: str, api_key: str = "", default_model: str = "") -> None:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        super().__init__(base, api_key)
        self.default_model = default_model
        self._no_think_field: set[str] = set()  # models that reject the `think` field

    def _model(self, model: str | None) -> str:
        m = model or self.default_model or settings.model_for("answer")
        if not m:
            raise LLMError("no LLM model configured (set LLM_DEFAULT_MODEL)")
        return m

    def ping(self) -> tuple[bool, str]:
        model = self.default_model or settings.model_for("answer")
        try:
            resp = self._http.get(f"{self.base_url}/api/tags", timeout=settings.llm_ping_timeout_s)
        except Exception as exc:  # noqa: BLE001
            return False, f"unreachable: {exc.__class__.__name__}"
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code} from {self.base_url}"
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        if model and model not in names and f"{model}:latest" not in names:
            return False, f"model '{model}' not pulled (run: ollama pull {model})"
        return True, f"{model} @ {self.base_url}"

    def _body(self, messages, model, temperature, max_tokens, json_mode, stream) -> dict:
        model = self._model(model)
        options: dict = {"temperature": temperature, "num_ctx": settings.llm_num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        if settings.llm_num_thread:
            options["num_thread"] = settings.llm_num_thread
        body = {
            "model": model, "messages": messages, "stream": stream,
            "keep_alive": settings.llm_keep_alive, "options": options,
        }
        if model not in self._no_think_field:
            body["think"] = settings.llm_think
        if json_mode:
            body["format"] = "json"
        return body

    def _with_think_fallback(self, fn, body: dict):
        try:
            return fn(body)
        except LLMError as exc:
            if "think" in body and "think" in str(exc).lower():
                self._no_think_field.add(body["model"])
                body = {k: v for k, v in body.items() if k != "think"}
                return fn(body)
            raise

    def chat(self, messages, *, temperature=0.0, model=None, json_mode=False, max_tokens=None):
        body = self._body(messages, model, temperature, max_tokens, json_mode, stream=False)
        data = self._with_think_fallback(lambda b: self._post(f"{self.base_url}/api/chat", b), body)
        return strip_think(data.get("message", {}).get("content", ""))

    def stream_chat(self, messages, *, temperature=0.0, model=None, max_tokens=None, cancel=None):
        body = self._body(messages, model, temperature, max_tokens, False, stream=True)
        filt = ThinkFilter()

        def run(b):
            for line in self._stream_lines(f"{self.base_url}/api/chat", b, cancel):
                obj = json.loads(line)
                if obj.get("error"):
                    raise LLMError(str(obj["error"]))
                piece = filt.feed(obj.get("message", {}).get("content", ""))
                if piece:
                    yield piece
                if obj.get("done"):
                    break
            tail = filt.flush()
            if tail:
                yield tail

        try:
            yield from run(body)
        except LLMError as exc:
            if "think" in body and "think" in str(exc).lower():
                self._no_think_field.add(body["model"])
                yield from run({k: v for k, v in body.items() if k != "think"})
            else:
                raise

    def warmup(self) -> None:
        """Load the model into RAM without generating (empty prompt)."""
        self._post(f"{self.base_url}/api/generate",
                   {"model": self._model(None), "prompt": "", "keep_alive": settings.llm_keep_alive})


class OpenAICompatClient(_HTTPClient):
    """OpenAI-compatible Chat Completions (Groq, OpenRouter, vLLM, ...)."""

    def __init__(self, base_url: str, api_key: str, default_model: str = "") -> None:
        super().__init__(base_url, api_key)
        self.default_model = default_model
        try:
            self.extra = json.loads(settings.llm_extra_body) if settings.llm_extra_body else {}
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM_EXTRA_BODY is not valid JSON: {exc}") from exc

    def _model(self, model):
        m = model or self.default_model or settings.model_for("answer")
        if not m:
            raise LLMError("no LLM model configured (set LLM_DEFAULT_MODEL)")
        return m

    def ping(self) -> tuple[bool, str]:
        model = self.default_model or settings.model_for("answer")
        try:
            resp = self._http.get(f"{self.base_url}/models", timeout=settings.llm_ping_timeout_s)
        except Exception as exc:  # noqa: BLE001
            return False, f"unreachable: {exc.__class__.__name__}"
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code} from {self.base_url}"
        try:
            ids = {m.get("id") for m in resp.json().get("data", [])}
        except Exception:  # noqa: BLE001
            ids = set()
        if ids and model and model not in ids:
            return False, f"model '{model}' not offered by {self.base_url}"
        return True, f"{model} @ {self.base_url}"

    def _body(self, messages, model, temperature, max_tokens, json_mode, stream) -> dict:
        body = {"model": self._model(model), "messages": messages,
                "temperature": temperature, "stream": stream}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra)
        return body

    def chat(self, messages, *, temperature=0.0, model=None, json_mode=False, max_tokens=None):
        data = self._post(f"{self.base_url}/chat/completions",
                          self._body(messages, model, temperature, max_tokens, json_mode, False))
        return strip_think(data["choices"][0]["message"].get("content") or "")

    def stream_chat(self, messages, *, temperature=0.0, model=None, max_tokens=None, cancel=None):
        body = self._body(messages, model, temperature, max_tokens, False, True)
        filt = ThinkFilter()
        for line in self._stream_lines(f"{self.base_url}/chat/completions", body, cancel):
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            choices = obj.get("choices") or []
            delta = (choices[0].get("delta") or {}).get("content") if choices else None
            if delta:
                piece = filt.feed(delta)
                if piece:
                    yield piece
        tail = filt.flush()
        if tail:
            yield tail

    def warmup(self) -> None:  # hosted endpoints need no warm-up
        return None


class MockLLMClient:
    """Deterministic offline stub. Echoes a canned, structure-aware response."""

    def ping(self) -> tuple[bool, str]:
        return True, "mock mode (no external model)"

    def chat(self, messages, *, temperature=0.0, model=None, json_mode=False, max_tokens=None):
        last = messages[-1]["content"] if messages else ""
        return f"[mock] received {len(messages)} message(s); last: {last[:80]}"

    def stream_chat(self, messages, *, temperature=0.0, model=None, max_tokens=None, cancel=None):
        text = self.chat(messages, temperature=temperature, model=model)
        for i, word in enumerate(text.split(" ")):
            if cancel is not None and cancel.is_set():
                return
            yield word if i == 0 else " " + word

    def warmup(self) -> None:
        return None


class FallbackClient:
    """Uses `primary`; on failure before any output, retries the call on `fallback`."""

    def __init__(self, primary, fallback) -> None:
        self.primary, self.fallback = primary, fallback

    def ping(self):
        ok, msg = self.primary.ping()
        if ok:
            return ok, msg
        ok2, msg2 = self.fallback.ping()
        return ok2, f"primary down ({msg}); fallback {msg2}"

    def chat(self, messages, **kw):
        try:
            return self.primary.chat(messages, **kw)
        except (LLMError, httpx.HTTPError):
            kw.pop("model", None)
            return self.fallback.chat(messages, **kw)

    def stream_chat(self, messages, **kw):
        started = False
        try:
            for piece in self.primary.stream_chat(messages, **kw):
                started = True
                yield piece
        except (LLMError, httpx.HTTPError):
            if started:
                raise
            kw.pop("model", None)
            yield from self.fallback.stream_chat(messages, **kw)

    def warmup(self):
        self.primary.warmup()


def make_client(mode: str, base_url: str, api_key: str, default_model: str = ""):
    if mode == "mock":
        return MockLLMClient()
    if mode == "openai":
        return OpenAICompatClient(base_url, api_key, default_model)
    if mode == "ollama":
        return OllamaClient(base_url, api_key, default_model)
    raise LLMError(f"unknown LLM_MODE: {mode}")


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    primary = make_client(settings.llm_mode, settings.llm_base_url, settings.llm_api_key)
    if settings.llm_fallback_mode and settings.llm_mode != "mock":
        fallback = make_client(settings.llm_fallback_mode, settings.llm_fallback_base_url,
                               settings.llm_fallback_api_key, settings.llm_fallback_model)
        return FallbackClient(primary, fallback)
    return primary


def reset_llm_client() -> None:
    get_llm_client.cache_clear()
