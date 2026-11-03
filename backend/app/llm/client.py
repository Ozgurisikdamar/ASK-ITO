"""LLM client abstraction with streaming (spec §7).

Providers (LLM_MODE):
  * "ollama": local Ollama native API (/api/chat). Sends `think` (off by default:
              reasoning tokens are slow on CPU), `keep_alive` (keeps the model in
              RAM between queries) and context/thread options. NDJSON streaming.
  * "openai": an OpenAI-compatible endpoint served locally (llama.cpp server,
              vLLM, LM Studio, ...). SSE streaming; LLM_EXTRA_BODY is merged
              into each request body.
  * "mock":   deterministic offline stub (dev/tests); streams word by word.

All providers expose `chat()` (optionally JSON mode) and `stream_chat()`, use a
persistent HTTP client, retry 429/5xx honouring Retry-After, and strip any
`<think>...</think>` block (also across stream chunk boundaries). There is no
fallback provider: a failing local model surfaces as an error.
Model names come from config only; callers pass `settings.model_for(role)`.

With LLM_LOCAL_ONLY=true (default) every endpoint must be local (app.llm.policy).
After each Ollama call `last_usage` holds the token counts and durations Ollama
reported; a prompt that fills the context window (Ollama would silently drop
its beginning) is logged as a warning and counted in `context_overflows`.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from functools import lru_cache
from typing import Iterator, Protocol

import httpx

from app.config import settings
from app.llm.policy import allowed_hosts_from_setting, ensure_local_url

log = logging.getLogger("askito.llm")
Message = dict[str, str]
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLMError(RuntimeError):
    """An LLM call failed. `code` says why, so the API can tell the user what to do:
    model_memory | model_not_found | model_unreachable | model_timeout |
    model_load_timeout | cancelled | generation_failed."""

    def __init__(self, message: str, code: str = "generation_failed") -> None:
        super().__init__(message)
        self.code = code


_MEMORY_HINTS = ("requires more system memory", "out of memory", "insufficient memory",
                 "not enough memory", "signal: killed", "cannot allocate memory",
                 "failed to allocate")


def classify_error_text(status: int | None, text: str) -> str:
    """Error code for an Ollama / OpenAI-compatible error response."""
    low = (text or "").lower()
    if any(h in low for h in _MEMORY_HINTS):
        return "model_memory"
    if status == 404 or ("model" in low and "not found" in low):
        return "model_not_found"
    if "timed out waiting for llama-server" in low or "timed out waiting for llama runner" in low:
        return "model_load_timeout"
    return "generation_failed"


def _transport_error(exc: Exception) -> LLMError:
    """Map an httpx exception to a classified LLMError (message keeps the class only)."""
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        return LLMError(f"model server unreachable ({exc.__class__.__name__})",
                        "model_unreachable")
    if isinstance(exc, httpx.TimeoutException):
        return LLMError(f"model server did not answer in time ({exc.__class__.__name__})",
                        "model_timeout")
    return LLMError(f"connection to the model server failed ({exc.__class__.__name__})",
                    "generation_failed")


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


def local_allowed_hosts() -> list[str]:
    return allowed_hosts_from_setting(settings.llm_allowed_hosts)


def _check_request_url(request: httpx.Request) -> None:
    """Re-validate the target of every outgoing LLM request (LLM_LOCAL_ONLY)."""
    ensure_local_url(str(request.url), "LLM request", local_allowed_hosts())


class _HTTPClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        hooks = {"request": [_check_request_url]} if settings.llm_local_only else {}
        self._http = httpx.Client(
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=settings.llm_connect_timeout_s),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            follow_redirects=False,  # a redirect could point outside the local host
            event_hooks=hooks,
        )

    def _retry_wait(self, resp: httpx.Response, attempt: int) -> float:
        header = resp.headers.get("Retry-After")
        try:
            return min(float(header), 30.0) if header else min(2.0 ** attempt, 30.0)
        except ValueError:
            return min(2.0 ** attempt, 30.0)

    def _post(self, url: str, body: dict, timeout: httpx.Timeout | None = None) -> dict:
        for attempt in range(settings.llm_max_retries + 1):
            try:
                resp = self._http.post(url, json=body, **({"timeout": timeout} if timeout else {}))
            except httpx.HTTPError as exc:
                raise _transport_error(exc) from exc
            code = classify_error_text(resp.status_code, resp.text) \
                if resp.status_code >= 400 else None
            if (resp.status_code == 429 or resp.status_code >= 500) and code in (
                    None, "generation_failed"):  # memory / missing model: retrying won't help
                if attempt < settings.llm_max_retries:
                    time.sleep(self._retry_wait(resp, attempt))
                    continue
            if 300 <= resp.status_code < 400:
                raise LLMError(f"HTTP {resp.status_code}: redirects are not followed")
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}", code)
            return resp.json()
        raise LLMError("retries exhausted")

    def _stream_lines(self, url: str, body: dict, cancel: threading.Event | None):
        for attempt in range(settings.llm_max_retries + 1):
            try:
                with self._http.stream("POST", url, json=body) as resp:
                    if (resp.status_code == 429 or resp.status_code >= 500) and \
                            attempt < settings.llm_max_retries:
                        resp.read()
                        if classify_error_text(resp.status_code, resp.text) == \
                                "generation_failed":
                            wait = self._retry_wait(resp, attempt)
                            resp.close()
                            time.sleep(wait)
                            continue
                    if 300 <= resp.status_code < 400:
                        raise LLMError(f"HTTP {resp.status_code}: redirects are not followed")
                    if resp.status_code >= 400:
                        resp.read()
                        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}",
                                       classify_error_text(resp.status_code, resp.text))
                    yield from self._lines_until_cancel(resp, cancel)
                    return
            except httpx.HTTPError as exc:
                if cancel is not None and cancel.is_set():
                    return  # we closed the connection ourselves
                raise _transport_error(exc) from exc
        raise LLMError("retries exhausted")

    @staticmethod
    def _lines_until_cancel(resp: httpx.Response, cancel: threading.Event | None):
        """Yield response lines; on cancel, close the response from a watcher thread so
        a request still in prompt evaluation (no tokens yet) stops at once and the
        model server aborts the generation."""
        done = threading.Event()
        if cancel is not None:
            def watch() -> None:
                while not done.is_set():
                    if cancel.wait(0.25):
                        try:
                            resp.close()
                        except Exception:  # noqa: BLE001
                            pass
                        return

            threading.Thread(target=watch, name="askito-llm-cancel", daemon=True).start()
        try:
            for line in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    return  # closing the response cancels generation server-side
                if line:
                    yield line
        finally:
            done.set()


class OllamaClient(_HTTPClient):
    """Ollama native API. A trailing /v1 (OpenAI-compat path) in the URL is ignored."""

    def __init__(self, base_url: str, api_key: str = "", default_model: str = "") -> None:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        super().__init__(base, api_key)
        self.default_model = default_model
        self._no_think_field: set[str] = set()  # models that reject the `think` field
        self._show_cache: dict[str, tuple[float, dict | None]] = {}
        self.last_usage: dict = {}
        self.context_overflows = 0

    def _show(self, model: str) -> dict | None:
        """/api/show for a model (cached; a failure is retried after 30 s)."""
        hit = self._show_cache.get(model)
        if hit is not None and (hit[1] is not None or time.monotonic() - hit[0] < 30):
            return hit[1]
        try:
            resp = self._http.post(f"{self.base_url}/api/show", json={"model": model},
                                   timeout=settings.llm_ping_timeout_s * 5)
            data = resp.json() if resp.status_code == 200 else None
        except Exception:  # noqa: BLE001
            data = None
        self._show_cache[model] = (time.monotonic(), data)
        return data

    def capabilities(self, model: str) -> list[str] | None:
        """Model capabilities from /api/show (cached); None if unknown."""
        data = self._show(model)
        return data.get("capabilities") if data else None

    def model_identity(self, model: str) -> dict:
        """What is actually served under a tag: weight quantization (from the
        GGUF metadata Ollama reports), parameter size, family and the local
        manifest digest. Unknown fields are None (Ollama unreachable)."""
        model = self._model(model)
        data = self._show(model) or {}
        details = data.get("details") or {}
        digest = None
        try:
            resp = self._http.get(f"{self.base_url}/api/tags",
                                  timeout=settings.llm_ping_timeout_s * 5)
            for m in resp.json().get("models", []) if resp.status_code == 200 else []:
                if m.get("name") == model or m.get("model") == model:
                    digest = m.get("digest")
        except Exception:  # noqa: BLE001
            pass
        return {"name": model, "quantization": details.get("quantization_level"),
                "parameter_size": details.get("parameter_size"),
                "family": details.get("family"), "digest": digest}

    def _record_usage(self, data: dict, body: dict) -> None:
        opts = body.get("options", {})
        usage = {
            "model": data.get("model") or body.get("model"),
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "prompt_eval_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 1),
            "eval_ms": round(data.get("eval_duration", 0) / 1e6, 1),
            "load_ms": round(data.get("load_duration", 0) / 1e6, 1),
            "total_ms": round(data.get("total_duration", 0) / 1e6, 1),
            "done_reason": data.get("done_reason"),
            "num_ctx": opts.get("num_ctx"),
        }
        limit = (opts.get("num_ctx") or 0) - (opts.get("num_predict") or 0)
        prompt_tokens = usage["prompt_tokens"] or 0
        usage["context_overflow"] = bool(opts.get("num_ctx")) and prompt_tokens >= limit
        if usage["context_overflow"]:
            self.context_overflows += 1
            log.warning(
                "LLM prompt fills the context window (%s prompt tokens, num_ctx=%s, "
                "num_predict=%s): the start of the prompt may have been truncated; "
                "raise LLM_NUM_CTX", prompt_tokens, opts.get("num_ctx"), opts.get("num_predict"))
        self.last_usage = usage

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
        caps = self.capabilities(model)
        supports_think = caps is None or "thinking" in caps
        if model not in self._no_think_field and supports_think:
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
        sent = {}

        def call(b):
            sent["body"] = b
            return self._post(f"{self.base_url}/api/chat", b)

        data = self._with_think_fallback(call, body)
        self._record_usage(data, sent["body"])
        return strip_think(data.get("message", {}).get("content", ""))

    def stream_chat(self, messages, *, temperature=0.0, model=None, max_tokens=None, cancel=None):
        body = self._body(messages, model, temperature, max_tokens, False, stream=True)
        filt = ThinkFilter()

        def run(b):
            for line in self._stream_lines(f"{self.base_url}/api/chat", b, cancel):
                obj = json.loads(line)
                if obj.get("error"):
                    raise LLMError(str(obj["error"]),
                                   classify_error_text(None, str(obj["error"])))
                piece = filt.feed(obj.get("message", {}).get("content", ""))
                if piece:
                    yield piece
                if obj.get("done"):
                    self._record_usage(obj, b)
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
        """Load the model into RAM without generating (blocks until it is loaded)."""
        self.ensure_loaded()

    # -- model residency -------------------------------------------------------
    def is_loaded(self, model: str | None = None) -> bool | None:
        """True/False from /api/ps; None if the server cannot tell us."""
        model = self._model(model)
        try:
            resp = self._http.get(f"{self.base_url}/api/ps",
                                  timeout=settings.llm_ping_timeout_s * 2)
            if resp.status_code != 200:
                return None
            names = {m.get("name") for m in resp.json().get("models", [])} | \
                {m.get("model") for m in resp.json().get("models", [])}
        except Exception:  # noqa: BLE001
            return None
        return model in names or f"{model}:latest" in names

    def begin_load(self, model: str | None = None) -> "ModelLoad":
        """Start loading the model in a background thread (one load per model and
        server, shared by every request) and return its handle. The load is never
        tied to a user request: Ollama aborts a load when the requesting client
        disconnects, so a request that gives up must not take the load with it."""
        return _load_for(self, self._model(model))

    def ensure_loaded(self, model: str | None = None, *, cancel: threading.Event | None = None,
                      progress=None) -> dict:
        """Block until the model is resident (or LLM_LOAD_TIMEOUT_S), reporting progress
        every few seconds. Raises a classified LLMError on failure; returns
        {"waited_s", "loaded"} — loaded=False only when `cancel` was set.

        /api/ps cannot be trusted for this: Ollama lists a model as soon as its
        runner starts loading. A load request (empty prompt) is instead always made
        through the shared loader; for a resident model it returns at once."""
        handle = self.begin_load(self._model(model))
        while True:
            if handle.done.wait(settings.llm_load_poll_s):
                if handle.error is not None:
                    raise handle.error
                return {"waited_s": round(handle.elapsed(), 1), "loaded": True}
            if cancel is not None and cancel.is_set():
                return {"waited_s": round(handle.elapsed(), 1), "loaded": False}
            if progress is not None:
                progress(handle.elapsed())


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


class ModelLoad:
    """One background model load (see OllamaClient.begin_load)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.started = time.monotonic()
        self.done = threading.Event()
        self.error: LLMError | None = None
        self.finished = 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started


_loads: dict[tuple[str, str], ModelLoad] = {}
_loads_lock = threading.Lock()


def load_status() -> list[dict]:
    """Last model-load result per model (for /api/health): ok / loading / error code."""
    out = []
    with _loads_lock:
        handles = list(_loads.values())
    for h in handles:
        state = "loading" if not h.done.is_set() else ("error" if h.error else "ok")
        out.append({"model": h.model, "state": state, "seconds": round(h.elapsed(), 1)
                    if not h.done.is_set() else round(h.finished - h.started, 1),
                    "error": getattr(h.error, "code", None)})
    return out


def _load_for(client: "OllamaClient", model: str) -> ModelLoad:
    key = (client.base_url, model)
    with _loads_lock:
        current = _loads.get(key)
        if current is not None and not current.done.is_set():
            return current
        handle = ModelLoad(model)
        _loads[key] = handle

    def run() -> None:
        try:
            timeout = httpx.Timeout(settings.llm_load_timeout_s,
                                    connect=settings.llm_connect_timeout_s)
            data = client._post(f"{client.base_url}/api/generate",
                                {"model": model, "prompt": "",
                                 "keep_alive": settings.llm_keep_alive}, timeout=timeout)
            if data.get("error"):
                raise LLMError(str(data["error"]), classify_error_text(None, str(data["error"])))
            log.info("model %s loaded in %.1f s", model, handle.elapsed())
        except LLMError as exc:
            if exc.code == "model_timeout":
                exc = LLMError(f"loading {model} took longer than LLM_LOAD_TIMEOUT_S",
                               "model_load_timeout")
            handle.error = exc
            log.warning("model %s failed to load after %.1f s: %s", model, handle.elapsed(),
                        exc.code)
        except Exception as exc:  # noqa: BLE001
            handle.error = LLMError(f"model load failed ({exc.__class__.__name__})")
        finally:
            handle.finished = time.monotonic()
            handle.done.set()

    threading.Thread(target=run, name=f"askito-load-{model}", daemon=True).start()
    return handle


def make_client(mode: str, base_url: str, api_key: str, default_model: str = ""):
    if mode == "mock":
        return MockLLMClient()
    if settings.llm_local_only:
        ensure_local_url(base_url, f"LLM endpoint ({mode})", local_allowed_hosts())
    if mode == "openai":
        return OpenAICompatClient(base_url, api_key, default_model)
    if mode == "ollama":
        return OllamaClient(base_url, api_key, default_model)
    raise LLMError(f"unknown LLM_MODE: {mode}")


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    from app.llm.metrics import instrumented

    return instrumented(make_client(settings.llm_mode, settings.llm_base_url,
                                    settings.llm_api_key))


def reset_llm_client() -> None:
    get_llm_client.cache_clear()
