"""OpenAI-compatible LLM client abstraction.

Two modes (selected by LLM_MODE):
  * "ollama" : POST to an OpenAI-compatible /chat/completions endpoint
               (default: local Ollama). Model names come from config.
  * "mock"   : deterministic offline stub, no network/model required. Used for
               dev and tests so the pipeline can run without downloads.

Only the surface needed by M1 (ping + a basic chat) is implemented here; the
agent loop in M5 builds on the same interface.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from app.config import settings

Message = dict[str, str]


class LLMClient(Protocol):
    def ping(self) -> tuple[bool, str]: ...
    def chat(
        self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None
    ) -> str: ...


class OpenAICompatClient:
    """Talks to any OpenAI-compatible Chat Completions endpoint (e.g. Ollama)."""

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key

    def ping(self) -> tuple[bool, str]:
        try:
            resp = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=3.0,
            )
            if resp.status_code == 200:
                return True, f"reachable at {self.base_url}"
            return False, f"HTTP {resp.status_code} from {self.base_url}"
        except Exception as exc:  # noqa: BLE001
            return False, f"unreachable: {exc.__class__.__name__}"

    def chat(
        self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None
    ) -> str:
        model = model or settings.model_for("answer")
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=240.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class MockLLMClient:
    """Deterministic offline stub. Echoes a canned, structure-aware response."""

    def ping(self) -> tuple[bool, str]:
        return True, "mock mode (no external model)"

    def chat(
        self, messages: list[Message], *, temperature: float = 0.0, model: str | None = None
    ) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[mock] received {len(messages)} message(s); last: {last[:80]}"


def get_llm_client() -> LLMClient:
    if settings.llm_mode == "mock":
        return MockLLMClient()
    return OpenAICompatClient()
