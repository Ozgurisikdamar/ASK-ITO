"""Model tag and inference-runtime configuration: one source for the model tag,
Q8_0 weights by default with the Q4_K_M baseline kept, F16 KV cache in compose,
and the served quantization read from the runtime (never from the tag text)."""
import re
from pathlib import Path

import httpx
import yaml

from app.config import Settings, settings
from app.llm.client import OllamaClient

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "backend" / "app"


def test_default_model_is_the_q8_tag_and_roles_fall_back_to_it(monkeypatch):
    assert Settings.model_fields["llm_default_model"].default == "qwen3.5:9b-q8_0"
    monkeypatch.setattr(settings, "llm_answer_model", "")
    monkeypatch.setattr(settings, "llm_agent_model", "")
    assert settings.model_for("answer") == settings.model_for("agent") == \
        settings.llm_default_model


def test_model_tags_are_not_hardcoded_outside_config():
    literal = re.compile(r"""["'](qwen|llama|mistral|gemma|phi)[\w.\-]*:[\w.\-]+["']""")
    offenders = [str(p.relative_to(ROOT)) for p in APP.rglob("*.py")
                 if p.name != "config.py" and literal.search(p.read_text())]
    assert offenders == []


def test_env_example_keeps_q8_default_and_q4_rollback_and_no_lower_quant():
    env = (ROOT / ".env.example").read_text()
    assert re.search(r"^LLM_DEFAULT_MODEL=qwen3\.5:9b-q8_0$", env, re.M)
    assert "qwen3.5:9b-q4_K_M" in env  # baseline / rollback documented
    assert not re.search(r"\b(q6_K|q5_K_M|q3_K|q2_K)\b", env, re.I)


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def test_compose_kv_cache_is_f16_by_default_with_flash_attention():
    env = _compose()["ollama"]["environment"]
    assert env["OLLAMA_KV_CACHE_TYPE"] == "${OLLAMA_KV_CACHE_TYPE:-f16}"
    assert str(env["OLLAMA_FLASH_ATTENTION"]) == "1"
    assert str(env["OLLAMA_NUM_PARALLEL"]) == "1" and env["OLLAMA_KEEP_ALIVE"] == "24h"


def test_compose_redis_is_local_and_never_published():
    svc = _compose()
    redis = svc["redis"]
    assert "ports" not in redis  # reachable only on the compose network
    assert "--maxmemory-policy" in redis["command"] and "--appendonly" in redis["command"]
    be = svc["backend"]["environment"]
    assert be["REDIS_URL"].startswith("redis://redis:") and be["REDIS_ALLOWED_HOSTS"] == "redis"


def test_model_identity_reports_runtime_quantization_and_digest():
    def handler(req: httpx.Request):
        if req.url.path == "/api/show":
            return httpx.Response(200, json={
                "capabilities": ["completion"],
                "details": {"quantization_level": "Q8_0", "parameter_size": "9.0B",
                            "family": "qwen35"}})
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3.5:9b-q8_0", "digest": "72b5aa6062ba0000"},
                {"name": "qwen3.5:9b", "digest": "56671c2ab9380000"}]})
        return httpx.Response(404)

    c = OllamaClient("http://127.0.0.1:11434")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    ident = c.model_identity("qwen3.5:9b-q8_0")
    assert ident["quantization"] == "Q8_0" and ident["digest"].startswith("72b5aa")


def test_model_identity_unknown_when_runtime_is_down():
    def handler(req):
        raise httpx.ConnectError("refused")

    c = OllamaClient("http://127.0.0.1:11434")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    ident = c.model_identity("qwen3.5:9b-q8_0")
    assert ident["quantization"] is None and ident["digest"] is None
