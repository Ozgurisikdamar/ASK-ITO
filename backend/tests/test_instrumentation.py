"""Request instrumentation: call accounting by role, tokens, errors; never changes outputs."""
import httpx
import psycopg
import pytest

from app.config import settings
from app.index import bm25_index
from app.llm.client import MockLLMClient, OllamaClient
from app.llm.metrics import CallRecorder, instrumented, llm_role, recording


class _UsageLLM:
    """Fake provider that reports Ollama-style usage."""

    def __init__(self, load_ms=5.0):
        self.last_usage = {}
        self.load_ms = load_ms

    def chat(self, messages, **kw):
        self.last_usage = {"model": "m1", "prompt_tokens": 100, "completion_tokens": 20,
                           "eval_ms": 400.0, "prompt_eval_ms": 200.0, "load_ms": self.load_ms,
                           "context_overflow": False}
        return "out"

    def stream_chat(self, messages, **kw):
        yield "a"
        yield "b"
        self.last_usage = {"model": "m1", "prompt_tokens": 50, "completion_tokens": 2}


def test_recorder_counts_calls_by_role_with_tokens():
    llm = instrumented(_UsageLLM())
    with recording() as rec:
        with llm_role("analyze"):
            llm.chat([{"role": "user", "content": "x"}], model="m1", json_mode=True)
        with llm_role("answer"):
            assert "".join(llm.stream_chat([{"role": "user", "content": "x"}])) == "ab"
    s = rec.summary()
    assert s["calls"] == 2 and set(s["by_role"]) == {"analyze", "answer"}
    assert s["prompt_tokens"] == 150 and s["completion_tokens"] == 22
    assert s["tokens_reported"] is True and s["errors"] == 0 and s["cold_loads"] == 0
    stream_rec = rec.calls[1]
    assert "ttft_ms" in stream_rec and stream_rec["role"] == "answer"
    assert "content" not in str(rec.calls)  # no prompt text is recorded


def test_cold_load_and_unknown_tokens_flagged():
    llm = instrumented(_UsageLLM(load_ms=5000.0))
    with recording() as rec:
        llm.chat([{"role": "user", "content": "x"}])
        instrumented(MockLLMClient()).chat([{"role": "user", "content": "x"}])
    assert rec.calls[0]["cold"] is True
    assert rec.calls[1]["tokens"] == "unknown"
    assert rec.summary()["tokens_reported"] is False and rec.summary()["prompt_tokens"] == 100


def test_errors_and_timeouts_recorded_and_reraised():
    class Slow:
        def chat(self, messages, **kw):
            raise httpx.ReadTimeout("slow")

    with recording() as rec:
        with pytest.raises(httpx.ReadTimeout):
            instrumented(Slow()).chat([{"role": "user", "content": "x"}])
    s = rec.summary()
    assert s["errors"] == 1 and s["timeouts"] == 1


def test_no_recorder_means_no_side_effects_and_same_output():
    plain, wrapped = MockLLMClient(), instrumented(MockLLMClient())
    msgs = [{"role": "user", "content": "same input"}]
    assert plain.chat(msgs) == wrapped.chat(msgs)
    assert instrumented(wrapped) is wrapped  # idempotent


def test_wrapper_delegates_attributes():
    c = OllamaClient("http://127.0.0.1:11434", default_model="m")
    w = instrumented(c)
    assert w.default_model == "m" and w.context_overflows == 0


def test_recorder_notes():
    rec = CallRecorder()
    rec.note("context_trim", {"chars_trimmed": 10})
    assert rec.summary()["notes"]["context_trim"]["chars_trimmed"] == 10


def test_context_trim_is_recorded_without_changing_context():
    from app.agent.answer import Source, build_context
    from app.retrieval.hydrate import ChunkInfo

    long = "Title\n\n" + "x" * 1500
    srcs = [Source(n=1, chunk_id=1, info=ChunkInfo(1, 1, "T", "", "u", "ai", long))]
    plain = build_context(srcs, max_chars=900)
    with recording() as rec:
        again = build_context(srcs, max_chars=900)
    assert plain == again
    trim = rec.notes["context_trim"]
    assert trim["trimmed_sources"] == 1 and trim["chars_trimmed"] == 600
    assert trim["chars_kept"] == 900


# -- integration (real retrieval; scripted LLM) -------------------------------

def _index_ready() -> bool:
    if not bm25_index.exists():
        return False
    try:
        with psycopg.connect(settings.database_url) as c:
            return c.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except Exception:  # noqa: BLE001
        return False


needs_index = pytest.mark.skipif(not _index_ready(), reason="index/DB not built")


@needs_index
def test_agent_metrics_attribute_calls_to_roles(monkeypatch):
    from tests.test_agent import HOPFIELD_Q, ScriptedLLM

    from app.agent.loop import run_agent
    from app.config import settings

    # The lean defaults skip rewrite and remove the sufficiency loop; enable both
    # so this test can assert every role is attributed.
    monkeypatch.setattr(settings, "agent_rewrite_policy", "always")
    monkeypatch.setattr(settings, "agent_max_rounds", 3)

    res = run_agent(HOPFIELD_Q, domain="all", final_k=5, llm=ScriptedLLM("A [1]."))
    m = res["metrics"]
    roles = m["llm"]["by_role"]
    assert {"analyze", "rewrite", "sufficiency", "answer"} <= set(roles)
    assert m["llm"]["calls"] == res["llm_calls"]
    assert m["agent"]["subquestions"] == 2 and m["agent"]["rounds"] >= 1
    assert m["candidates"]["context_chunks"] == len(res["sources"])
    assert {"search", "final_rerank"} <= set(m["retrieval_ms"])


@needs_index
def test_instrumentation_does_not_change_agent_output():
    from tests.test_agent import HOPFIELD_Q, ScriptedLLM

    from app.agent.loop import run_agent
    from app.retrieval import pipeline

    pipeline.cache_clear()
    a = run_agent(HOPFIELD_Q, domain="all", final_k=5, llm=ScriptedLLM("A [1]. B [2]."))
    pipeline.cache_clear()
    with recording():
        b = run_agent(HOPFIELD_Q, domain="all", final_k=5, llm=ScriptedLLM("A [1]. B [2]."))
    for key in ("answer", "citations", "sources", "rounds", "llm_calls"):
        assert a[key] == b[key]


@needs_index
def test_run_search_metrics_schema():
    from app.retrieval.service import run_search

    res = run_search("transmon qubit charge noise", mode="hybrid_rerank", final_k=5)
    m = res["metrics"]
    assert m["mode"] == "hybrid_rerank" and m["total_ms"] >= 0
    assert {"bm25", "vector", "rrf", "rerank"} <= set(m["retrieval_ms"])
    assert m["candidates"]["reranked"] == min(m["candidates"]["fused"], settings.rerank_pool)
    assert m["candidates"]["context_chunks"] == len(res["sources"])
    assert m["llm"]["calls"] == 1 and m["llm"]["by_role"] == {"answer": m["llm"]["by_role"]["answer"]}
