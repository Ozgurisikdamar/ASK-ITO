"""RAGAS generation-quality metrics (spec §8.2, mandatory component).

Computes faithfulness, answer relevancy, context precision and context recall
(plus answer correctness) for a batch of answered questions, using a local LLM
as the judge (LLM_JUDGE_MODEL role, through the project's own LLM client) and
the project's embedding model. Jobs run one at a time (single local model).

RAGAS needs a working LLM judge + embeddings. When they are unavailable (e.g.
no local Ollama, or ragas not installed) this returns the four metric keys as
None with a status string, so an eval run still completes and the panel shows
the retrieval metrics; on a machine with a local LLM it returns real scores.

Pinned to ragas==0.2.14 (API is version-sensitive; see docs/DECISIONS.md).
"""
from __future__ import annotations

from app.config import settings

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _empty(status: str) -> dict:
    out = {k: None for k in METRIC_KEYS}
    out["answer_correctness"] = None
    out["ragas_status"] = status
    return out


def available() -> tuple[bool, str]:
    try:
        import ragas  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"ragas not installed: {exc.__class__.__name__}"
    ok, msg = settings_llm_ping()
    return ok, msg


def settings_llm_ping() -> tuple[bool, str]:
    from app.llm.client import get_llm_client

    return get_llm_client().ping()


def judge_client():
    """The judge's own LLM client (LLM_JUDGE_MODEL role, same local endpoint)."""
    from app.llm.client import make_client

    model = settings.model_for("judge") or settings.model_for("answer")
    return make_client(settings.llm_mode, settings.llm_base_url, settings.llm_api_key, model), model


def _judge():
    """Build a ragas LLM judge + embeddings from project config.

    The judge runs through the project's own LLM client (Ollama native API with
    thinking off, keep_alive and JSON mode), not through an OpenAI client, so it
    honours the same local-only endpoint policy and settings as the app.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    client, model = judge_client()
    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return make_judge_llm(client, model), LangchainEmbeddingsWrapper(emb)


def make_judge_llm(client, model: str):
    """A ragas BaseRagasLLM backed by a project LLM client (sequential, JSON mode)."""
    import asyncio
    from dataclasses import dataclass, field

    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    @dataclass
    class ProjectJudgeLLM(BaseRagasLLM):
        calls: list = field(default_factory=list, repr=False)

        def is_finished(self, response) -> bool:  # local models: no finish_reason parsing
            return True

        def _complete(self, prompt, temperature) -> str:
            import time

            t0 = time.perf_counter()
            text = client.chat(
                [{"role": "user", "content": prompt.to_string()}],
                temperature=float(temperature or 0.0), model=model, json_mode=True,
                max_tokens=settings.llm_judge_max_tokens,
            )
            usage = dict(getattr(client, "last_usage", {}) or {})
            usage["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.calls.append(usage)
            return text

        def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
            return LLMResult(generations=[[Generation(text=self._complete(prompt, temperature))
                                           for _ in range(max(1, n))]])

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            return await asyncio.to_thread(self.generate_text, prompt, n,
                                           temperature or 1e-8, stop, callbacks)

    return ProjectJudgeLLM()


def run_config():
    """Single worker (one local model, one request at a time) and a long timeout."""
    from ragas.run_config import RunConfig

    return RunConfig(timeout=settings.ragas_timeout_s, max_workers=1, max_retries=2,
                     max_wait=10)


def compute(samples: list[dict]) -> dict:
    """samples: [{question, answer, contexts:[str], reference}]. Returns mean
    scores for the four RAGAS metrics (+ answer_correctness), or None each."""
    answerable = [s for s in samples if s.get("reference")]
    if not answerable:
        return _empty("no answerable samples with reference")
    if settings.llm_mode == "mock":
        return _empty("LLM judge unavailable (mock mode)")
    try:
        client, _model = judge_client()
        ok, msg = client.ping()
    except Exception as exc:  # noqa: BLE001  (e.g. non-local endpoint refused)
        return _empty(f"LLM judge unavailable: {exc}")
    if not ok:
        return _empty(f"LLM judge unreachable: {msg}")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness, answer_relevancy, context_precision,
            context_recall, faithfulness,
        )

        llm, emb = _judge()
        ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "reference": s["reference"],
            }
            for s in answerable
        ])
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision,
                     context_recall, answer_correctness],
            llm=llm, embeddings=emb, run_config=run_config(),
            raise_exceptions=False, show_progress=False,
        )
        import math

        df = result.to_pandas()
        out = {}
        for key in METRIC_KEYS + ["answer_correctness"]:
            if key in df:
                v = float(df[key].mean())
                out[key] = None if math.isnan(v) else v
            else:
                out[key] = None
        has_values = any(out[k] is not None for k in METRIC_KEYS)
        out["ragas_status"] = "ok" if has_values else "no judge values (LLM unavailable)"
        return out
    except Exception as exc:  # noqa: BLE001
        return _empty(f"unavailable: {exc.__class__.__name__}: {exc}")
