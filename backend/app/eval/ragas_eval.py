"""RAGAS generation-quality metrics (spec §8.2, mandatory component).

Computes faithfulness, answer relevancy, context precision and context recall
(plus answer correctness) for a batch of answered questions, using the project's
LLM as the judge and the project's embedding model.

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


def _judge():
    """Build a ragas LLM judge + embeddings from project config."""
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    model = settings.model_for("judge") or settings.model_for("answer")
    chat = ChatOpenAI(
        model=model, base_url=settings.llm_base_url,
        api_key=settings.llm_api_key, temperature=0,
    )
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def compute(samples: list[dict]) -> dict:
    """samples: [{question, answer, contexts:[str], reference}]. Returns mean
    scores for the four RAGAS metrics (+ answer_correctness), or None each."""
    answerable = [s for s in samples if s.get("reference")]
    if not answerable:
        return _empty("no answerable samples with reference")
    if settings.llm_mode == "mock":
        return _empty("LLM judge unavailable (mock mode)")
    ok, msg = settings_llm_ping()
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
            llm=llm, embeddings=emb,
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
