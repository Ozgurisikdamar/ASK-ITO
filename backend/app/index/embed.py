"""Embedding model wrapper (sentence-transformers, CPU).

bge-v1.5 for s2p retrieval: passages are embedded as-is; queries get the
instruction prefix. Embeddings are L2-normalised so cosine == dot product and
pgvector's cosine distance is well-behaved.

Lazily loaded so importing this module (and running non-embedding code/tests)
does not pull in torch.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from app.config import settings
from app.ml import configure_torch, inference_lock


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    configure_torch()
    return SentenceTransformer(settings.embedding_model, device="cpu")


def token_counter() -> Callable[[str], int]:
    """Return a function counting tokens with the embedding model's tokenizer."""
    tok = _model().tokenizer

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vecs = _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    model = _model()
    with inference_lock:
        vec = model.encode(
            settings.query_instruction + text,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
    return vec.tolist()


def warmup() -> None:
    embed_query("warm up")
