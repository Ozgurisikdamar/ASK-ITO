"""Embedding model wrapper (sentence-transformers, CPU).

bge-v1.5 for s2p retrieval: passages are embedded as-is; queries get the
instruction prefix. Embeddings are L2-normalised so cosine == dot product and
pgvector's cosine distance is well-behaved.

Lazily loaded so importing this module (and running non-embedding code/tests)
does not pull in torch.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
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


# -- query-embedding cache ----------------------------------------------------
# L1: in-process LRU; L2: optional local Redis (float32 bytes). The key covers
# everything that changes the vector: model, dimension, query instruction and
# the exact (whitespace-normalised) query text.
_emb_cache: OrderedDict[str, list[float]] = OrderedDict()
_emb_lock = threading.Lock()
EMBED_CACHE_VERSION = "1"


def embedding_cache_key(text: str) -> str:
    from app.redis_cache import hash_key

    return hash_key((EMBED_CACHE_VERSION, settings.embedding_model, settings.embedding_dim,
                     settings.query_instruction, " ".join(text.split())))


def embedding_cache_clear() -> None:
    with _emb_lock:
        _emb_cache.clear()


def _l1_put(key: str, vec: list[float]) -> None:
    if settings.embedding_cache_size <= 0:
        return
    with _emb_lock:
        _emb_cache[key] = vec
        _emb_cache.move_to_end(key)
        while len(_emb_cache) > settings.embedding_cache_size:
            _emb_cache.popitem(last=False)


def cached_query_embedding(text: str) -> tuple[list[float] | None, str | None]:
    """(vector, tier) from the L1/L2 cache, or (None, None) on a miss."""
    if settings.embedding_cache_size <= 0 and not settings.redis_enabled:
        return None, None
    key = embedding_cache_key(text)
    with _emb_lock:
        hit = _emb_cache.get(key)
        if hit is not None:
            _emb_cache.move_to_end(key)
            return list(hit), "l1"
    from app.redis_cache import cache

    raw = cache.get_bytes("emb", key)
    if raw is not None and len(raw) == 4 * settings.embedding_dim:
        import numpy as np

        vec = np.frombuffer(raw, dtype=np.float32).astype(float).tolist()
        _l1_put(key, vec)
        return list(vec), "l2"
    return None, None


def store_query_embedding(text: str, vec: list[float]) -> None:
    key = embedding_cache_key(text)
    _l1_put(key, list(vec))
    from app.redis_cache import cache

    if cache.enabled:
        import numpy as np

        cache.set_bytes("emb", key, np.asarray(vec, dtype=np.float32).tobytes())


def embed_query(text: str) -> list[float]:
    return embed_query_ex(text)[0]


def embed_query_ex(text: str) -> tuple[list[float], str]:
    """(vector, source) where source is "l1", "l2" (Redis) or "model"."""
    vec, tier = cached_query_embedding(text)
    if vec is not None:
        return vec, tier
    model = _model()
    with inference_lock:
        arr = model.encode(
            settings.query_instruction + text,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
    out = arr.tolist()
    store_query_embedding(text, out)
    return out, "model"


def warmup() -> None:
    # bypass the cache: the point is to load the model and run one forward pass
    model = _model()
    with inference_lock:
        model.encode(settings.query_instruction + "warm up", normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
