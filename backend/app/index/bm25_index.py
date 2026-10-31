"""BM25 index over chunk texts using `bm25s` (spec §5 step 1).

English stopword removal + stemming; k1/b come from config. The index is saved
to data/bm25/ together with the chunk_id mapping (bm25s works on corpus
positions, which we map back to chunk ids). The same tokenizer settings are used
for indexing and querying (consistent analysis, per the Qdrant-1.20 note).
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import bm25s

from app.config import settings

BM25_DIR = Path(__file__).resolve().parents[2] / "data" / "bm25"
IDS_FILE = BM25_DIR / "chunk_ids.json"
META_FILE = BM25_DIR / "_meta.json"


@lru_cache(maxsize=1)
def _stemmer():
    try:
        import Stemmer  # PyStemmer

        return Stemmer.Stemmer("english")
    except Exception:  # noqa: BLE001
        return None


def index_version() -> str:
    """Changes whenever `make index` rebuilds the index (used to invalidate caches)."""
    try:
        st = META_FILE.stat()
        return f"{st.st_mtime_ns}-{st.st_size}"
    except FileNotFoundError:
        return "none"


def _tokenize(texts: list[str]):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), show_progress=False)


def build(chunk_ids: list[int], texts: list[str]) -> int:
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    tokens = _tokenize(texts)
    retriever = bm25s.BM25(k1=settings.bm25_k1, b=settings.bm25_b)
    retriever.index(tokens, show_progress=False)
    retriever.save(str(BM25_DIR))
    IDS_FILE.write_text(json.dumps(chunk_ids), encoding="utf-8")
    META_FILE.write_text(
        json.dumps({"count": len(chunk_ids), "k1": settings.bm25_k1, "b": settings.bm25_b}),
        encoding="utf-8",
    )
    return len(chunk_ids)


class Bm25Searcher:
    def __init__(self) -> None:
        self.retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=False)
        self.chunk_ids: list[int] = json.loads(IDS_FILE.read_text(encoding="utf-8"))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.chunk_ids))
        if k == 0:
            return []
        q = _tokenize([query])
        idxs, scores = self.retriever.retrieve(q, k=k, show_progress=False)
        out: list[tuple[int, float]] = []
        for pos, score in zip(idxs[0], scores[0]):
            out.append((self.chunk_ids[int(pos)], float(score)))
        return out


def exists() -> bool:
    return IDS_FILE.exists() and META_FILE.exists()


_searcher: Bm25Searcher | None = None
_searcher_version: str | None = None
_searcher_lock = threading.Lock()


def get_searcher() -> Bm25Searcher:
    """Process-wide searcher, reloaded when the index version changes."""
    global _searcher, _searcher_version
    version = index_version()
    if _searcher is None or _searcher_version != version:
        with _searcher_lock:
            if _searcher is None or _searcher_version != version:
                _searcher = Bm25Searcher()
                _searcher_version = version
    return _searcher
