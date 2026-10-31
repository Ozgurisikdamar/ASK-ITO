"""Startup warm-up and background pre-computation.

1. Warm-up: load the chunk store, BM25 index, embedder and reranker (and load
   the Ollama model into RAM) so the first user search is as fast as later ones.
2. Prewarm: compute and cache the overview answers for the featured queries,
   so clicking a suggestion chip is instant. It always yields to real users: a
   user search pauses it and abandons its in-flight answer, which is retried
   once the user is idle again. Answers persist in `answer_cache`, so this
   work happens once per model/prompt/index version.
"""
from __future__ import annotations

import logging
import threading
import time

from app.config import settings

log = logging.getLogger("askito.warmup")

FEATURED_QUERIES = [
    "Who shared the 2020 Chemistry prize and for what?",
    "How is the 2025 Physics prize related to quantum computers?",
    "What did Katalin Karikó and Drew Weissman win the Nobel for?",
    "Compare the 2022 and 2023 Physics laureates' discoveries.",
    "What is a transmon qubit?",
    "Which 2024 laureates worked on protein structure?",
]

state: dict = {
    "status": "idle",  # idle | warming | ready | error
    "steps": {},
    "error": None,
    "prewarm": {"done": 0, "total": len(FEATURED_QUERIES), "status": "idle"},
}


class Activity:
    """Tracks in-flight user searches so background work can step aside."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.preempt = threading.Event()

    def enter(self) -> None:
        with self._lock:
            self._n += 1
            self.idle.clear()
            self.preempt.set()

    def exit(self) -> None:
        with self._lock:
            self._n = max(0, self._n - 1)
            if self._n == 0:
                self.idle.set()


activity = Activity()


def _timed(name: str, fn) -> None:
    t = time.perf_counter()
    fn()
    state["steps"][name] = round(time.perf_counter() - t, 2)


def run_warmup() -> None:
    from app.index import bm25_index, embed
    from app.retrieval import reranker
    from app.retrieval.store import store

    state["status"] = "warming"
    try:
        if bm25_index.exists():
            _timed("chunk_store", store.ensure_loaded)
            _timed("bm25", bm25_index.get_searcher)
        _timed("embedding", embed.warmup)
        _timed("reranker", reranker.warmup)
        state["status"] = "ready"
    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["error"] = f"{exc.__class__.__name__}: {exc}"
        log.exception("warm-up failed")
        return
    try:  # best effort: load the local model into RAM
        from app.llm.client import get_llm_client

        _timed("llm", get_llm_client().warmup)
    except Exception as exc:  # noqa: BLE001
        state["steps"]["llm"] = f"skipped ({exc.__class__.__name__})"


def prewarm_featured() -> None:
    from app.agent.answer import stream_answer
    from app.index import bm25_index
    from app.retrieval import answer_cache, service

    if not bm25_index.exists():
        return
    pre = state["prewarm"]
    pre["status"] = "running"
    for query in FEATURED_QUERIES:
        while True:
            activity.idle.wait()
            activity.preempt.clear()
            try:
                sources = service.prepare_sources(query, "overview")
                if answer_cache.get(query, "overview", "all"):
                    break
                done = None
                for kind, payload in stream_answer(query, sources, mode="overview",
                                                   cancel=activity.preempt):
                    if kind == "done":
                        done = payload
                if done is None:  # preempted by a user search: retry when idle
                    continue
                answer_cache.put(query, "overview", "all", done["answer"], done["citations"],
                                 [s.chunk_id for s in sources])
                break
            except Exception as exc:  # noqa: BLE001
                pre["status"] = f"error: {exc.__class__.__name__}"
                return
        pre["done"] += 1
    pre["status"] = "done"


def start_background() -> threading.Thread | None:
    if not settings.warmup_on_start:
        return None

    def work():
        run_warmup()
        if settings.prewarm_featured and state["status"] == "ready":
            prewarm_featured()

    t = threading.Thread(target=work, name="askito-warmup", daemon=True)
    t.start()
    return t
