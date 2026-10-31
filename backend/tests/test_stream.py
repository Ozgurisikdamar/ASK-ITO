"""Streaming answers, the SSE endpoint, caches and background-activity tracking."""
import json
import threading

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.agent.answer import NOT_FOUND, Source, generate_answer, stream_answer
from app.api import stream as stream_api
from app.config import settings
from app.index import bm25_index
from app.llm.client import MockLLMClient
from app.main import app
from app.retrieval.hydrate import ChunkInfo
from app.warmup import Activity

client = TestClient(app)


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        if not lines:
            continue
        event = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = json.loads(next(ln[6:] for ln in lines if ln.startswith("data: ")))
        out.append((event, data))
    return out


def _src(n, cid, text):
    return Source(n=n, chunk_id=cid, info=ChunkInfo(cid, cid, "Title", "", "u", "ai", text))


# -- answers -----------------------------------------------------------------

def test_no_sources_returns_not_found_instead_of_closed_book():
    answer, cites = generate_answer("anything", [], llm=MockLLMClient(), mode="overview")
    assert answer == NOT_FOUND and cites == []
    events = list(stream_answer("anything", [], llm=MockLLMClient(), mode="hybrid_rerank"))
    assert events[-1] == ("done", {"answer": NOT_FOUND, "citations": []})


def test_stream_answer_yields_tokens_then_validated_done():
    class Cited(MockLLMClient):
        def stream_chat(self, messages, **kw):
            yield from ["A transmon is a qubit ", "[1]", " and more [9]."]

    events = list(stream_answer("q", [_src(1, 5, "Transmon text")], llm=Cited(), mode="overview"))
    kinds = [k for k, _ in events]
    assert kinds[:-1] == ["token"] * 3 and kinds[-1] == "done"
    done = events[-1][1]
    assert "[9]" not in done["answer"]  # invalid citation stripped in the final answer
    assert [c["n"] for c in done["citations"]] == [1]


def test_stream_answer_stops_on_cancel():
    cancel = threading.Event()
    cancel.set()
    events = list(stream_answer("q", [_src(1, 5, "text")], llm=MockLLMClient(),
                                mode="overview", cancel=cancel))
    assert all(k != "token" for k, _ in events)


# -- SSE endpoint (no index needed) ------------------------------------------

def test_sse_endpoint_streams_events_in_order(monkeypatch):
    def fake_produce(q, mode, domain, cancel):
        yield "results", {"results": [{"chunk_id": 1}], "total": 1}
        yield "sources", {"sources": []}
        yield "token", {"t": "Hello"}
        yield "token", {"t": " world"}
        yield "done", {"answer": "Hello world", "citations": [], "mode": mode}

    monkeypatch.setattr(stream_api, "produce", fake_produce)
    resp = client.get("/api/search/stream", params={"q": "x", "mode": "overview"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    events = _events(resp.text)
    assert [e for e, _ in events] == ["results", "sources", "token", "token", "done"]
    assert "".join(d["t"] for e, d in events if e == "token") == "Hello world"


def test_sse_worker_error_becomes_error_event(monkeypatch):
    def boom(q, mode, domain, cancel):
        yield "results", {"results": [], "total": 0}
        raise RuntimeError("db down")

    monkeypatch.setattr(stream_api, "produce", boom)
    events = _events(client.get("/api/search/stream", params={"q": "x"}).text)
    assert events[-1][0] == "error"
    assert events[-1][1] == {"error": "search_failed", "detail": "db down"}


def test_sse_rejects_bad_mode_with_error_event():
    events = _events(client.get("/api/search/stream", params={"q": "x", "mode": "nope"}).text)
    assert events[0][0] == "error" and events[0][1]["error"] == "bad_request"


def test_featured_suggestions_endpoint():
    body = client.get("/api/suggest/featured").json()
    assert len(body["queries"]) >= 3 and all(isinstance(q, str) for q in body["queries"])


def test_activity_tracks_in_flight_searches():
    a = Activity()
    assert a.idle.is_set()
    a.enter()
    a.enter()
    assert not a.idle.is_set() and a.preempt.is_set()
    a.exit()
    assert not a.idle.is_set()
    a.exit()
    assert a.idle.is_set()


# -- integration: real index + DB (skipped on a fresh checkout) ---------------

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
def test_chunk_store_mirrors_db_and_filters_domain():
    from app.retrieval.store import store

    store.ensure_loaded()
    with psycopg.connect(settings.database_url) as c:
        n = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
        cid, domain = c.execute("SELECT id, domain FROM chunks LIMIT 1").fetchone()
    assert store.size == n
    meta = store.get(cid)
    assert meta is not None and meta.domain == domain and meta.doc_title and meta.url
    pairs = [(cid, 1.0)]
    assert store.filter_domain(pairs, domain) == pairs
    other = next(d for d in ("nobel", "quantum", "ai") if d != domain)
    assert store.filter_domain(pairs, other) == []
    assert cid in store.doc_chunk_ids(meta.doc_id)


@needs_index
def test_answer_cache_roundtrip():
    from app.retrieval import answer_cache

    q = "answer cache roundtrip test query"
    answer_cache.put(q, "overview", "all", "Cached answer [1].", [{"n": 1}], [1, 2])
    hit = answer_cache.get(q.upper() + "  ", "overview", "all")  # normalised key
    assert hit == {"answer": "Cached answer [1].", "citations": [{"n": 1}], "chunk_ids": [1, 2]}
    assert answer_cache.get(q, "overview", "nobel") is None
    with psycopg.connect(settings.database_url) as c:
        c.execute("DELETE FROM answer_cache WHERE query = %s", (q,))


@needs_index
def test_sse_overview_full_sequence_then_cached():
    from app.retrieval import pipeline

    pipeline.cache_clear()
    q = "transmon qubit superconducting charge noise"
    params = {"q": q, "mode": "overview", "domain": "all"}
    first = _events(client.get("/api/search/stream", params=params).text)
    kinds = [e for e, _ in first]
    assert kinds[:3] == ["results", "reranked", "sources"]
    assert kinds[-1] == "done" and "token" in kinds
    results = first[0][1]
    assert results["results"] and results["stage"] == "hybrid"
    assert {"chunk_id", "doc_title", "url", "snippet", "domain"} <= set(results["results"][0])
    sources = first[2][1]["sources"]
    assert 1 <= len(sources) <= settings.overview_context_chunks
    assert first[-1][1]["answer"]

    second = _events(client.get("/api/search/stream", params=params).text)
    assert second[0][0] == "results" and second[0][1]["cached"] is True
    assert "reranked" not in [e for e, _ in second]
    assert second[0][1]["total"] == results["total"]  # cache keeps the full list
    assert second[-1][1]["cached"] is True  # answer served from answer_cache
