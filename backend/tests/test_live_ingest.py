"""Dynamic Wikipedia ingestion (app.ingest.live).

Unit tests use a fake Wikipedia client and a fake database connection; the
integration test runs against a throw-away copy of the PostgreSQL database and a
temporary BM25 directory in a subprocess (real chunker, real embedding model,
real pgvector + BM25), and is skipped when such a copy cannot be created."""
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from app.config import settings
from app.ingest import live
from app.ingest.wiki import PageResult, WikiClient

LONG_TEXT = ("The transmon is a superconducting qubit. " * 80).strip()


def _page(title, pageid, revid=1, text=LONG_TEXT, **kw):
    return PageResult(requested_title=title, pageid=pageid, revid=revid, title=title,
                      url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                      extract=text, **kw)


class FakeWiki:
    def __init__(self, titles, pages, fail_search=False):
        self.titles, self.pages, self.fail_search = titles, pages, fail_search
        self.fetched = []

    def search_titles(self, query, limit=5):
        if self.fail_search:
            raise httpx.ConnectError("offline")
        return self.titles[:limit]

    def fetch_page(self, title):
        self.fetched.append(title)
        return self.pages[title]

    def close(self):
        pass


class FakeConn:
    """Enough of a psycopg connection for live.py: records statements."""

    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.db["sql"].append(sql.split()[0])
        if sql.startswith("SELECT id, revid, origin FROM documents"):
            row = self.db["docs"].get(params[0])
            return _Res([row] if row else [])
        if "INSERT INTO documents" in sql:
            pageid, revid = params[0], params[1]
            doc_id = self.db["docs"].get(pageid, (len(self.db["docs"]) + 100,))[0]
            self.db["docs"][pageid] = (doc_id, revid, "dynamic")
            return _Res([(doc_id,)])
        if sql.startswith("SELECT id, text FROM chunks"):
            return _Res([(i, t) for i, t in enumerate(self.db["chunks"])])
        return _Res([])

    def cursor(self):
        db = self.db

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def executemany(self, sql, rows):
                db["chunks"].extend(r[3] for r in rows)

        return Cur()

    def commit(self):
        self.db["commits"] += 1


class _Res:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


@pytest.fixture
def db(monkeypatch):
    state = {"docs": {}, "chunks": [], "sql": [], "commits": 0}
    built = []
    monkeypatch.setattr("app.index.bm25_index.build", lambda ids, texts: built.append(len(ids)))
    state["built"] = built
    return state


def _run(db, wiki, **kw):
    return live.ingest_for_query(
        "what is a transmon", client=wiki, connect=lambda: FakeConn(db),
        embed_fn=lambda texts: [[0.0] * settings.embedding_dim for _ in texts],
        count_tokens=lambda t: len(t.split()), save_raw=False, **kw)


def test_adds_articles_and_rebuilds_bm25(db, monkeypatch):
    monkeypatch.setattr(settings, "live_ingest_max_articles", 2)
    wiki = FakeWiki(["Transmon", "Qubit", "Charge qubit"],
                    {t: _page(t, i) for i, t in enumerate(["Transmon", "Qubit", "Charge qubit"], 1)})
    stages = []
    res = _run(db, wiki, progress=lambda st, d: stages.append(st))
    assert [a["title"] for a in res.added] == ["Transmon", "Qubit"]  # bounded to 2
    assert wiki.fetched == ["Transmon", "Qubit"] and res.error is None
    assert all(a["chunks"] > 0 for a in res.added) and db["commits"] == 2
    assert db["built"] == [len(db["chunks"])]  # BM25 rebuilt once, from the database
    assert stages[0] == "wiki_search" and "wiki_store" in stages


def test_idempotent_and_never_rewrites_the_snapshot(db):
    pages = {"Transmon": _page("Transmon", 1, revid=7), "Qubit": _page("Qubit", 2)}
    db["docs"][1] = (11, 7, "dynamic")  # same revision already stored
    db["docs"][2] = (12, 99, "snapshot")  # curated snapshot article
    res = _run(db, FakeWiki(["Transmon", "Qubit"], pages))
    assert res.added == [] and db["built"] == []
    assert {s["reason"] for s in res.skipped} == {"already in the corpus"}


def test_newer_revision_of_a_dynamic_article_is_refreshed(db):
    db["docs"][1] = (11, 6, "dynamic")
    res = _run(db, FakeWiki(["Transmon"], {"Transmon": _page("Transmon", 1, revid=7)}))
    assert [a["title"] for a in res.added] == ["Transmon"] and "DELETE" in db["sql"]


def test_skips_disambiguation_missing_and_short_pages(db):
    pages = {"A": _page("A", 1, is_disambiguation=True), "B": _page("B", 0, missing=True),
             "C": _page("C", 3, text="Too short.")}
    res = _run(db, FakeWiki(["A", "B", "C"], pages))
    assert res.added == [] and len(res.skipped) == 3


def test_wikipedia_unreachable_is_reported_not_raised(db):
    res = _run(db, FakeWiki([], {}, fail_search=True))
    assert res.error == "wikipedia_unavailable" and res.added == [] and db["commits"] == 0


def test_a_failing_store_is_rolled_back_and_others_continue(db, monkeypatch):
    real = live.store_article
    calls = {"n": 0}

    def flaky(conn, doc, embed_fn, count_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return real(conn, doc, embed_fn, count_tokens)

    monkeypatch.setattr(live, "store_article", flaky)
    pages = {"A": _page("A", 1), "B": _page("B", 2)}
    res = _run(db, FakeWiki(["A", "B"], pages))
    assert [a["title"] for a in res.added] == ["B"]
    assert res.skipped == [{"title": "A", "reason": "store failed"}]


def test_only_one_ingestion_at_a_time(db):
    assert live._lock.acquire(blocking=False)
    try:
        assert _run(db, FakeWiki(["A"], {"A": _page("A", 1)})).error == "busy"
    finally:
        live._lock.release()


def test_search_titles_uses_the_mediawiki_search_api(monkeypatch):
    seen = {}

    def handler(req):
        seen.update(dict(req.url.params))
        return httpx.Response(200, json={"query": {"search": [{"title": "Transmon"},
                                                              {"title": "Qubit"}]}})

    monkeypatch.setattr("app.ingest.wiki.time.sleep", lambda s: None)
    c = WikiClient(delay=0, transport=httpx.MockTransport(handler), backend="httpx")
    assert c.search_titles("what is a transmon", limit=3) == ["Transmon", "Qubit"]
    assert seen["list"] == "search" and seen["srnamespace"] == "0" and seen["srlimit"] == "3"


def test_ai_mode_ingests_only_when_the_evidence_is_weak(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import stream as stream_api
    from app.main import app

    calls = []
    monkeypatch.setattr(settings, "live_ingest_enabled", True)

    def fake_check(q, domain, t0):
        yield "results", {"results": [], "total": 0}
        return "agentic RAG" in q  # weak evidence for this one

    def fake_ingest(q):
        calls.append(q)
        yield "status", {"stage": "wiki_done", "added": [{"title": "Agentic RAG", "url": "u",
                                                          "chunks": 3}], "error": None}

    def fake_fast(q, mode, domain, cancel):
        yield "done", {"answer": "ok [1].", "citations": []}

    monkeypatch.setattr(stream_api, "_evidence_check", fake_check)
    monkeypatch.setattr(stream_api, "_wiki_ingest", fake_ingest)
    real_produce = stream_api._produce
    monkeypatch.setattr(stream_api, "_produce", lambda q, m, d, c: fake_fast(q, m, d, c)
                        if m == "overview" else real_produce(q, m, d, c))
    c = TestClient(app)
    for q, expected in (("what is agentic RAG", 1), ("what is a qubit", 1)):
        text = c.get("/api/search/stream", params={"q": q, "mode": "ai"}).text
        assert len(calls) == expected and "event: done" in text
    assert "wiki_done" in c.get("/api/search/stream",
                                params={"q": "what is agentic RAG", "mode": "ai"}).text


# -- real PostgreSQL + embedding + BM25 (copy of the database) ---------------------

def _copy_db():
    """Create a throw-away copy of the corpus database; None if not possible."""
    if shutil.which("su") is None or os.geteuid() != 0:
        return None
    name = "askito_livetest"
    cmd = (f'psql -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS {name}" '
           f'-c "CREATE DATABASE {name} TEMPLATE askito OWNER askito"')
    r = subprocess.run(["su", "postgres", "-c", cmd], capture_output=True, text=True)
    return name if r.returncode == 0 else None


def _redis_server():
    """A throwaway loopback redis-server for the cache checks (None if absent)."""
    exe = shutil.which("redis-server")
    if exe is None:
        return None, None
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([exe, "--port", str(port), "--bind", "127.0.0.1", "--save", "",
                             "--appendonly", "no"], stdout=subprocess.DEVNULL)
    for _ in range(50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    return proc, f"redis://127.0.0.1:{port}/0"


@pytest.mark.skipif(not os.environ.get("ASKITO_LIVE_DB_TEST"),
                    reason="set ASKITO_LIVE_DB_TEST=1 (needs a copyable local database)")
def test_ingested_article_is_found_by_dense_and_bm25(tmp_path):
    """After an ingestion the new chunks are found by BM25 and pgvector, and no
    cache serves a pre-ingestion result: retrieval L1 and Redis L2 miss, the
    answer cache key changes, the in-memory chunk store reloads."""
    name = _copy_db()
    if name is None:
        pytest.skip("cannot copy the database")
    redis_proc, redis_url = _redis_server()
    bm25 = tmp_path / "bm25"
    shutil.copytree(Path(__file__).resolve().parents[1] / "data" / "bm25", bm25)
    url = settings.database_url.rsplit("/", 1)[0] + f"/{name}"
    code = textwrap.dedent('''
        import json, sys
        from app.ingest import live
        from app.ingest.wiki import PageResult
        from app.retrieval import pipeline
        from app.retrieval.search import bm25_search, vector_search
        text = ("Zorblax quantum flux capacitors are a fictional device used only in this "
                "integration test. The Zorblax device stores imaginary qubits. ") * 40
        class Wiki:
            def search_titles(self, q, limit=5): return ["Zorblax device"]
            def fetch_page(self, t):
                return PageResult(requested_title=t, pageid=990000001, revid=1, title=t,
                                  url="https://en.wikipedia.org/wiki/Zorblax", extract=text)
            def close(self): pass
        from app.redis_cache import cache
        from app.retrieval import answer_cache
        from app.retrieval.store import store
        q = "zorblax flux capacitor"
        before = pipeline.retrieval_fingerprint("search")
        answer_key = answer_cache.make_key(q, "overview", "all")
        pipeline.run_retrieval(q, "search", domain="all")  # cached in L1 (+ L2)
        warm = pipeline.run_retrieval(q, "search", domain="all")
        pipeline.cache_clear()
        warm_l2 = pipeline.run_retrieval(q, "search", domain="all")
        store.ensure_loaded()
        n_before = len(store._chunks)
        res = live.ingest_for_query("zorblax", client=Wiki(), save_raw=False)
        bm = [c for c, _ in bm25_search(q, 5)]
        vec = [c for c, _ in vector_search(q, 5)]
        after = pipeline.run_retrieval(q, "search", domain="all")  # old L1 entry present
        pipeline.cache_clear()
        after_l2 = pipeline.run_retrieval(q, "search", domain="all")  # only Redis left
        store.ensure_loaded()
        again = live.ingest_for_query("zorblax", client=Wiki(), save_raw=False)
        print(json.dumps({"added": res.added, "bm25": bm, "vector": vec,
                          "again": again.skipped,
                          "version_changed": before != pipeline.retrieval_fingerprint("search"),
                          "answer_key_changed":
                              answer_key != answer_cache.make_key(q, "overview", "all"),
                          "redis": cache.enabled and cache.status()["available"],
                          "warm": [warm.cached, warm.cache_tier],
                          "warm_l2": [warm_l2.cached, warm_l2.cache_tier],
                          "after": [after.cached, [c for c, _ in after.final]],
                          "after_l2": [after_l2.cached, after_l2.cache_tier],
                          "store_grew": len(store._chunks) - n_before}))
    ''')
    env = {**os.environ, "DATABASE_URL": url, "ASKITO_BM25_DIR": str(bm25),
           "LLM_MODE": "mock", "WARMUP_ON_START": "false", "SEARCH_RERANK": "false",
           "REDIS_ENABLED": "true" if redis_url else "false",
           "REDIS_URL": redis_url or "redis://127.0.0.1:1/0"}
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                           env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        out = json.loads(r.stdout.strip().splitlines()[-1])
        assert out["added"] and out["added"][0]["chunks"] > 0
        import psycopg

        with psycopg.connect(url) as conn:
            new_ids = {row[0] for row in conn.execute(
                "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
                "WHERE d.pageid = 990000001 AND d.origin = 'dynamic'").fetchall()}
        assert new_ids & set(out["bm25"]) and new_ids & set(out["vector"])
        assert out["version_changed"] and out["answer_key_changed"]
        assert out["warm"] == [True, "l1"]
        # the result cached before the ingestion is not served afterwards
        assert out["after"][0] is False and new_ids & set(out["after"][1])
        assert out["store_grew"] == len(new_ids)
        if redis_url:
            assert out["redis"] and out["warm_l2"] == [True, "l2"]
            assert out["after_l2"] == [True, "l2"]  # the new, post-ingestion entry
        assert out["again"] == [{"title": "Zorblax device", "reason": "already in the corpus"}]
    finally:
        subprocess.run(["su", "postgres", "-c", f'psql -c "DROP DATABASE IF EXISTS {name}"'],
                       capture_output=True)
        if redis_proc:
            redis_proc.terminate()
            redis_proc.wait(timeout=10)
