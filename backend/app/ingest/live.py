"""Query-time Wikipedia ingestion for AI Mode (dynamic corpus expansion).

When the local corpus has no strong evidence for a question, AI Mode calls
`ingest_for_query`: search Wikipedia for the question, fetch a few matching
articles, clean and chunk them with the snapshot pipeline (same chunker, same
embedding model), store document + chunks + embeddings in PostgreSQL in one
transaction per article, then rebuild the BM25 index from the database. The
rebuild bumps the index version, so the chunk store, BM25 searcher and every
retrieval / answer cache key pick up the new chunks on the next request;
pgvector sees them as soon as they are committed.

Bounds: LIVE_INGEST_MAX_ARTICLES articles, LIVE_INGEST_MAX_CHARS characters per
article, LIVE_INGEST_TIMEOUT_S for the Wikipedia part, a short HTTP timeout and
few retries. Idempotent: an article already stored with the same revision is
skipped, a changed revision replaces its chunks. Wikipedia text is untrusted
evidence (the answer prompt treats every passage as data, never instructions).
Errors never leave a partial article behind: the transaction is rolled back and
the result reports what happened. Only one ingestion runs at a time per process.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import settings

log = logging.getLogger("askito.ingest.live")
_lock = threading.Lock()


@dataclass
class LiveResult:
    query: str
    searched: list[str] = field(default_factory=list)  # titles Wikipedia returned
    added: list[dict] = field(default_factory=list)  # {title, url, chunks}
    skipped: list[dict] = field(default_factory=list)  # {title, reason}
    error: str | None = None  # wikipedia_unavailable | busy | ...
    ms: int = 0

    def as_dict(self) -> dict:
        return {"searched": self.searched, "added": self.added, "skipped": self.skipped,
                "error": self.error, "ms": self.ms}


def content_hash(sections: list[dict]) -> str:
    raw = "\n\n".join(f"{s['title']}\n{s['text']}" for s in sections)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _default_client():
    from app.ingest.wiki import WikiClient

    return WikiClient(delay=settings.live_ingest_delay_s, max_retries=2, max_backoff=5.0,
                      timeout=settings.live_ingest_http_timeout_s)


def _trim(sections: list[dict], max_chars: int) -> list[dict]:
    out, used = [], 0
    for s in sections:
        if used >= max_chars:
            break
        text = s["text"][: max_chars - used]
        used += len(text)
        out.append({"title": s["title"], "text": text})
    return out


def _existing(conn, pageid: int) -> tuple[int, int, str] | None:
    row = conn.execute("SELECT id, revid, origin FROM documents WHERE pageid = %s",
                       (pageid,)).fetchone()
    return (int(row[0]), int(row[1]), row[2]) if row else None


def store_article(conn, doc: dict, embed_fn, count_tokens) -> int:
    """Upsert one article and replace its chunks (caller owns the transaction).
    Returns the number of chunks written."""
    from app.index.chunk import chunk_document
    from app.retrieval.vectorstore import to_pgvector

    chunks = chunk_document(doc["title"], doc["domain"], doc["sections"], count_tokens,
                            target=settings.chunk_target_tokens,
                            overlap=settings.chunk_overlap_tokens)
    if not chunks:
        return 0
    vectors = embed_fn([c.text for c in chunks])
    doc_id = conn.execute(
        """
        INSERT INTO documents (pageid, revid, title, url, domain, also_quantum, is_seed,
                               categories, fetched_at, origin, language, content_hash,
                               ingested_at)
        VALUES (%s,%s,%s,%s,%s,false,false,%s,%s,'dynamic','en',%s,now())
        ON CONFLICT (pageid) DO UPDATE SET
            revid = EXCLUDED.revid, title = EXCLUDED.title, url = EXCLUDED.url,
            categories = EXCLUDED.categories, fetched_at = EXCLUDED.fetched_at,
            content_hash = EXCLUDED.content_hash, ingested_at = now()
        RETURNING id
        """,
        (doc["pageid"], doc["revid"], doc["title"], doc["url"], doc["domain"],
         doc["categories"], doc["fetched_at"], doc["content_hash"]),
    ).fetchone()[0]
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_id, section, ord, text, token_count, domain, embedding) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s::vector)",
            [(doc_id, c.section or None, c.ord, c.text, c.token_count, c.domain,
              to_pgvector(v)) for c, v in zip(chunks, vectors)],
        )
    return len(chunks)


def rebuild_bm25(conn) -> int:
    """BM25 over every chunk in the database (no re-embedding); bumps the index
    version, which makes the in-memory store, the searcher and caches reload."""
    from app.index import bm25_index

    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    return bm25_index.build([int(r[0]) for r in rows], [r[1] for r in rows])


def ingest_for_query(query: str, *, client=None, connect=None, embed_fn=None,
                     count_tokens=None, save_raw=True, progress=None) -> LiveResult:
    """Fetch and store up to LIVE_INGEST_MAX_ARTICLES Wikipedia articles for
    `query`. `progress(stage, data)` is called with "wiki_search" / "wiki_fetch" /
    "wiki_store"; the other arguments exist for tests."""
    import psycopg

    from app.ingest.clean import char_count, clean_extract

    res = LiveResult(query=query)
    t0 = time.perf_counter()
    if not _lock.acquire(blocking=False):
        res.error = "busy"
        return res
    own_client = client is None
    client = client or _default_client()
    connect = connect or (lambda: psycopg.connect(settings.database_url))
    try:
        deadline = time.monotonic() + settings.live_ingest_timeout_s
        if progress:
            progress("wiki_search", {})
        try:
            titles = client.search_titles(query, limit=settings.live_ingest_search_limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("wikipedia search failed: %s", exc.__class__.__name__)
            res.error = "wikipedia_unavailable"
            return res
        res.searched = titles
        pages = []
        with connect() as conn:
            for title in titles:
                if len(pages) >= settings.live_ingest_max_articles:
                    break
                if time.monotonic() > deadline:
                    res.skipped.append({"title": title, "reason": "time budget"})
                    continue
                if progress:
                    progress("wiki_fetch", {"title": title})
                try:
                    page = client.fetch_page(title)
                except Exception as exc:  # noqa: BLE001
                    log.warning("wikipedia fetch failed for a title: %s",
                                exc.__class__.__name__)
                    res.skipped.append({"title": title, "reason": "fetch failed"})
                    continue
                if page.missing or page.is_disambiguation or not page.pageid:
                    res.skipped.append({"title": title, "reason": "missing or disambiguation"})
                    continue
                cleaned = clean_extract(page.extract or "")
                if char_count(cleaned) < settings.ingest_min_chars:
                    res.skipped.append({"title": page.title, "reason": "too short"})
                    continue
                sections = _trim([{"title": s.title, "text": s.text} for s in cleaned],
                                 settings.live_ingest_max_chars)
                old = _existing(conn, page.pageid)
                digest = content_hash(sections)
                # the curated snapshot is never rewritten; a dynamic article is
                # refreshed only when Wikipedia has a newer revision
                if old and (old[2] == "snapshot" or old[1] == page.revid):
                    res.skipped.append({"title": page.title, "reason": "already in the corpus"})
                    continue
                pages.append((page, sections, digest))
        if not pages:
            return res
        if embed_fn is None or count_tokens is None:
            from app.index import embed
            from app.ml import inference_lock

            def embed_fn(texts):  # share the CPU politely with live requests
                with inference_lock:
                    return embed.embed_passages(texts)

            count_tokens = count_tokens or embed.token_counter()
        stored = False
        for page, sections, digest in pages:
            doc = {"pageid": page.pageid, "revid": page.revid, "title": page.title,
                   "url": page.url, "domain": "general", "categories": page.categories,
                   "fetched_at": datetime.now(timezone.utc).isoformat(),
                   "content_hash": digest, "sections": sections, "origin": "dynamic",
                   "kind": "dynamic", "is_seed": False, "also_quantum": False,
                   "requested_title": page.requested_title,
                   "char_count": sum(len(s["text"]) for s in sections)}
            if progress:
                progress("wiki_store", {"title": page.title})
            try:
                with connect() as conn:  # one transaction per article
                    n = store_article(conn, doc, embed_fn, count_tokens)
                    conn.commit()
            except Exception as exc:  # noqa: BLE001  (rolled back by the context)
                log.warning("storing a wikipedia article failed: %s", exc.__class__.__name__)
                res.skipped.append({"title": page.title, "reason": "store failed"})
                continue
            if save_raw:  # a later full `make index` keeps the article
                _save_raw(doc)
            res.added.append({"title": page.title, "url": page.url, "chunks": n})
            stored = True
        if stored:
            with connect() as conn:
                rebuild_bm25(conn)
        return res
    finally:
        if own_client:
            client.close()
        res.ms = int((time.perf_counter() - t0) * 1000)
        _lock.release()
        log.info("live ingest: searched=%d added=%d skipped=%d error=%s ms=%d",
                 len(res.searched), len(res.added), len(res.skipped), res.error, res.ms)


def _save_raw(doc: dict) -> None:
    from app.ingest.run import RAW_DIR

    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"{doc['pageid']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as exc:  # the database copy is what matters
        log.warning("could not save raw article: %s", exc.__class__.__name__)
