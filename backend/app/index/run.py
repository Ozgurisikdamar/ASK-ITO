"""Indexer (spec M3): chunk documents, embed, write pgvector, build BM25.

Reads the frozen snapshot in data/raw/, rebuilds the chunks table and the BM25
index. Re-running is a full rebuild (idempotent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg

from app.config import settings
from app.index import bm25_index
from app.index.chunk import chunk_document

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _load_docs() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in RAW_DIR.glob("*.json")
        if p.name != "_index.json"
    ]


def run() -> dict:
    from app.index import embed  # lazy: pulls torch only when indexing

    docs = _load_docs()
    if not docs:
        raise RuntimeError("no documents in data/raw — run `make ingest` first")

    with psycopg.connect(settings.database_url) as conn:
        pageid2id = {
            int(r[1]): int(r[0])
            for r in conn.execute("SELECT id, pageid FROM documents").fetchall()
        }

    count_tokens = embed.token_counter()
    items: list[tuple[int, object]] = []  # (doc_id, Chunk)
    for d in docs:
        doc_id = pageid2id.get(d["pageid"])
        if doc_id is None:
            continue
        for ch in chunk_document(
            d["title"], d["domain"], d["sections"], count_tokens,
            target=settings.chunk_target_tokens, overlap=settings.chunk_overlap_tokens,
        ):
            items.append((doc_id, ch))

    texts = [ch.text for _, ch in items]
    print(f"chunking: {len(docs)} docs -> {len(texts)} chunks; embedding ...", flush=True)
    embeddings = embed.embed_passages(texts)

    from app.retrieval.vectorstore import to_pgvector

    chunk_ids: list[int] = []
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("TRUNCATE chunks RESTART IDENTITY")
        with conn.cursor() as cur:
            for (doc_id, ch), emb in zip(items, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, section, ord, text, token_count, domain, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::vector) RETURNING id
                    """,
                    (doc_id, ch.section or None, ch.ord, ch.text, ch.token_count,
                     ch.domain, to_pgvector(emb)),
                )
                chunk_ids.append(int(cur.fetchone()[0]))
        conn.commit()

    print(f"building BM25 over {len(texts)} chunks ...", flush=True)
    bm25_index.build(chunk_ids, texts)

    return {
        "documents": len(docs),
        "chunks": len(chunk_ids),
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "bm25": {"k1": settings.bm25_k1, "b": settings.bm25_b},
    }


def main() -> None:
    try:
        summary = run()
    except Exception as exc:  # noqa: BLE001
        print(f"Indexing failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
