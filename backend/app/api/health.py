"""Health endpoint.

Reports the status of every pipeline dependency. In M1 most components are not
yet built (index, models), so each reports an honest status rather than being
assumed present:
  * "ok"            working / reachable
  * "empty"         working but no data yet (e.g. DB with no documents)
  * "absent"        not built yet (e.g. BM25 index files missing)
  * "not_installed" optional heavy library not installed in this environment
  * "unavailable"   configured but cannot be reached (e.g. LLM endpoint down)

Loading heavy models is NOT triggered here; health reports configuration and
import availability so it stays fast. The mandatory reranker is reported
explicitly and is never silently skipped.
"""
from __future__ import annotations

import importlib.util
from importlib import metadata
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.llm.client import get_llm_client

router = APIRouter()

BM25_DIR = Path(__file__).resolve().parents[2] / "data" / "bm25"


def _lib_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_db() -> dict:
    try:
        from app.db.connection import get_connection

        with get_connection() as conn:
            has_vector = conn.execute(
                "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0]
            doc_count = chunk_count = 0
            snapshot = None
            reg = conn.execute(
                "SELECT to_regclass('public.documents'), to_regclass('public.chunks')"
            ).fetchone()
            if reg[0]:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                snapshot = conn.execute(
                    "SELECT MAX(fetched_at) FROM documents"
                ).fetchone()[0]
            if reg[1]:
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "status": "ok" if doc_count else "empty",
            "vector_extension": bool(has_vector),
            "migrated": bool(reg[0] and reg[1]),
            "documents": doc_count,
            "chunks": chunk_count,
            "snapshot_at": snapshot.isoformat() if snapshot else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "detail": exc.__class__.__name__}


def _check_bm25() -> dict:
    exists = BM25_DIR.exists() and any(BM25_DIR.glob("*")) and any(
        p.name != ".gitkeep" for p in BM25_DIR.glob("*")
    )
    return {"status": "ok" if exists else "absent", "path": str(BM25_DIR)}


def _check_embedding() -> dict:
    installed = _lib_installed("sentence_transformers")
    return {
        "status": "ok" if installed else "not_installed",
        "model": settings.embedding_model,
        "dim": settings.embedding_dim,
    }


def _check_reranker() -> dict:
    installed = _lib_installed("sentence_transformers")
    return {
        "status": "ok" if installed else "not_installed",
        "model": settings.reranker_model,
    }


def _check_ragas() -> dict:
    installed = _lib_installed("ragas")
    detail = {"status": "ok" if installed else "not_installed"}
    if installed:
        try:
            detail["version"] = metadata.version("ragas")
        except metadata.PackageNotFoundError:
            pass
    return detail


def _check_llm() -> dict:
    client = get_llm_client()
    ok, msg = client.ping()
    return {
        "status": "ok" if ok else "unavailable",
        "mode": settings.llm_mode,
        "detail": msg,
    }


@router.get("/health")
def health() -> dict:
    components = {
        "db": _check_db(),
        "bm25": _check_bm25(),
        "embedding": _check_embedding(),
        "reranker": _check_reranker(),
        "ragas": _check_ragas(),
        "llm": _check_llm(),
    }
    overall = "ok" if components["db"]["status"] in {"ok", "empty"} else "degraded"
    corpus = {
        "documents": components["db"].get("documents", 0),
        "chunks": components["db"].get("chunks", 0),
        "snapshot_at": components["db"].get("snapshot_at"),
    }
    return {"status": overall, "components": components, "corpus": corpus}
