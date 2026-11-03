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
from app.retrieval import service

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
            doc_count = chunk_count = dynamic_count = 0
            snapshot = None
            reg = conn.execute(
                "SELECT to_regclass('public.documents'), to_regclass('public.chunks')"
            ).fetchone()
            if reg[0]:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                # the snapshot date ignores articles AI Mode added later
                cols = {r[0] for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'documents'").fetchall()}
                if "origin" in cols:
                    snapshot = conn.execute("SELECT MAX(fetched_at) FROM documents "
                                            "WHERE origin = 'snapshot'").fetchone()[0]
                    dynamic_count = conn.execute("SELECT COUNT(*) FROM documents "
                                                 "WHERE origin = 'dynamic'").fetchone()[0]
                else:
                    snapshot = conn.execute(
                        "SELECT MAX(fetched_at) FROM documents").fetchone()[0]
            if reg[1]:
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "status": "ok" if doc_count else "empty",
            "vector_extension": bool(has_vector),
            "migrated": bool(reg[0] and reg[1]),
            "documents": doc_count,
            "dynamic_documents": dynamic_count,
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


def _model_status(step: str) -> str:
    from app.warmup import state

    if not _lib_installed("sentence_transformers"):
        return "not_installed"
    if step in state["steps"]:
        return "ok"
    if state["status"] == "error":
        return "error"
    return "loading" if state["status"] == "warming" else "not_loaded"


def _check_embedding() -> dict:
    return {
        "status": _model_status("embedding"),
        "model": settings.embedding_model,
        "dim": settings.embedding_dim,
    }


def _check_reranker() -> dict:
    from app.retrieval import reranker

    st = reranker.status()
    status = "error" if st["error"] else _model_status("reranker")
    return {"status": status, "model": settings.reranker_model, "error": st["error"]}


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
    from app.llm.client import load_status

    client = get_llm_client()
    ok, msg = client.ping()
    return {
        "status": "ok" if ok else "unavailable",
        "mode": settings.llm_mode,
        "model": settings.model_for("answer"),
        "detail": msg,
        "loads": load_status(),  # last background model load: ok / loading / error code
    }


def _served_model() -> dict | None:
    """The answer model's tag, weight quantization and digest (None if unknown)."""
    if settings.llm_mode == "mock":
        return None
    from app.llm.identity import served_model

    return served_model("answer")


def _check_redis() -> dict:
    """Optional local L2 cache: "disabled", "ok" or "unavailable" (falls back)."""
    from app.redis_cache import cache

    st = cache.status()
    status = "disabled" if not st["enabled"] else ("ok" if st["available"] else "unavailable")
    return {"status": status, **st}


@router.get("/health")
def health() -> dict:
    components = {
        "db": _check_db(),
        "bm25": _check_bm25(),
        "embedding": _check_embedding(),
        "reranker": _check_reranker(),
        "ragas": _check_ragas(),
        "llm": _check_llm(),
        "redis": _check_redis(),
    }
    overall = "ok" if components["db"]["status"] in {"ok", "empty"} else "degraded"
    corpus = {
        "documents": components["db"].get("documents", 0),
        "dynamic_documents": components["db"].get("dynamic_documents", 0),
        "chunks": components["db"].get("chunks", 0),
        "snapshot_at": components["db"].get("snapshot_at"),
    }
    from app.warmup import state

    return {
        "status": overall, "components": components, "corpus": corpus,
        "llm_model": _served_model(),
        "warmup": {"status": state["status"], "steps": state["steps"],
                   "error": state["error"], "prewarm": state["prewarm"]},
        "default_mode": settings.default_mode,
        "evidence_cutoff": {
            "active_for_answers": service.evidence_cutoff_active("overview"),
            "min_score": settings.fast_min_evidence_score,
            "calibrated_for": settings.evidence_score_model,
            "reranker": settings.reranker_model,
        },
    }
