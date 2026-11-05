"""Evaluation jobs started from the evaluation page.

POST /api/eval/runs prepares the test set first (so a missing or unusable test
set is reported to the caller, not lost in a background thread), creates one
"queued" eval_runs row per configuration, and runs them one after another in a
background thread. Every row then shows its progress ("running", done/total),
"done", or "error" with the message. One job at a time per backend; a job cut
short by a backend restart is marked "error" at the next start.

Kept apart from app.eval.run so importing the API does not import RAGAS.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

import psycopg

from app.config import settings

log = logging.getLogger("askito.eval")
_lock = threading.Lock()


class EvalBusy(RuntimeError):
    pass


def active() -> bool:
    return _lock.locked()


def _ensure_model(modes: list[str], answers: bool) -> None:
    """Load the local model once before the first question (a cold Q8_0 load can
    take minutes; questions would otherwise time out one by one)."""
    if not answers and "hybrid_rerank_rewrite" not in modes:
        return
    from app.llm.client import get_llm_client

    client = get_llm_client()
    if not hasattr(client, "ensure_loaded"):
        return
    roles = {"answer"} if answers else set()
    if "agentic" in modes:
        roles.add("agent")
    if "hybrid_rerank_rewrite" in modes:
        roles.add("rewrite")
    for model in {settings.model_for(r) for r in roles}:
        client.ensure_loaded(model)


def start(configs: list[str], *, answers: bool, ragas: bool,
          subset: str | None = None) -> dict:
    """Validate, prepare the questions, queue the runs and start the thread.
    `subset` (a name in eval/subsets.json) runs only that stratified slice.
    Raises ValueError (bad request), EvalBusy, or RuntimeError (no questions)."""
    from app.eval import run as eval_run

    modes = eval_run.ALL_MODES if configs == ["all"] else list(dict.fromkeys(configs))
    if not modes:
        raise ValueError("no configurations selected")
    eval_run.check_modes(modes, answers)
    if not _lock.acquire(blocking=False):
        raise EvalBusy("an evaluation is already running")
    try:
        conn = psycopg.connect(settings.database_url)
        try:
            questions, prep = eval_run.load_questions(conn, subset)
            run_ids = [eval_run.create_run(conn, m, answers=answers, ragas=ragas,
                                           status="queued", source="api") for m in modes]
        except BaseException:
            conn.close()
            raise
        threading.Thread(target=_work, name="eval-job", daemon=True,
                         args=(conn, modes, run_ids, questions, answers, ragas)).start()
    except BaseException:
        _lock.release()
        raise
    return {"configs": modes, "run_ids": run_ids, "questions": len(questions),
            "subset": subset, "skipped": prep["skipped"]}


def _work(conn, modes, run_ids, questions, answers, ragas) -> None:
    """Run the queued configurations. The results stay in the database for the
    page; docs/EVAL_REPORT.md is written only by `make eval` / ./askito eval."""
    from app.eval import run as eval_run

    try:
        try:
            _ensure_model(modes, answers)
        except Exception as exc:  # noqa: BLE001  every queued run reports it
            msg = eval_run.error_text(exc)
            for rid in run_ids:
                eval_run.mark_error(conn, rid, msg)
            log.warning("eval job: model not ready: %s", exc.__class__.__name__)
            return
        for mode, rid in zip(modes, run_ids):
            try:
                eval_run.run_config(conn, mode, questions, answers=answers, ragas=ragas,
                                    run_id=rid)
            except Exception:  # noqa: BLE001  stored on the row; go on with the next
                log.exception("eval run %s (%s) failed", rid, mode)
    finally:
        conn.close()
        _lock.release()


def mark_interrupted() -> int:
    """At backend start: runs this API started that never finished were cut
    short by a restart; say so instead of showing them as running forever."""
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            n = conn.execute(
                "UPDATE eval_runs SET status='error', finished_at=%s, "
                "summary = COALESCE(summary, '{}'::jsonb) || %s::jsonb "
                "WHERE status IN ('queued','running') AND config_json->>'source' = 'api'",
                (datetime.now(timezone.utc),
                 json.dumps({"error": "interrupted: the backend restarted during the run"})),
            ).rowcount
            conn.commit()
            return n
    except Exception as exc:  # noqa: BLE001  no database yet: nothing to clean
        log.info("eval cleanup skipped: %s", exc.__class__.__name__)
        return 0
