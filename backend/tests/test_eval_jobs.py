"""Evaluation page: test set resolved against the loaded corpus, runs that are
queued at once and report progress or their error, retrieval-only runs.

Regression: on a fresh setup eval_questions was empty; POST /api/eval/runs
answered "started", the background task raised, and the page showed nothing."""
from __future__ import annotations

import json

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.eval import jobs, prepare
from app.eval import run as eval_run
from app.main import app

SEEDS = [
    {"qtype": "single_hop", "question": "q1", "gold_titles": ["Alpha"], "keywords": "a"},
    {"qtype": "single_hop", "question": "q2", "gold_titles": ["Missing"], "keywords": "b"},
    {"qtype": "unanswerable", "question": "q3", "gold_titles": [], "keywords": ""},
]


def test_seed_ids_follow_the_build_seed_numbering():
    assert list(prepare.seed_ids(SEEDS)) == [
        "seed-single_hop-001", "seed-single_hop-002", "seed-unanswerable-001"]


class _IdConn:
    """Only answers `SELECT id FROM <table> WHERE id = ANY(...)`."""

    def __init__(self, docs, chunks):
        self.tables = {"documents": set(docs), "chunks": set(chunks)}

    def execute(self, sql, params=()):
        table = "documents" if "FROM documents" in sql else "chunks"
        rows = [(i,) for i in params[0] if i in self.tables[table]]
        return type("R", (), {"fetchall": lambda _self: rows})()


def test_resolve_regolds_seed_questions_and_drops_unusable_ones(monkeypatch):
    titles = {"Alpha": 7}
    monkeypatch.setattr(prepare, "resolve_doc_ids",
                        lambda ts: [titles[t] for t in ts if t in titles])
    monkeypatch.setattr(prepare, "best_chunks", lambda docs, kw, k=2: [70, 71])
    rows = [
        # committed ids are stale (another corpus): re-resolved by title
        {"id": "seed-single_hop-001", "question": "q1 edited", "reference": "r",
         "gold_doc_ids": [1], "gold_chunk_ids": [10], "qtype": "single_hop", "domain": "ai"},
        # its article is not in this corpus: left out, reported
        {"id": "seed-single_hop-002", "question": "q2", "reference": "r",
         "gold_doc_ids": [2], "gold_chunk_ids": [20], "qtype": "single_hop", "domain": "ai"},
        {"id": "seed-unanswerable-001", "question": "q3", "reference": None,
         "gold_doc_ids": [], "gold_chunk_ids": [], "qtype": "unanswerable", "domain": "ai"},
        # not a seed question: keeps only the ids that still exist
        {"id": "draft-1", "question": "q4", "reference": "r", "gold_doc_ids": [5, 6],
         "gold_chunk_ids": [50, 60], "qtype": "multi_hop", "domain": "ai"},
        {"id": "bad", "question": "q5", "qtype": "nonsense", "domain": "ai"},
    ]
    usable, skipped = prepare.resolve(rows, SEEDS, _IdConn(docs=[5], chunks=[50]))
    by_id = {q["id"]: q for q in usable}
    assert set(by_id) == {"seed-single_hop-001", "seed-unanswerable-001", "draft-1"}
    first = by_id["seed-single_hop-001"]
    assert first["gold_doc_ids"] == [7] and first["gold_chunk_ids"] == [70, 71]
    assert first["question"] == "q1 edited"  # the reviewed text is kept
    assert by_id["draft-1"]["gold_doc_ids"] == [5] and by_id["draft-1"]["gold_chunk_ids"] == [50]
    reasons = {s["id"]: s["reason"] for s in skipped}
    assert reasons["seed-single_hop-002"] == "gold article not in this corpus"
    assert "bad qtype" in reasons["bad"]


# -- API -------------------------------------------------------------------------

def _post(body):
    return TestClient(app).post("/api/eval/runs", json=body)


def test_no_usable_questions_is_reported_not_swallowed(monkeypatch):
    def none(conn):
        raise eval_run.NoQuestions("No evaluation questions: eval/testset.jsonl is empty.")

    monkeypatch.setattr(eval_run, "load_questions", none)
    monkeypatch.setattr(jobs.psycopg, "connect", lambda *a, **k: _NullConn())
    r = _post({"configs": ["bm25"], "answers": False})
    assert r.status_code == 409 and r.json()["error"] == "no_questions"
    assert "testset.jsonl" in r.json()["message"]
    assert not jobs.active()  # the job lock is released again


def test_answer_only_configs_need_answers():
    r = _post({"configs": ["closed_book", "bm25"], "answers": False})
    assert r.status_code == 400 and "closed_book" in r.json()["message"]
    r = _post({"configs": ["nope"]})
    assert r.status_code == 400


def test_one_job_at_a_time():
    assert jobs._lock.acquire(blocking=False)
    try:
        r = _post({"configs": ["bm25"], "answers": False})
        assert r.status_code == 409 and r.json()["error"] == "busy"
    finally:
        jobs._lock.release()


class _NullConn:
    def close(self):
        pass


# -- run_config: progress, done and error rows ------------------------------------

class _EvalConn:
    """Enough of a connection for run_config / _summarize."""

    def __init__(self):
        self.runs: dict[int, dict] = {}
        self.results: dict[tuple, tuple] = {}
        self.updates: list[dict] = []

    def commit(self):
        pass

    def rollback(self):
        pass

    def execute(self, sql, params=()):
        rows: list = []
        if sql.startswith("INSERT INTO eval_runs"):
            rid = len(self.runs) + 1
            self.runs[rid] = {"status": params[3], "config": json.loads(params[1])}
            rows = [(rid,)]
        elif sql.startswith("UPDATE eval_runs"):
            rid = params[-1]
            entry = {"sql": sql, "params": params}
            self.updates.append(entry)
            if "status='error'" in sql:
                self.runs[rid].update(status="error", summary=json.loads(params[1]))
            elif "status='done'" in sql:
                self.runs[rid].update(status="done", summary=json.loads(params[1]))
            elif "status='running'" in sql:
                self.runs[rid]["status"] = "running"
            else:
                self.runs[rid]["summary"] = json.loads(params[0])
        elif "FROM chunks WHERE id = ANY" in sql:
            rows = [(c, c // 10) for c in params[0]]
        elif sql.lstrip().startswith("INSERT INTO eval_results"):
            self.results[(params[0], params[1])] = params
        elif "FROM eval_results" in sql:
            rows = [(p[1], p[2], p[3], json.loads(p[4]), p[6], p[7])
                    for (rid, _), p in self.results.items() if rid == params[0]]
        return type("R", (), {"fetchone": lambda _s: rows[0] if rows else None,
                              "fetchall": lambda _s: rows,
                              "rowcount": len(rows)})()


QUESTIONS = [
    {"id": "a", "question": "qa", "reference": "r", "gold_chunk_ids": {10}, "gold_doc_ids": {1},
     "qtype": "single_hop", "domain": "ai"},
    {"id": "b", "question": "qb", "reference": "r", "gold_chunk_ids": {20}, "gold_doc_ids": {2},
     "qtype": "single_hop", "domain": "ai"},
]


def test_retrieval_only_run_scores_passages_without_answers(monkeypatch):
    seen = []

    def fake_retrieve(mode, q):
        seen.append((mode, q["id"]))
        return {"answer": "", "ranked_chunk_ids": [10, 30], "trace": [], "llm_calls": 0,
                "metrics": {"llm": {"calls": 0}, "retrieval_ms": 3}}

    monkeypatch.setattr(eval_run, "_retrieve_for", fake_retrieve)
    monkeypatch.setattr(eval_run, "_answer_for",
                        lambda m, q: pytest.fail("no answers in a retrieval-only run"))
    monkeypatch.setattr(eval_run.ragas_eval, "compute",
                        lambda s: pytest.fail("no RAGAS without answers"))
    conn = _EvalConn()
    rid = eval_run.run_config(conn, "bm25", QUESTIONS, answers=False, ragas=True)
    run = conn.runs[rid]
    assert run["status"] == "done" and run["config"]["answers"] is False
    assert run["config"]["ragas"] is False  # RAGAS needs answers
    assert run["summary"]["progress"] == {"done": 2, "total": 2}
    assert run["summary"]["retrieval"]["doc_recall@5"] == 0.5  # hit for a, miss for b
    assert run["summary"]["ragas"]["faithfulness"] is None
    assert seen == [("bm25", "a"), ("bm25", "b")]
    # progress is written after every question
    assert [json.loads(u["params"][0])["progress"]["done"] for u in conn.updates
            if u["sql"].startswith("UPDATE eval_runs SET summary")] == [1, 2]


def test_a_failing_run_is_stored_with_its_message(monkeypatch):
    from app.llm.client import LLMError

    calls = []

    def fail_second(mode, q):
        calls.append(q["id"])
        if q["id"] == "b":
            raise LLMError("model requires more system memory", code="model_memory")
        return {"answer": "x", "ranked_chunk_ids": [10], "trace": [], "llm_calls": 1,
                "metrics": {"llm": {"calls": 1}}}

    monkeypatch.setattr(eval_run, "_answer_for", fail_second)
    conn = _EvalConn()
    rid = eval_run.create_run(conn, "hybrid", status="queued", source="api")
    with pytest.raises(LLMError):
        eval_run.run_config(conn, "hybrid", QUESTIONS, answers=True, ragas=False, run_id=rid)
    run = conn.runs[rid]
    assert run["status"] == "error"
    assert "not enough memory" in run["summary"]["error"]  # the actionable message
    assert run["summary"]["progress"] == {"done": 1, "total": 2}


def test_job_runs_every_config_and_keeps_going_after_a_failure(monkeypatch):
    ran, errors = [], []

    def fake_run_config(conn, mode, questions, *, answers, ragas, run_id):
        ran.append(mode)
        if mode == "vector":
            errors.append(run_id)
            raise RuntimeError("boom")
        return run_id

    monkeypatch.setattr(eval_run, "run_config", fake_run_config)
    # the tracked docs/EVAL_REPORT.md is left alone (the CLI writes it)
    monkeypatch.setattr("app.eval.report.generate", lambda ids: pytest.fail("no report"))
    assert jobs._lock.acquire(blocking=False)  # start() holds it for the thread
    jobs._work(_NullConn(), ["bm25", "vector", "hybrid"], [1, 2, 3], QUESTIONS, False, False)
    assert ran == ["bm25", "vector", "hybrid"]
    assert errors == [2] and not jobs.active()


def test_job_reports_a_model_that_cannot_load_on_every_queued_run(monkeypatch):
    from app.llm.client import LLMError

    marked = []

    def no_model(modes, answers):
        raise LLMError("pull it first", code="model_not_found")

    monkeypatch.setattr(jobs, "_ensure_model", no_model)
    monkeypatch.setattr(eval_run, "mark_error", lambda c, rid, msg, p=None: marked.append((rid, msg)))
    monkeypatch.setattr(eval_run, "run_config", lambda *a, **k: pytest.fail("must not run"))
    assert jobs._lock.acquire(blocking=False)
    jobs._work(_NullConn(), ["hybrid", "agentic"], [4, 5], QUESTIONS, True, False)
    assert [rid for rid, _ in marked] == [4, 5]
    assert all("./askito model" in msg for _, msg in marked)
    assert not jobs.active()


def test_runs_cut_short_by_a_restart_are_marked(monkeypatch):
    seen = {}

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            seen["sql"], seen["params"] = sql, params
            return type("R", (), {"rowcount": 2})()

        def commit(self):
            pass

    monkeypatch.setattr(jobs.psycopg, "connect", lambda *a, **k: Conn())
    assert jobs.mark_interrupted() == 2
    assert "config_json->>'source' = 'api'" in seen["sql"]  # CLI runs are left alone
    assert "interrupted" in seen["params"][1]


def test_runs_list_exposes_status_progress_and_options():
    try:
        r = TestClient(app).get("/api/eval/runs")
    except psycopg.Error:
        pytest.skip("database not available")
    assert r.status_code == 200 and "active" in r.json()
    for run in r.json()["runs"][:3]:
        assert {"progress", "error", "options"} <= set(run)


# -- real database: the committed test set matches the corpus it was built on -------

def _db_has_corpus() -> bool:
    try:
        with psycopg.connect(settings.database_url) as c:
            return c.execute("SELECT count(*) FROM documents").fetchone()[0] > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _db_has_corpus(), reason="corpus database not available")
def test_resolution_reproduces_the_committed_gold_on_the_original_corpus():
    from app.eval import testset

    rows = testset.read_jsonl(testset.TESTSET)
    seeds = json.loads(prepare.SEED_FILE.read_text(encoding="utf-8"))
    with psycopg.connect(settings.database_url) as conn:
        usable, skipped = prepare.resolve(rows, seeds, conn)
    committed = {r["id"]: r for r in rows}
    assert not skipped and len(usable) == len(rows)
    for q in usable:
        assert q["gold_doc_ids"] == committed[q["id"]]["gold_doc_ids"]
        assert q["gold_chunk_ids"] == committed[q["id"]]["gold_chunk_ids"]
