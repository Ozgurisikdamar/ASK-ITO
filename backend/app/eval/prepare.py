"""Evaluation questions for the corpus that is loaded now.

eval/testset.jsonl stores gold document and chunk ids, and ids depend on the
order the articles were ingested: on a re-ingested or rebuilt corpus the
committed ids point at other articles (or at nothing). So before every run the
hand-authored seed questions get their gold re-resolved by article title
(eval/seed_questions.json: gold_titles + keywords), the same way
`make testset-seed` builds them, without rewriting the committed files. On the
corpus the test set was built from, this reproduces the committed ids exactly.

Questions that do not come from the seed file keep their ids, limited to ids
that still exist. An answerable question whose gold articles are not in this
corpus is left out of the run and reported, so it cannot count as a miss.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.eval import testset
from app.eval.gold import best_chunks, resolve_doc_ids

SEED_FILE = Path(__file__).resolve().parents[2] / "eval" / "seed_questions.json"


def seed_ids(seeds: list[dict]) -> dict[str, dict]:
    """Test-set id -> seed entry (seed-<qtype>-<n>, numbered per type in file order)."""
    counts: dict[str, int] = {}
    out: dict[str, dict] = {}
    for s in seeds:
        qt = s["qtype"]
        counts[qt] = counts.get(qt, 0) + 1
        out[f"seed-{qt}-{counts[qt]:03d}"] = s
    return out


def _existing(conn, table: str, ids: list[int] | None) -> list[int]:
    if not ids:
        return []
    found = {int(r[0]) for r in conn.execute(
        f"SELECT id FROM {table} WHERE id = ANY(%s)", (list(ids),)).fetchall()}  # noqa: S608
    return [i for i in ids if i in found]


def resolve(rows: list[dict], seeds: list[dict], conn) -> tuple[list[dict], list[dict]]:
    """-> (questions with gold for this corpus, skipped [{id, reason}])."""
    by_id = seed_ids(seeds)
    usable: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        errs = testset.validate(row)
        if errs:
            skipped.append({"id": row.get("id"), "reason": "; ".join(errs)})
            continue
        q = dict(row)
        seed = by_id.get(q["id"])
        if seed is not None:
            doc_ids = resolve_doc_ids(seed.get("gold_titles", []))
            q["gold_doc_ids"] = doc_ids
            q["gold_chunk_ids"] = (best_chunks(doc_ids, seed.get("keywords", ""), k=2)
                                   if doc_ids else [])
        else:
            q["gold_doc_ids"] = _existing(conn, "documents", q.get("gold_doc_ids"))
            q["gold_chunk_ids"] = _existing(conn, "chunks", q.get("gold_chunk_ids"))
        if q["qtype"] != "unanswerable" and not q["gold_doc_ids"]:
            skipped.append({"id": q["id"], "reason": "gold article not in this corpus"})
            continue
        usable.append(q)
    return usable, skipped


def upsert(conn, questions: list[dict]) -> None:
    for q in questions:
        conn.execute(
            """
            INSERT INTO eval_questions
              (id, question, reference, gold_chunk_ids, gold_doc_ids, qtype, domain)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO UPDATE SET
              question=EXCLUDED.question, reference=EXCLUDED.reference,
              gold_chunk_ids=EXCLUDED.gold_chunk_ids, gold_doc_ids=EXCLUDED.gold_doc_ids,
              qtype=EXCLUDED.qtype, domain=EXCLUDED.domain
            """,
            (q["id"], q["question"], q.get("reference"), q.get("gold_chunk_ids") or [],
             q.get("gold_doc_ids") or [], q["qtype"], q["domain"]),
        )


def prepare(conn) -> dict:
    """Resolve the test set against this corpus and store it in eval_questions.
    Returns {"questions": [...], "skipped": [...], "total": n}."""
    rows = testset.read_jsonl(testset.TESTSET)
    seeds = json.loads(SEED_FILE.read_text(encoding="utf-8")) if SEED_FILE.exists() else []
    questions, skipped = resolve(rows, seeds, conn)
    upsert(conn, questions)
    conn.commit()
    return {"questions": questions, "skipped": skipped, "total": len(rows)}
