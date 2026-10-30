"""Load the reviewed test set (eval/testset.jsonl) into eval_questions."""
from __future__ import annotations

import json
import sys

import psycopg

from app.config import settings
from app.eval import testset


def run() -> dict:
    rows = testset.read_jsonl(testset.TESTSET)
    valid, errors = [], []
    for q in rows:
        errs = testset.validate(q)
        if errs:
            errors.append({"id": q.get("id"), "errors": errs})
        else:
            valid.append(q)
    with psycopg.connect(settings.database_url) as conn:
        for q in valid:
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
                (q["id"], q["question"], q.get("reference"),
                 q.get("gold_chunk_ids") or [], q.get("gold_doc_ids") or [],
                 q["qtype"], q["domain"]),
            )
        conn.commit()
    return {"loaded": len(valid), "invalid": len(errors), "errors": errors[:10]}


def main() -> None:
    try:
        print(json.dumps(run(), indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"load-testset failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
