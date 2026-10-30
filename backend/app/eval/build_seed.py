"""Build the hand-authored seed test set.

Reads eval/seed_questions.json, resolves gold doc/chunk ids from the DB, and
writes them as accepted entries into eval/testset.jsonl (and the draft + review
state so they appear as 'accepted' in the review UI). These are real,
human-authored questions that let the evaluation (M8) run in this environment;
the full >=150 set is produced with `make testset-draft` + review on a machine
with a local LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.eval import testset
from app.eval.gold import best_chunks, resolve_doc_ids

SEED_FILE = Path(__file__).resolve().parents[2] / "eval" / "seed_questions.json"


def run() -> dict:
    seeds = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    rows: list[dict] = []
    counts: dict[str, int] = {}
    warnings: list[str] = []
    for s in seeds:
        qt = s["qtype"]
        counts[qt] = counts.get(qt, 0) + 1
        qid = f"seed-{qt}-{counts[qt]:03d}"
        doc_ids = resolve_doc_ids(s.get("gold_titles", []))
        chunk_ids = best_chunks(doc_ids, s.get("keywords", ""), k=2) if doc_ids else []
        if qt != "unanswerable" and not doc_ids:
            warnings.append(f"{qid}: no gold docs for titles {s.get('gold_titles')}")
        row = {
            "id": qid,
            "question": s["question"],
            "reference": s.get("reference"),
            "gold_chunk_ids": chunk_ids,
            "gold_doc_ids": doc_ids,
            "qtype": qt,
            "domain": s["domain"],
        }
        errs = testset.validate(row)
        if errs:
            warnings.append(f"{qid}: {errs}")
            continue
        rows.append(row)

    # draft (with a source tag), review state (all accepted), and testset
    drafts = [{**r, "source": "seed"} for r in rows]
    testset.write_jsonl(testset.DRAFT, drafts)
    testset.write_json(testset.REVIEW_STATE, {r["id"]: "accepted" for r in rows})
    testset.write_jsonl(testset.TESTSET, rows)

    by_type = {}
    for r in rows:
        by_type[r["qtype"]] = by_type.get(r["qtype"], 0) + 1
    return {"written": len(rows), "by_type": by_type, "warnings": warnings}


def main() -> None:
    try:
        print(json.dumps(run(), indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(f"build-seed failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
