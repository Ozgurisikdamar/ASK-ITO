"""Load the reviewed test set (eval/testset.jsonl) into eval_questions, with the
gold ids resolved against the corpus that is loaded now (see app.eval.prepare)."""
from __future__ import annotations

import json
import sys

import psycopg

from app.config import settings
from app.eval import prepare


def run() -> dict:
    with psycopg.connect(settings.database_url) as conn:
        out = prepare.prepare(conn)
    return {"loaded": len(out["questions"]), "skipped": len(out["skipped"]),
            "skipped_detail": out["skipped"][:10]}


def main() -> None:
    try:
        print(json.dumps(run(), indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"load-testset failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
