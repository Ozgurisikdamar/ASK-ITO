"""Test set schema, type distribution and file IO (spec §8.1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
TESTSET = EVAL_DIR / "testset.jsonl"
DRAFT = EVAL_DIR / "testset.draft.jsonl"
REVIEW_STATE = EVAL_DIR / "testset.review.json"

QTYPES = ["single_hop", "multi_hop", "comparison", "recent", "quantum", "unanswerable"]

# target distribution (fractions), spec §8.1
DISTRIBUTION = {
    "single_hop": 0.30,
    "multi_hop": 0.25,
    "comparison": 0.15,
    "recent": 0.15,
    "quantum": 0.10,
    "unanswerable": 0.05,
}

REQUIRED_FIELDS = {"id", "question", "qtype", "domain"}


def targets(total: int) -> dict[str, int]:
    return {qt: max(1, round(total * frac)) for qt, frac in DISTRIBUTION.items()}


def validate(q: dict) -> list[str]:
    errs = []
    if not REQUIRED_FIELDS <= set(q):
        errs.append(f"missing fields: {REQUIRED_FIELDS - set(q)}")
    if q.get("qtype") not in QTYPES:
        errs.append(f"bad qtype: {q.get('qtype')}")
    if q.get("qtype") == "unanswerable":
        if q.get("reference") not in (None, ""):
            errs.append("unanswerable must have null reference")
    else:
        if not q.get("gold_doc_ids"):
            errs.append("missing gold_doc_ids")
    return errs


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
