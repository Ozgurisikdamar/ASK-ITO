"""Stratified, deterministic question subsets for expensive (LLM) benchmarks.

  python scripts/subsets.py build    # writes eval/subsets.json from eval_questions

Subsets are keyed by the question text (stable across DB reloads) and nested:
`lat12` (latency repetitions, RAGAS) is contained in `sub20`, which is contained
in the full set. Every question type is represented; selection inside a type
is evenly spaced over the id order (no randomness).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUBSETS_PATH = Path(__file__).resolve().parents[1] / "eval" / "subsets.json"
SIZES = {"lat12": 12, "sub20": 20}


def all_questions(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, question, reference, gold_chunk_ids, gold_doc_ids, qtype, domain "
        "FROM eval_questions ORDER BY id"
    ).fetchall()
    return [
        {"id": r[0], "question": r[1], "reference": r[2],
         "gold_chunk_ids": set(r[3] or []), "gold_doc_ids": set(r[4] or []),
         "qtype": r[5], "domain": r[6]}
        for r in rows
    ]


PROBES_PATH = Path(__file__).resolve().parents[1] / "eval" / "refusal_probes.json"


def load_probes() -> list[dict]:
    data = json.loads(PROBES_PATH.read_text())
    return [{"id": p["id"], "question": p["question"], "reference": None,
             "gold_chunk_ids": set(), "gold_doc_ids": set(), "qtype": "unanswerable",
             "domain": p.get("kind")} for p in data["questions"]]


def load_questions(conn, name: str | None) -> list[dict]:
    if name == "probes":
        return load_probes()
    qs = all_questions(conn)
    if not name or name == "all":
        return qs
    wanted = json.loads(SUBSETS_PATH.read_text())["subsets"][name]
    by_text = {q["question"]: q for q in qs}
    missing = [w for w in wanted if w not in by_text]
    if missing:
        raise RuntimeError(f"subset {name}: {len(missing)} questions not in eval_questions")
    return [by_text[w] for w in wanted]


def _pick(items: list, n: int) -> list:
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def build(conn) -> dict:
    qs = all_questions(conn)
    by_type: dict[str, list[dict]] = {}
    for q in qs:
        by_type.setdefault(q["qtype"], []).append(q)
    out: dict[str, list[str]] = {}
    chosen: list[dict] = []
    for name, size in sorted(SIZES.items(), key=lambda kv: kv[1]):
        # proportional allocation, at least one per type, keep previous subset
        alloc = {t: max(1, round(size * len(v) / len(qs))) for t, v in by_type.items()}
        while sum(alloc.values()) > size:
            t = max(alloc, key=lambda k: alloc[k])
            alloc[t] -= 1
        while sum(alloc.values()) < size:
            t = max(by_type, key=lambda k: len(by_type[k]) - alloc[k])
            alloc[t] += 1
        picked = list(chosen)
        for t, items in by_type.items():
            have = [q for q in picked if q["qtype"] == t]
            rest = [q for q in items if q not in have]
            picked += _pick(rest, max(0, alloc[t] - len(have)))
        picked.sort(key=lambda q: q["id"])
        out[name] = [q["question"] for q in picked]
        chosen = picked
    summary = {name: _counts([q for q in qs if q["question"] in set(v)]) for name, v in out.items()}
    return {"subsets": out, "composition": summary, "total_questions": len(qs)}


def _counts(qs: list[dict]) -> dict:
    c: dict[str, int] = {}
    for q in qs:
        c[q["qtype"]] = c.get(q["qtype"], 0) + 1
    return c


if __name__ == "__main__":
    import psycopg

    from app.config import settings

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        with psycopg.connect(settings.database_url) as conn:
            data = build(conn)
        SUBSETS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(data["composition"], indent=2))
    else:
        print(__doc__)
