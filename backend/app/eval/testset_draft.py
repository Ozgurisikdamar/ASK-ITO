"""Draft test-set generation (spec §8.1).

For each question type, sample source chunk(s) from the corpus, ask the LLM to
write a question + reference answer, and record the gold chunk/doc ids
automatically. Output goes to eval/testset.draft.jsonl for human review.

Requires a real LLM (local Ollama) to produce usable questions; with the mock
LLM it still runs but yields placeholder text. The number per type follows the
spec §8.1 distribution.
"""
from __future__ import annotations

import json
import random
import re
import sys

import psycopg

from app.agent.prompts import load
from app.config import settings
from app.eval import testset
from app.eval.llm_cache import CachedLLM

_JSON = re.compile(r"\{.*\}", re.DOTALL)

TASKS = {
    "single_hop": "Write one factual question answerable from the single passage.",
    "multi_hop": "Write one question that requires combining facts from BOTH passages.",
    "comparison": "Write one question comparing the two entities (by date, number, field or discovery).",
    "recent": "Write one detailed factual question about the 2025 or 2026 prize in this passage.",
    "quantum": "Write one factual question about this quantum-computing topic.",
}


def _rows(sql, params=()):
    with psycopg.connect(settings.database_url) as conn:
        return conn.execute(sql, params).fetchall()


def _chunk_text(cid: int) -> tuple[str, int]:
    r = _rows("SELECT text, doc_id FROM chunks WHERE id = %s", (cid,))
    return (r[0][0], int(r[0][1])) if r else ("", 0)


def _sample_single(domain_sql: str, params=()) -> list[int]:
    r = _rows(
        f"""SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE {domain_sql} AND c.token_count > 120
            ORDER BY random() LIMIT 1""",
        params,
    )
    return [int(r[0][0])] if r else []


def _sample_two_docs(domain: str) -> list[int]:
    r = _rows(
        """SELECT DISTINCT ON (doc_id) id FROM chunks
           WHERE domain = %s AND token_count > 120
           ORDER BY doc_id, random() LIMIT 40""",
        (domain,),
    )
    ids = [int(x[0]) for x in r]
    random.shuffle(ids)
    return ids[:2]


def _parse(raw: str) -> dict | None:
    m = _JSON.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _gen_answerable(llm, task: str, chunk_ids: list[int]) -> dict | None:
    blocks, doc_ids = [], []
    for i, cid in enumerate(chunk_ids, 1):
        text, doc_id = _chunk_text(cid)
        body = text.split("\n\n", 1)[-1][:700]
        blocks.append(f"[Passage {i}] {body}")
        doc_ids.append(doc_id)
    prompt = load("gen_answerable").replace("{task}", task).replace("{context}", "\n\n".join(blocks))
    obj = _parse(llm.chat([{"role": "user", "content": prompt}], temperature=0.3))
    if not obj or not obj.get("question"):
        return None
    return {
        "question": obj["question"].strip(),
        "reference": (obj.get("reference") or "").strip() or None,
        "gold_chunk_ids": chunk_ids,
        "gold_doc_ids": sorted(set(doc_ids)),
    }


def _gen_unanswerable(llm, domain: str) -> dict | None:
    prompt = load("gen_unanswerable").replace("{domain}", domain)
    obj = _parse(llm.chat([{"role": "user", "content": prompt}], temperature=0.5))
    if not obj or not obj.get("question"):
        return None
    return {
        "question": obj["question"].strip(), "reference": None,
        "gold_chunk_ids": [], "gold_doc_ids": [],
    }


def _select_and_generate(llm, qtype: str) -> dict | None:
    if qtype == "single_hop":
        cids = _sample_single("TRUE")
        return _gen_answerable(llm, TASKS[qtype], cids) if cids else None
    if qtype == "quantum":
        cids = _sample_single("c.domain = %s", ("quantum",))
        return _gen_answerable(llm, TASKS[qtype], cids) if cids else None
    if qtype == "recent":
        cids = _sample_single(
            "(d.title ILIKE %s OR d.title ILIKE %s OR c.text ILIKE %s OR c.text ILIKE %s)",
            ("%2025%", "%2026%", "%2025 Nobel%", "%2026 Nobel%"),
        )
        return _gen_answerable(llm, TASKS[qtype], cids) if cids else None
    if qtype in ("multi_hop", "comparison"):
        domain = random.choice(["nobel", "ai", "quantum"])
        cids = _sample_two_docs(domain)
        return _gen_answerable(llm, TASKS[qtype], cids) if len(cids) == 2 else None
    if qtype == "unanswerable":
        return _gen_unanswerable(llm, random.choice(["Nobel Prizes", "quantum computing", "AI"]))
    return None


def _domain_of(doc_ids: list[int]) -> str:
    if not doc_ids:
        return "nobel"
    r = _rows("SELECT domain FROM documents WHERE id = ANY(%s)", (doc_ids,))
    return r[0][0] if r else "nobel"


def run(total: int | None = None, seed: int = 0, llm=None) -> dict:
    random.seed(seed)
    total = total or settings.testset_target
    llm = CachedLLM(llm=llm, enabled=False) if llm is not None else CachedLLM()
    tgt = testset.targets(total)
    drafts: list[dict] = []
    counts = {qt: 0 for qt in testset.QTYPES}
    for qtype, want in tgt.items():
        attempts = 0
        while counts[qtype] < want and attempts < want * 4:
            attempts += 1
            item = _select_and_generate(llm, qtype)
            if not item:
                continue
            n = len(drafts) + 1
            item.update(
                id=f"{qtype}-{counts[qtype] + 1:03d}",
                qtype=qtype,
                domain=item.get("domain") or _domain_of(item.get("gold_doc_ids", [])),
                source="draft",
            )
            drafts.append(item)
            counts[qtype] += 1
    testset.write_jsonl(testset.DRAFT, drafts)
    return {"drafts": len(drafts), "by_type": counts, "targets": tgt, "file": str(testset.DRAFT)}


def main() -> None:
    try:
        print(json.dumps(run(), indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"testset-draft failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
