"""Generate docs/EVAL_REPORT.md from eval runs (spec §8.3)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import psycopg

from app.config import settings

REPORT_PATH = Path(__file__).resolve().parents[3] / "docs" / "EVAL_REPORT.md"

ABLATION_ROWS = [
    ("doc_recall@5", "retrieval"),
    ("chunk_recall@5", "retrieval"),
    ("doc_mrr@10", "retrieval"),
    ("doc_ndcg@10", "retrieval"),
    ("faithfulness", "ragas"),
    ("answer_relevancy", "ragas"),
    ("context_precision", "ragas"),
    ("context_recall", "ragas"),
]


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def _load(conn, run_ids: list[int]) -> list[dict]:
    rows = conn.execute(
        "SELECT id, config_name, summary FROM eval_runs WHERE id = ANY(%s) ORDER BY id",
        (run_ids,),
    ).fetchall()
    return [{"id": r[0], "mode": r[1], "summary": r[2] or {}} for r in rows]


def generate(run_ids: list[int]) -> Path:
    with psycopg.connect(settings.database_url) as conn:
        runs = _load(conn, run_ids)
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    if runs:
        n = runs[0]["summary"].get("n_questions", "?")
        lines.append(f"Test set: {n} questions. Runs: {', '.join(r['mode'] for r in runs)}.")
        lines.append("")

    # ablation table: rows = metrics, columns = modes; best per row in bold
    header = "| metric | " + " | ".join(r["mode"] for r in runs) + " |"
    sep = "|---|" + "|".join(["---"] * len(runs)) + "|"
    lines += ["## Ablation", "", header, sep]
    for metric, group in ABLATION_ROWS:
        vals = [r["summary"].get(group, {}).get(metric) for r in runs]
        nums = [v for v in vals if isinstance(v, (int, float))]
        best = max(nums) if nums else None
        cells = []
        for v in vals:
            s = _fmt(v)
            if best is not None and isinstance(v, (int, float)) and v == best:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    # operational + behaviour
    lines += ["## Operational & behaviour", "",
              "| mode | latency p50 (ms) | latency p95 (ms) | avg LLM calls | unanswerable refusal |",
              "|---|---|---|---|---|"]
    for r in runs:
        op = r["summary"].get("operational", {})
        lat = op.get("latency_ms", {})
        beh = r["summary"].get("behaviour", {})
        lines.append(
            f"| {r['mode']} | {lat.get('p50', '—')} | {lat.get('p95', '—')} | "
            f"{_fmt(op.get('avg_llm_calls'))} | {_fmt(beh.get('unanswerable_refusal_rate'))} |"
        )
    lines.append("")

    ragas_status = {r["mode"]: r["summary"].get("ragas", {}).get("ragas_status") for r in runs}
    if any(s and s != "ok" for s in ragas_status.values()):
        lines += ["> RAGAS note: generation metrics require a working LLM judge. Status per run:",
                  ""]
        for mode, st in ragas_status.items():
            lines.append(f"> - {mode}: {st}")
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH
