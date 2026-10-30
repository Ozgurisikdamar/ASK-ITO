import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { ThemeToggle } from "../components/ThemeToggle";
import { getRunDetail, RunDetail as RunDetailT, RunResult } from "../api/client";
import { DomainBadge } from "../components/DomainBadge";

export function RunDetail() {
  const { id } = useParams();
  const [data, setData] = useState<RunDetailT | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [qtype, setQtype] = useState("all");
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    getRunDetail(Number(id)).then(setData).catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  const filtered = useMemo(
    () => (data?.results ?? []).filter((r) => qtype === "all" || r.qtype === qtype),
    [data, qtype]
  );
  const qtypes = useMemo(
    () => Array.from(new Set((data?.results ?? []).map((r) => r.qtype))),
    [data]
  );

  return (
    <div className="min-h-screen bg-surface">
      <header className="flex items-center gap-3 border-b border-mauve-200 px-4 py-3">
        <Link to="/" aria-label="Ask ITO home"><Logo variant="small" /></Link>
        <Link to="/eval" className="flex-1 text-sm text-link hover:underline">← Evaluation</Link>
        <ThemeToggle />
      </header>
      <main className="mx-auto max-w-4xl px-4 py-6">
        {error && <p className="text-content-muted">{error}</p>}
        {data && (
          <>
            <h1 className="font-display text-xl font-bold text-content">
              Run #{data.id} · {data.config_name}
            </h1>
            <p className="mt-1 text-xs text-content-muted">
              status {data.status} · {data.results.length} questions ·
              commit {(data.config_json as any)?.git_commit ?? "?"}
            </p>

            <div className="my-4 flex gap-2 text-sm">
              <select value={qtype} onChange={(e) => setQtype(e.target.value)}
                className="rounded-md border border-mauve-200 bg-surface p-1.5 text-content">
                <option value="all">all types</option>
                {qtypes.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>

            <div className="space-y-2">
              {filtered.map((r) => (
                <QuestionRow key={r.question_id} r={r}
                  open={open === r.question_id}
                  onToggle={() => setOpen(open === r.question_id ? null : r.question_id)} />
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  );
}

function metric(r: RunResult, key: string): string {
  const v = r.metrics[key];
  return typeof v === "number" ? v.toFixed(2) : "—";
}

function QuestionRow({ r, open, onToggle }: { r: RunResult; open: boolean; onToggle: () => void }) {
  const goldSet = new Set(r.gold_chunk_ids);
  const retrievedTop = r.retrieved_chunk_ids.slice(0, 10);
  return (
    <article className="rounded-xl border border-mauve-200 bg-surface-raised p-3">
      <button onClick={onToggle} className="flex w-full items-center justify-between text-left">
        <span className="flex items-center gap-2 text-sm">
          <span className="rounded-full bg-mauve-200 px-2 py-0.5 text-xs text-content">{r.qtype}</span>
          <DomainBadge domain={r.domain} />
          <span className="text-content">{r.question}</span>
        </span>
        <span className="ml-2 shrink-0 text-xs text-content-muted">
          nDCG {metric(r, "doc_ndcg@10")} · R@5 {metric(r, "doc_recall@5")}
        </span>
      </button>
      {open && (
        <div className="mt-3 space-y-3 border-t border-mauve-200 pt-3 text-sm">
          {r.reference && (
            <p className="text-content-muted"><span className="text-content">Reference:</span> {r.reference}</p>
          )}
          <p className="text-content-muted"><span className="text-content">Answer:</span> {r.answer}</p>
          <div>
            <p className="mb-1 text-content">Gold chunks</p>
            <div className="flex flex-wrap gap-1">
              {r.gold_chunk_ids.length === 0 && <span className="text-xs text-content-muted">(none — unanswerable)</span>}
              {r.gold_chunk_ids.map((c) => (
                <span key={c} className={"rounded px-1.5 py-0.5 text-xs " +
                  (retrievedTop.includes(c) ? "bg-green-600 text-white" : "bg-red-600 text-white")}>
                  {c}
                </span>
              ))}
            </div>
          </div>
          <div>
            <p className="mb-1 text-content">Retrieved (top 10)</p>
            <div className="flex flex-wrap gap-1">
              {retrievedTop.map((c) => (
                <span key={c} className={"rounded px-1.5 py-0.5 text-xs " +
                  (goldSet.has(c) ? "bg-green-600 text-white" : "bg-mauve-200 text-content")}>
                  {c}
                </span>
              ))}
            </div>
          </div>
          <details>
            <summary className="cursor-pointer text-xs text-content-muted">trace ({r.trace.length} steps)</summary>
            <pre className="mt-1 max-h-48 overflow-auto rounded bg-surface p-2 text-xs text-content-muted">
              {JSON.stringify(r.trace, null, 1)}
            </pre>
          </details>
        </div>
      )}
    </article>
  );
}
