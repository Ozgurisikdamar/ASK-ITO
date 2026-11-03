import { useEffect, useMemo, useState } from "react";
import { strings } from "../strings";
import { Link } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Logo } from "../components/Logo";
import { ThemeToggle } from "../components/ThemeToggle";
import { EvalRun, getRuns, startRun } from "../api/client";

const ALL_CONFIGS = [
  "closed_book", "bm25", "vector", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite", "agentic",
];
// these score the answer itself, so they need "Generate answers"
const ANSWER_ONLY = ["closed_book", "agentic"];
const RETRIEVAL_METRICS = ["doc_recall@5", "chunk_recall@5", "doc_mrr@10", "doc_ndcg@10"];
const RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"];

function num(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}

export function EvalPanel() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>(["bm25", "vector", "hybrid", "hybrid_rerank"]);
  const [answers, setAnswers] = useState(false);
  const [ragas, setRagas] = useState(false);
  const [starting, setStarting] = useState(false);

  function load() {
    getRuns()
      .then((d) => {
        setRuns(d.runs);
        setError(null);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }
  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  // latest run per config
  const latest = useMemo(() => {
    const m: Record<string, EvalRun> = {};
    for (const r of runs) if (r.status === "done" && !m[r.config_name]) m[r.config_name] = r;
    return ALL_CONFIGS.map((c) => m[c]).filter(Boolean) as EvalRun[];
  }, [runs]);

  const active = runs.some((r) => r.status === "running" || r.status === "queued");

  const chartData = latest.map((r) => ({
    mode: r.config_name,
    "doc_recall@5": num(r.retrieval["doc_recall@5"]) ?? 0,
    "doc_ndcg@10": num(r.retrieval["doc_ndcg@10"]) ?? 0,
  }));

  function toggleAnswers(on: boolean) {
    setAnswers(on);
    if (!on) {
      setRagas(false);
      setSelected((s) => s.filter((c) => !ANSWER_ONLY.includes(c)));
    }
  }

  async function run() {
    setStarting(true);
    setError(null);
    setNotice(null);
    try {
      const r = await startRun(selected, { answers, ragas });
      const skipped = r.skipped.length
        ? ` ${r.skipped.length} left out (gold article not in this corpus).`
        : "";
      setNotice(`Queued ${r.configs.length} configuration(s) over ${r.questions} questions.${skipped}`);
      load();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="min-h-screen bg-surface">
      <header className="flex items-center gap-2 border-b sm:gap-3 border-mauve-200 px-4 py-3">
        <Link to="/" aria-label="Ask ITO home"><Logo variant="small" /></Link>
        <h1 className="min-w-0 flex-1 truncate font-display text-lg font-bold text-content">Evaluation</h1>
        <Link to="/eval/review" className="shrink-0 text-sm text-link hover:underline">Test set review</Link>
        <ThemeToggle />
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        <section className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
          <h2 className="mb-2 font-semibold text-content">Run evaluation</h2>
          <p className="mb-3 text-xs text-content-muted">{strings.ragasOffline}</p>
          <div className="mb-3 flex flex-wrap gap-2">
            {ALL_CONFIGS.map((c) => {
              const needsAnswers = !answers && ANSWER_ONLY.includes(c);
              return (
                <label
                  key={c}
                  title={needsAnswers ? "Needs “Generate answers”" : undefined}
                  className={
                    "flex items-center gap-1 rounded-full border border-mauve-200 px-3 py-1 text-sm text-content-muted" +
                    (needsAnswers ? " opacity-50" : "")
                  }
                >
                  <input
                    type="checkbox"
                    disabled={needsAnswers}
                    checked={selected.includes(c)}
                    onChange={(e) =>
                      setSelected(e.target.checked ? [...selected, c] : selected.filter((x) => x !== c))
                    }
                  />
                  {c}
                </label>
              );
            })}
          </div>
          <div className="mb-3 space-y-1 text-sm text-content-muted">
            <label className="flex items-start gap-2">
              <input type="checkbox" className="mt-1" checked={answers} onChange={(e) => toggleAnswers(e.target.checked)} />
              <span>
                Generate answers with the local model (needed for closed_book, agentic, refusal and
                RAGAS metrics). Slow on CPU: about a minute per question, and AI Mode answers wait
                for the model meanwhile. Off: retrieval metrics only, a few minutes in total.
              </span>
            </label>
            <label className={"flex items-start gap-2" + (answers ? "" : " opacity-50")}>
              <input type="checkbox" className="mt-1" disabled={!answers} checked={ragas} onChange={(e) => setRagas(e.target.checked)} />
              <span>RAGAS judge over the answers (many more model calls; much slower).</span>
            </label>
          </div>
          <button
            onClick={run}
            disabled={starting || active || selected.length === 0}
            className="rounded-full bg-brand-700 px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {starting ? "Starting…" : active ? "Running…" : "Run selected"}
          </button>
          {notice && <p className="mt-2 text-sm text-content-muted" role="status">{notice}</p>}
          {error && (
            <p className="mt-2 rounded-md border border-lilac-300 bg-mauve-200 px-3 py-2 text-sm font-medium text-content" role="alert" data-testid="eval-error">
              {error}
            </p>
          )}
          <RecentRuns runs={runs} />
        </section>

        <section className="overflow-x-auto rounded-2xl border border-mauve-200 bg-surface-raised p-4">
          <h2 className="mb-3 font-semibold text-content">Ablation</h2>
          <AblationTable latest={latest} />
        </section>

        <section className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
          <h2 className="mb-3 font-semibold text-content">Retrieval quality by mode</h2>
          {chartData.length === 0 ? (
            <p className="text-sm text-content-muted">No completed runs yet.</p>
          ) : (
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--mauve-200)" />
                <XAxis dataKey="mode" tick={{ fill: "var(--text-muted)", fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                <YAxis domain={[0, 1]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "var(--surface-raised)", border: "1px solid var(--mauve-200)", color: "var(--text)" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="doc_recall@5" fill="var(--brand-500)" />
                <Bar dataKey="doc_ndcg@10" fill="var(--brand-900)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          )}
        </section>

        <CompareSection runs={latest} />
      </main>
    </div>
  );
}

function statusText(r: EvalRun): string {
  if (r.status === "queued") return "queued";
  if (r.status === "running")
    return r.progress ? `running ${r.progress.done}/${r.progress.total}` : "running";
  if (r.status === "error") return `failed: ${r.error ?? "unknown error"}`;
  return r.status;
}

function RecentRuns({ runs }: { runs: EvalRun[] }) {
  const recent = runs.slice(0, 8);
  if (recent.length === 0) return null;
  return (
    <div className="mt-4">
      <h3 className="mb-1 text-sm font-semibold text-content">Recent runs</h3>
      <ul className="space-y-0.5 text-sm" data-testid="recent-runs">
        {recent.map((r) => (
          <li key={r.id} className="flex flex-wrap gap-x-2 text-content-muted">
            <span className="tabular-nums">#{r.id}</span>
            <span className="text-content">{r.config_name}</span>
            {r.options && <span>{r.options.answers ? (r.options.ragas ? "answers + RAGAS" : "answers") : "retrieval only"}</span>}
            <span className={r.status === "error" ? "font-medium text-content" : ""}>{statusText(r)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function AblationTable({ latest }: { latest: EvalRun[] }) {
  if (latest.length === 0) return <p className="text-sm text-content-muted">No completed runs yet.</p>;
  const rows = [
    ...RETRIEVAL_METRICS.map((m) => ({ m, group: "retrieval" as const })),
    ...RAGAS_METRICS.map((m) => ({ m, group: "ragas" as const })),
  ];
  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="text-content-muted">
          <th className="p-2 text-left">metric</th>
          {latest.map((r) => (
            <th key={r.id} className="p-2 text-right">
              <Link to={`/eval/runs/${r.id}`} className="text-link hover:underline">{r.config_name}</Link>
              {r.options && !r.options.answers && (
                <span className="block text-xs font-normal">retrieval only</span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(({ m, group }) => {
          const vals = latest.map((r) => num((r as any)[group]?.[m]));
          const best = Math.max(...(vals.filter((v) => v != null) as number[]), -Infinity);
          return (
            <tr key={m} className="border-t border-mauve-200">
              <td className="p-2 text-content">{m}</td>
              {vals.map((v, i) => (
                <td key={i} className={"p-2 text-right " + (v != null && v === best ? "font-bold text-content" : "text-content-muted")}>
                  {v != null ? v.toFixed(3) : "—"}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function CompareSection({ runs }: { runs: EvalRun[] }) {
  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);
  if (runs.length < 2) return null;
  return (
    <section className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
      <h2 className="mb-3 font-semibold text-content">Compare runs</h2>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <RunSelect runs={runs} value={a} onChange={setA} />
        <span className="text-content-muted">vs</span>
        <RunSelect runs={runs} value={b} onChange={setB} />
        {a && b && a !== b && (
          <Link to={`/eval/compare/${a}/${b}`} className="rounded-full bg-brand-700 px-3 py-1 text-white hover:opacity-90">
            Compare
          </Link>
        )}
      </div>
    </section>
  );
}

function RunSelect({ runs, value, onChange }: { runs: EvalRun[]; value: number | null; onChange: (n: number) => void }) {
  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(Number(e.target.value))}
      className="rounded-md border border-mauve-200 bg-surface p-1.5 text-content"
    >
      <option value="">select run…</option>
      {runs.map((r) => (
        <option key={r.id} value={r.id}>#{r.id} {r.config_name}</option>
      ))}
    </select>
  );
}
