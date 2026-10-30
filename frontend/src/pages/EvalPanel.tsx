import { useEffect, useMemo, useState } from "react";
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
const RETRIEVAL_METRICS = ["doc_recall@5", "chunk_recall@5", "doc_mrr@10", "doc_ndcg@10"];
const RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"];

function num(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}

export function EvalPanel() {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>(["bm25", "vector", "hybrid", "hybrid_rerank"]);
  const [starting, setStarting] = useState(false);

  function load() {
    getRuns().then((d) => setRuns(d.runs)).catch((e) => setError(String(e.message ?? e)));
  }
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  // latest run per config
  const latest = useMemo(() => {
    const m: Record<string, EvalRun> = {};
    for (const r of runs) if (r.status === "done" && !m[r.config_name]) m[r.config_name] = r;
    return ALL_CONFIGS.map((c) => m[c]).filter(Boolean) as EvalRun[];
  }, [runs]);

  const running = runs.filter((r) => r.status === "running");

  const chartData = latest.map((r) => ({
    mode: r.config_name,
    "doc_recall@5": num(r.retrieval["doc_recall@5"]) ?? 0,
    "doc_ndcg@10": num(r.retrieval["doc_ndcg@10"]) ?? 0,
  }));

  async function run() {
    setStarting(true);
    try {
      await startRun(selected);
      setTimeout(load, 1000);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="min-h-screen bg-surface">
      <header className="flex items-center gap-3 border-b border-mauve-200 px-4 py-3">
        <Link to="/" aria-label="Ask ITO home"><Logo variant="small" /></Link>
        <h1 className="flex-1 font-display text-lg font-bold text-content">Evaluation</h1>
        <Link to="/eval/review" className="text-sm text-link hover:underline">Test set review</Link>
        <ThemeToggle />
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-4 py-6">
        {error && <p className="text-sm text-content-muted">{error}</p>}

        <section className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
          <h2 className="mb-2 font-semibold text-content">Run evaluation</h2>
          <div className="mb-3 flex flex-wrap gap-2">
            {ALL_CONFIGS.map((c) => (
              <label key={c} className="flex items-center gap-1 rounded-full border border-mauve-200 px-3 py-1 text-sm text-content-muted">
                <input
                  type="checkbox"
                  checked={selected.includes(c)}
                  onChange={(e) =>
                    setSelected(e.target.checked ? [...selected, c] : selected.filter((x) => x !== c))
                  }
                />
                {c}
              </label>
            ))}
          </div>
          <button
            onClick={run}
            disabled={starting || selected.length === 0}
            className="rounded-full bg-brand-700 px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {starting ? "Starting…" : "Run selected"}
          </button>
          {running.length > 0 && (
            <span className="ml-3 text-sm text-content-muted">
              running: {running.map((r) => r.config_name).join(", ")} …
            </span>
          )}
        </section>

        <section className="overflow-x-auto rounded-2xl border border-mauve-200 bg-surface-raised p-4">
          <h2 className="mb-3 font-semibold text-content">Ablation</h2>
          <AblationTable latest={latest} />
        </section>

        <section className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
          <h2 className="mb-3 font-semibold text-content">Retrieval quality by mode</h2>
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
        </section>

        <CompareSection runs={latest} />
      </main>
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
