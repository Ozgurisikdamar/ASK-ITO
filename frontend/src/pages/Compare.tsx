import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { ThemeToggle } from "../components/ThemeToggle";
import { CompareRow, getCompare } from "../api/client";

export function Compare() {
  const { a, b } = useParams();
  const [data, setData] = useState<{
    metric: string;
    config_a?: string | null;
    config_b?: string | null;
    improved: CompareRow[];
    regressed: CompareRow[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCompare(Number(a), Number(b)).then(setData).catch((e) => setError(String(e.message ?? e)));
  }, [a, b]);

  const ca = data?.config_a || `run #${a}`;
  const cb = data?.config_b || `run #${b}`;

  return (
    <div className="min-h-screen bg-surface">
      <header className="flex items-center gap-3 border-b border-mauve-200 px-4 py-3">
        <Link to="/" aria-label="Ask ITO home"><Logo variant="small" /></Link>
        <Link to="/eval" className="flex-1 text-sm text-link hover:underline">← Evaluation</Link>
        <ThemeToggle />
      </header>
      <main className="mx-auto max-w-3xl px-4 py-6">
        {error && <p className="text-content-muted">{error}</p>}
        {data && (
          <>
            <h1 className="font-display text-xl font-bold text-content">
              Compare: {ca} vs {cb}
            </h1>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm text-content-muted">
              <span><span className="font-semibold text-content">A</span> = run #{a} · {ca}</span>
              <span><span className="font-semibold text-content">B</span> = run #{b} · {cb}</span>
            </div>
            <p className="mt-1 text-xs text-content-muted">
              metric: {data.metric} · each row shows {ca} → {cb} (Δ = B − A)
            </p>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <Column title={`Improved in B (${cb})`} rows={data.improved} positive />
              <Column title={`Regressed in B (${cb})`} rows={data.regressed} positive={false} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

function Column({ title, rows, positive }: { title: string; rows: CompareRow[]; positive: boolean }) {
  return (
    <div className="rounded-xl border border-mauve-200 bg-surface-raised p-3">
      <h2 className="mb-2 font-semibold text-content">{title}</h2>
      {rows.length === 0 && <p className="text-sm text-content-muted">none</p>}
      <ul className="space-y-1 text-sm">
        {rows.map((r) => (
          <li key={r.question_id} className="flex items-center justify-between gap-2">
            <span className="text-content-muted">{r.question_id}</span>
            <span className={positive ? "text-green-600" : "text-red-600"}>
              {r.a.toFixed(2)} → {r.b.toFixed(2)} ({r.delta > 0 ? "+" : ""}{r.delta.toFixed(2)})
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
