import { TraceStep } from "../api/client";
import { strings } from "../strings";

function count(v: unknown): number | null {
  if (Array.isArray(v)) return v.length;
  return null;
}

function summarize(step: TraceStep): string {
  const out = step.output;
  switch (step.step) {
    case "analyze": {
      const o = out as { qtype?: string; subquestions?: string[] } | undefined;
      const n = o?.subquestions?.length ?? 0;
      return `${o?.qtype ?? "?"} · ${n} sub-question${n === 1 ? "" : "s"}`;
    }
    case "rewrite":
      return typeof out === "string" ? `→ "${out}"` : "";
    case "sufficiency": {
      const o = out as { sufficient?: boolean; next_query?: string | null } | undefined;
      return o?.sufficient ? "sufficient ✓" : `need more → "${o?.next_query ?? ""}"`;
    }
    case "answer":
      return typeof out === "string" ? `${out.slice(0, 80)}…` : "";
    default: {
      const c = count(out) ?? count(step.candidate_ids);
      return c != null ? `${c} candidate${c === 1 ? "" : "s"}` : "";
    }
  }
}

export function TracePanel({ trace }: { trace: TraceStep[] }) {
  if (!trace.length) return null;
  return (
    <details className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
      <summary className="cursor-pointer font-medium text-content">
        {strings.results.howItSearched}
        <span className="ml-2 text-xs text-content-muted">({trace.length} steps)</span>
      </summary>
      <ol className="mt-3 space-y-2">
        {trace.map((s, i) => (
          <li key={i} className="flex items-start gap-3 text-sm">
            <span className="mt-0.5 min-w-[92px] rounded-full bg-mauve-200 px-2 py-0.5 text-center text-xs font-medium text-brand-900">
              {s.step}
            </span>
            <div className="flex-1">
              {typeof s.input === "string" && s.input && (
                <p className="text-content-muted">
                  <span className="text-content">input:</span> {s.input}
                </p>
              )}
              <p className="text-content-muted">{summarize(s)}</p>
            </div>
            <span className="shrink-0 text-xs text-content-muted">{s.duration_ms}ms</span>
          </li>
        ))}
      </ol>
    </details>
  );
}
