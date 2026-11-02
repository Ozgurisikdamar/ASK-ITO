import { strings } from "../strings";

const ORDER = ["embedding", "bm25", "vector", "rrf", "rerank", "context", "first_token", "answer", "total"];

function fmt(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

/** Per-stage timings reported by the server for this search, plus the models used. */
export function LatencyBreakdown({
  timings,
  model,
  reranker,
}: {
  timings: Record<string, number>;
  model: string | null;
  reranker: string | null;
}) {
  const rows = ORDER.filter((k) => typeof timings[k] === "number");
  if (!rows.length) return null;
  return (
    <details className="rounded-2xl border border-mauve-200 bg-surface-raised p-4 text-sm">
      <summary className="cursor-pointer font-medium text-content">{strings.results.latency}</summary>
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-3">
        {rows.map((k) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-content-muted">{strings.results.stages[k] ?? k}</dt>
            <dd className={"tabular-nums " + (k === "total" ? "font-semibold text-content" : "text-content")}>
              {fmt(timings[k])}
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-xs text-content-muted">
        {strings.results.llmModel}: {model ?? "–"} · {strings.results.reranker}:{" "}
        {reranker ?? strings.results.rerankerOff}
      </p>
    </details>
  );
}
