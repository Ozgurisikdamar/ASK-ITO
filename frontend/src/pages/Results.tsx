import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { SearchBar } from "../components/SearchBar";
import { ThemeToggle } from "../components/ThemeToggle";
import { AnswerCard } from "../components/AnswerCard";
import { SourceCard } from "../components/SourceCard";
import { TracePanel } from "../components/TracePanel";
import { search, SearchResponse } from "../api/client";
import { strings } from "../strings";

export function Results() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const q = params.get("q") ?? "";
  const mode = params.get("mode") ?? "hybrid_rerank";
  const domain = params.get("domain") ?? "all";

  const [query, setQuery] = useState(q);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);
  const clearTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    setQuery(q);
    if (!q.trim()) return;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);
    search({ query: q, mode, domain }, ctrl.signal)
      .then((r) => setData(r))
      .catch((e) => {
        if (e?.name !== "AbortError") setError(String(e.message ?? e));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [q, mode, domain]);

  function onCite(n: number) {
    setHighlight(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    window.clearTimeout(clearTimer.current);
    clearTimer.current = window.setTimeout(() => setHighlight(null), 2500);
  }

  function submit() {
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query)}&mode=${mode}&domain=${domain}`);
    }
  }

  return (
    <div className="min-h-screen bg-surface">
      <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-mauve-200 bg-surface px-4 py-3">
        <Link to="/" aria-label="Ask ITO home" className="shrink-0">
          <Logo variant="small" />
        </Link>
        <div className="max-w-xl flex-1">
          <SearchBar value={query} onChange={setQuery} onSubmit={submit} size="md" />
        </div>
        <ThemeToggle />
      </header>

      <main className="mx-auto max-w-3xl space-y-6 px-4 py-8">
        {loading && <LoadingSkeleton />}

        {error && (
          <div className="rounded-2xl border border-lilac-300 bg-mauve-200 p-5">
            <h2 className="font-semibold text-content">{strings.results.errorTitle}</h2>
            <p className="mt-1 text-sm text-content-muted">{error}</p>
          </div>
        )}

        {data && !loading && (
          <>
            <AnswerCard
              answer={data.answer}
              mode={data.mode}
              timings={data.timings}
              onCite={onCite}
            />

            {data.sources.length > 0 && (
              <section aria-label="Sources" className="space-y-3">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-content-muted">
                  {strings.results.sources}
                </h2>
                {data.sources.map((s, i) => (
                  <SourceCard
                    key={s.chunk_id}
                    source={s}
                    n={i + 1}
                    query={q}
                    highlighted={highlight === i + 1}
                  />
                ))}
              </section>
            )}

            <TracePanel trace={data.trace} />
          </>
        )}
      </main>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6" aria-label={strings.results.searching} aria-busy="true">
      <div className="h-32 animate-pulse rounded-2xl bg-surface-raised" />
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl bg-surface-raised" />
        ))}
      </div>
    </div>
  );
}
