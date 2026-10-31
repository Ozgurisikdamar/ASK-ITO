import { useEffect, useReducer, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { SearchBar } from "../components/SearchBar";
import { ThemeToggle } from "../components/ThemeToggle";
import { AnswerCard } from "../components/AnswerCard";
import { OrganicResult } from "../components/OrganicResult";
import { SourceCard } from "../components/SourceCard";
import { TracePanel } from "../components/TracePanel";
import { openSearchStream } from "../api/client";
import { initialState, reducer } from "../search/state";
import { strings } from "../strings";

const DEFAULT_MODE = "overview";

export function Results() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const q = params.get("q") ?? "";
  const mode = params.get("mode") ?? DEFAULT_MODE;
  const domain = params.get("domain") ?? "all";

  const [query, setQuery] = useState(q);
  const [state, dispatch] = useReducer(reducer, initialState);
  const [highlight, setHighlight] = useState<number | null>(null);
  const clearTimer = useRef<number | undefined>(undefined);
  const startedAt = useRef<number>(0);

  useEffect(() => {
    setQuery(q);
    if (!q.trim()) return;
    dispatch({ type: "reset" });
    startedAt.current = performance.now();
    const close = openSearchStream(
      { query: q, mode, domain },
      {
        // Stats line shows the time the user actually waited for the first results.
        onResults: (data) =>
          dispatch({
            type: "results",
            data: {
              ...data,
              timings: { ...data.timings, client: Math.round(performance.now() - startedAt.current) },
            },
          }),
        onReranked: (data) => dispatch({ type: "reranked", data }),
        onSources: (sources) => dispatch({ type: "sources", sources }),
        onStep: (step) => dispatch({ type: "step", step }),
        onToken: (text) => dispatch({ type: "token", text }),
        onDone: (data) => dispatch({ type: "done", data }),
        onError: (e) => dispatch({ type: "error", detail: e.detail || e.error }),
      }
    );
    return close; // leaving the page / new query closes the stream (stops generation)
  }, [q, mode, domain]);

  function onCite(n: number) {
    setHighlight(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    window.clearTimeout(clearTimer.current);
    clearTimer.current = window.setTimeout(() => setHighlight(null), 2500);
  }

  function go(next: { q?: string; mode?: string }) {
    const nq = next.q ?? query;
    if (!nq.trim()) return;
    const sp = new URLSearchParams({ q: nq, mode: next.mode ?? mode, domain });
    navigate(`/search?${sp.toString()}`);
  }

  const answering = state.phase === "answering" || state.phase === "searching";
  const shownAnswer = state.answer ?? state.streamText;
  const resultsMs = state.timings.client ?? state.timings.results ?? state.timings.retrieval ?? 0;

  return (
    <div className="min-h-screen bg-surface">
      <header className="sticky top-0 z-10 border-b border-mauve-200 bg-surface px-4 pt-3">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 sm:flex-nowrap">
          <Link to="/" aria-label="Ask ITO home" className="shrink-0">
            <Logo variant="small" />
          </Link>
          <div className="order-last w-full min-w-0 sm:order-none sm:max-w-xl sm:flex-1">
            <SearchBar value={query} onChange={setQuery} onSubmit={() => go({})} size="md" />
          </div>
          <div className="ml-auto sm:ml-0">
            <ThemeToggle />
          </div>
        </div>
        <nav className="mt-2 flex gap-4 text-sm" aria-label="Search mode">
          {[
            { id: "overview", label: strings.modes.search },
            { id: "agentic", label: strings.modes.aiMode },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => go({ mode: t.id })}
              aria-current={mode === t.id ? "page" : undefined}
              className={
                "border-b-2 pb-2 " +
                (mode === t.id
                  ? "border-brand-700 font-medium text-content"
                  : "border-transparent text-content-muted hover:text-content")
              }
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-3xl space-y-6 px-4 py-6">
        {state.phase === "error" && (
          <div className="rounded-2xl border border-lilac-300 bg-mauve-200 p-5">
            <h2 className="font-semibold text-content">{strings.results.errorTitle}</h2>
            <p className="mt-1 text-sm text-content-muted">{state.error}</p>
          </div>
        )}

        {state.results.length > 0 && (
          <p className="text-sm text-content-muted">
            {strings.results.stats(state.results.length, resultsMs)}
          </p>
        )}

        {mode !== "closed_book" || shownAnswer ? (
          <AnswerCard
            answer={shownAnswer}
            mode={mode}
            timings={state.timings}
            onCite={onCite}
            streaming={state.phase === "answering" && !state.answer}
            waiting={answering && !shownAnswer}
            model={state.model}
            cached={state.cached && state.phase === "done"}
          />
        ) : null}

        {state.sources.length > 0 && (
          <details className="rounded-2xl border border-mauve-200 bg-surface-raised p-4">
            <summary className="cursor-pointer text-sm font-medium text-content">
              {strings.results.sourcesUsed} ({state.sources.length})
            </summary>
            <div className="mt-3 space-y-3">
              {state.sources.map((s, i) => (
                <SourceCard
                  key={s.chunk_id}
                  source={s}
                  n={s.n ?? i + 1}
                  query={q}
                  highlighted={highlight === (s.n ?? i + 1)}
                />
              ))}
            </div>
          </details>
        )}

        {state.steps.length > 0 && <TracePanel trace={state.steps} />}

        {state.phase === "searching" && state.results.length === 0 && <LoadingSkeleton />}

        {state.results.length > 0 && (
          <section aria-label={strings.results.webResults} className="space-y-6">
            {state.results.slice(0, 10).map((r) => (
              <OrganicResult key={r.chunk_id} r={r} query={q} />
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6" aria-label={strings.results.searching} aria-busy="true">
      {[0, 1, 2].map((i) => (
        <div key={i} className="space-y-2">
          <div className="h-3 w-40 animate-pulse rounded bg-surface-raised" />
          <div className="h-5 w-72 animate-pulse rounded bg-surface-raised" />
          <div className="h-3 w-full animate-pulse rounded bg-surface-raised" />
        </div>
      ))}
    </div>
  );
}
