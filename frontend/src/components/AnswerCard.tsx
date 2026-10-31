import { useState } from "react";
import { strings } from "../strings";

interface AnswerCardProps {
  answer: string;
  mode: string;
  timings: Record<string, number>;
  onCite: (n: number) => void;
  streaming?: boolean; // tokens still arriving
  waiting?: boolean; // no token yet
  model?: string | null;
  cached?: boolean;
}

function renderWithCitations(answer: string, onCite: (n: number) => void) {
  const parts = answer.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const n = Number(m[1]);
      return (
        <button
          key={i}
          onClick={() => onCite(n)}
          aria-label={strings.results.sourceAria(n)}
          className="mx-0.5 inline-flex items-center rounded bg-brand-700 px-1.5 text-xs font-semibold text-white align-baseline hover:bg-brand-500"
        >
          {n}
        </button>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function AnswerCard({
  answer, mode, timings, onCite, streaming = false, waiting = false, model, cached = false,
}: AnswerCardProps) {
  const [copied, setCopied] = useState(false);
  const notFound = !streaming && answer.toLowerCase().includes(strings.results.notFoundPhrase);

  async function copy() {
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }

  const totalMs = timings.total ?? 0;

  return (
    <section
      className={
        "rounded-2xl border p-5 shadow-sm " +
        (notFound ? "border-lilac-300 bg-mauve-200" : "border-mauve-200 bg-surface-raised")
      }
      aria-label={strings.results.overview}
      aria-busy={streaming || waiting}
    >
      <div className="mb-3 flex items-center justify-between text-xs text-content-muted">
        <span className="flex items-center gap-2">
          <span className="whitespace-nowrap font-semibold text-content">✦ {strings.results.overview}</span>
          <span className="hidden rounded-full border border-mauve-200 px-2 py-0.5 sm:inline">{mode}</span>
          {cached && <span className="rounded-full border border-mauve-200 px-2 py-0.5">{strings.results.cached}</span>}
        </span>
        <div className="flex items-center gap-3">
          {model && <span className="hidden sm:inline">{model}</span>}
          {!streaming && !waiting && totalMs > 0 && <span>{(totalMs / 1000).toFixed(1)}s</span>}
          {!streaming && !waiting && (
            <button onClick={copy} className="rounded px-2 py-0.5 hover:bg-mauve-200 hover:text-content">
              {copied ? strings.results.copied : strings.results.copy}
            </button>
          )}
        </div>
      </div>
      {waiting ? (
        <div className="space-y-2" aria-label={strings.results.generating}>
          <p className="text-sm text-content-muted">{strings.results.generating}</p>
          <div className="h-3 w-11/12 animate-pulse rounded bg-mauve-200" />
          <div className="h-3 w-9/12 animate-pulse rounded bg-mauve-200" />
          <p className="text-xs text-content-muted">{strings.results.cpuNote}</p>
        </div>
      ) : (
        <p className="whitespace-pre-wrap leading-relaxed text-content">
          {answer ? renderWithCitations(answer, onCite) : strings.results.noAnswer}
          {streaming && <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-brand-500 align-middle" />}
        </p>
      )}
    </section>
  );
}
