import { useState } from "react";
import type { StatusEvent, TraceStep } from "../api/client";
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
  status?: StatusEvent | null; // queued / loading the local model
  lastStep?: TraceStep | null; // AI Mode: the agent step in progress
  error?: string | null; // generation failed (results and sources stay)
  errorCode?: string | null;
  onRetry?: () => void;
}

/** One line describing what is happening while no answer text is shown yet. */
function progressText(status?: StatusEvent | null, step?: TraceStep | null): string {
  if (status?.stage === "queued") return strings.results.queued;
  if (status?.stage === "model_loading")
    return strings.results.modelLoading(status.model ?? "", status.elapsed_s ?? 0);
  if (step) return strings.results.agentStep(step.step, step.input);
  return strings.results.generating;
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
  status = null, lastStep = null, error = null, errorCode = null, onRetry,
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
        (notFound || error ? "border-lilac-300 bg-mauve-200" : "border-mauve-200 bg-surface-raised")
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
      {answer && (
        <p className="whitespace-pre-wrap leading-relaxed text-content">
          {renderWithCitations(answer, onCite)}
          {streaming && <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-brand-500 align-middle" />}
        </p>
      )}
      {error ? (
        <div role="alert" className={answer ? "mt-3 border-t border-lilac-300 pt-3" : ""}>
          <p className="text-sm font-medium text-content">
            {answer ? strings.results.answerInterrupted : strings.results.answerFailed}
          </p>
          <p className="mt-1 text-sm text-content-muted" data-error-code={errorCode ?? undefined}>
            {error}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 rounded-full bg-brand-700 px-4 py-1.5 text-sm font-medium text-white hover:bg-brand-500"
            >
              {strings.results.retry}
            </button>
          )}
        </div>
      ) : waiting ? (
        <div className="space-y-2" aria-live="polite">
          <p className="text-sm text-content-muted">{progressText(status, lastStep)}</p>
          <div className="h-3 w-11/12 animate-pulse rounded bg-mauve-200" />
          <div className="h-3 w-9/12 animate-pulse rounded bg-mauve-200" />
        </div>
      ) : !answer ? (
        <p className="text-sm text-content-muted">{strings.results.noAnswer}</p>
      ) : null}
    </section>
  );
}
