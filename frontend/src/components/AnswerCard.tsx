import { useState } from "react";
import { strings } from "../strings";

interface AnswerCardProps {
  answer: string;
  mode: string;
  timings: Record<string, number>;
  onCite: (n: number) => void;
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

export function AnswerCard({ answer, mode, timings, onCite }: AnswerCardProps) {
  const [copied, setCopied] = useState(false);
  const notFound = answer.toLowerCase().includes(strings.results.notFoundPhrase);

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
        (notFound
          ? "border-lilac-300 bg-mauve-200"
          : "border-mauve-200 bg-surface-raised")
      }
      aria-label="Answer"
    >
      <div className="mb-3 flex items-center justify-between text-xs text-content-muted">
        <span className="rounded-full border border-mauve-200 px-2 py-0.5">{mode}</span>
        <div className="flex items-center gap-3">
          <span>{(totalMs / 1000).toFixed(1)}s</span>
          <button onClick={copy} className="rounded px-2 py-0.5 hover:bg-mauve-200 hover:text-content">
            {copied ? strings.results.copied : strings.results.copy}
          </button>
        </div>
      </div>
      <p className="whitespace-pre-wrap leading-relaxed text-content">
        {answer ? renderWithCitations(answer, onCite) : strings.results.noAnswer}
      </p>
    </section>
  );
}
