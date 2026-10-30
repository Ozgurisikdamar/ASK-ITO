import { Source } from "../api/client";
import { strings } from "../strings";
import { DomainBadge } from "./DomainBadge";

function highlight(text: string, query: string) {
  const terms = Array.from(new Set(query.toLowerCase().match(/\w+/g) ?? []))
    .filter((t) => t.length >= 4)
    .sort((a, b) => b.length - a.length);
  if (terms.length === 0) return text;
  const termSet = new Set(terms);
  const re = new RegExp(`(${terms.map(escapeRe).join("|")})`, "gi");
  return text.split(re).map((part, i) =>
    termSet.has(part.toLowerCase()) ? (
      <mark key={i} className="rounded bg-mauve-200 px-0.5 text-content">
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

interface SourceCardProps {
  source: Source;
  n: number;
  query: string;
  highlighted: boolean;
}

export function SourceCard({ source, n, query, highlighted }: SourceCardProps) {
  return (
    <article
      id={`source-${n}`}
      className={
        "scroll-mt-24 rounded-xl border p-4 transition-colors " +
        (highlighted
          ? "border-brand-500 bg-mauve-200 ring-2 ring-brand-500"
          : "border-mauve-200 bg-surface-raised")
      }
    >
      <div className="mb-1 flex items-start justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold text-brand-700">[{n}]</span>
          <h3 className="font-semibold text-content">{source.doc_title}</h3>
        </div>
        <DomainBadge domain={source.domain} />
      </div>
      {source.section && (
        <p className="mb-1 text-xs uppercase tracking-wide text-content-muted">{source.section}</p>
      )}
      <p className="text-sm leading-relaxed text-content-muted">
        {highlight(source.snippet, query)}
      </p>
      <div className="mt-2 flex items-center gap-3 text-xs">
        <a
          href={source.url}
          target="_blank"
          rel="noreferrer"
          className="text-link hover:underline"
        >
          {strings.results.openWiki} ↗
        </a>
        {source.score != null && (
          <span className="text-content-muted">score {source.score.toFixed(3)}</span>
        )}
      </div>
    </article>
  );
}
