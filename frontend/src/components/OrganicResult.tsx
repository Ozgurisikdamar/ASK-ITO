import type { ResultItem } from "../api/client";
import { DomainBadge } from "./DomainBadge";
import { highlight } from "./highlight";

/** Google-style result: source breadcrumb, title link, highlighted snippet. */
export function OrganicResult({ r, query }: { r: ResultItem; query: string }) {
  const crumb = ["en.wikipedia.org", "wiki", r.doc_title].join(" › ");
  return (
    <article className="max-w-2xl">
      <div className="mb-0.5 flex items-center gap-2 text-xs text-content-muted">
        <span
          aria-hidden="true"
          className="flex h-6 w-6 items-center justify-center rounded-full bg-mauve-200 font-serif text-sm font-bold text-content"
        >
          W
        </span>
        <span className="truncate">
          Wikipedia <span className="text-content-muted">· {crumb}</span>
        </span>
      </div>
      <a href={r.url} target="_blank" rel="noreferrer" className="group">
        <h3 className="text-lg leading-snug text-link group-hover:underline">
          {r.doc_title}
          {r.section && <span className="text-content-muted"> — {r.section}</span>}
        </h3>
      </a>
      <p className="mt-1 text-sm leading-relaxed text-content-muted">{highlight(r.snippet, query)}</p>
      <div className="mt-1">
        <DomainBadge domain={r.domain} />
      </div>
    </article>
  );
}
