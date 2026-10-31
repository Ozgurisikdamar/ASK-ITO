import type { ReactNode } from "react";

function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Wrap query terms (4+ chars) found in `text` in <mark>. */
export function highlight(text: string, query: string): ReactNode {
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
