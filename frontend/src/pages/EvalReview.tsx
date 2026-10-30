import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Logo } from "../components/Logo";
import { ThemeToggle } from "../components/ThemeToggle";
import { DomainBadge } from "../components/DomainBadge";
import { getReview, postReview, ReviewItem, ReviewResponse } from "../api/client";

const STATUS_STYLE: Record<string, string> = {
  accepted: "bg-brand-700 text-white",
  rejected: "border border-lilac-300 text-content-muted",
  pending: "bg-mauve-200 text-content",
};

export function EvalReview() {
  const [data, setData] = useState<ReviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [edits, setEdits] = useState<Record<string, { question: string; reference: string }>>({});
  const [busy, setBusy] = useState<string | null>(null);

  function load() {
    getReview().then(setData).catch((e) => setError(String(e.message ?? e)));
  }
  useEffect(load, []);

  function editOf(item: ReviewItem) {
    return edits[item.id] ?? { question: item.question, reference: item.reference ?? "" };
  }

  async function act(item: ReviewItem, action: "accept" | "reject") {
    setBusy(item.id);
    const e = editOf(item);
    try {
      await postReview({
        id: item.id,
        action,
        ...(action === "accept"
          ? { question: e.question, reference: item.qtype === "unanswerable" ? null : e.reference }
          : {}),
      });
      load();
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(null);
    }
  }

  if (error) return <Shell><p className="text-content-muted">{error}</p></Shell>;
  if (!data) return <Shell><p className="text-content-muted">Loading…</p></Shell>;

  const byType: Record<string, number> = {};
  data.items.forEach((i) => (byType[i.qtype] = (byType[i.qtype] ?? 0) + 1));

  return (
    <Shell>
      <div className="mb-6 rounded-2xl border border-mauve-200 bg-surface-raised p-4">
        <div className="flex flex-wrap gap-4 text-sm">
          <Stat label="Total" value={data.summary.total} />
          <Stat label="Accepted" value={data.summary.accepted} />
          <Stat label="Pending" value={data.summary.pending} />
          <Stat label="Rejected" value={data.summary.rejected} />
        </div>
        <div className="mt-2 flex flex-wrap gap-2 text-xs text-content-muted">
          {Object.entries(byType).map(([t, n]) => (
            <span key={t} className="rounded-full bg-mauve-200 px-2 py-0.5 text-content">
              {t}: {n}
            </span>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        {data.items.map((item) => {
          const e = editOf(item);
          return (
            <article key={item.id} className="rounded-xl border border-mauve-200 bg-surface-raised p-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs">
                  <span className="rounded-full bg-mauve-200 px-2 py-0.5 text-content">{item.qtype}</span>
                  <DomainBadge domain={item.domain} />
                  <span className="text-content-muted">{item.id}</span>
                </div>
                <span className={`rounded-full px-2 py-0.5 text-xs ${STATUS_STYLE[item.status]}`}>
                  {item.status}
                </span>
              </div>

              <label className="mb-1 block text-xs font-medium text-content-muted">Question</label>
              <textarea
                value={e.question}
                onChange={(ev) => setEdits({ ...edits, [item.id]: { ...e, question: ev.target.value } })}
                className="mb-2 w-full rounded-md border border-mauve-200 bg-surface p-2 text-sm text-content"
                rows={2}
              />

              {item.qtype !== "unanswerable" && (
                <>
                  <label className="mb-1 block text-xs font-medium text-content-muted">Reference answer</label>
                  <textarea
                    value={e.reference}
                    onChange={(ev) => setEdits({ ...edits, [item.id]: { ...e, reference: ev.target.value } })}
                    className="mb-2 w-full rounded-md border border-mauve-200 bg-surface p-2 text-sm text-content"
                    rows={2}
                  />
                </>
              )}

              {item.gold_chunks.length > 0 && (
                <details className="mb-2">
                  <summary className="cursor-pointer text-xs text-content-muted">
                    Gold chunks ({item.gold_chunks.length})
                  </summary>
                  <ul className="mt-1 space-y-1">
                    {item.gold_chunks.map((g) => (
                      <li key={g.chunk_id} className="rounded bg-surface p-2 text-xs text-content-muted">
                        <span className="font-medium text-content">{g.doc_title}</span>
                        {g.section ? ` · ${g.section}` : ""} — {g.snippet}
                      </li>
                    ))}
                  </ul>
                </details>
              )}

              <div className="flex gap-2">
                <button
                  disabled={busy === item.id}
                  onClick={() => act(item, "accept")}
                  className="rounded-full bg-brand-700 px-4 py-1 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
                >
                  Accept
                </button>
                <button
                  disabled={busy === item.id}
                  onClick={() => act(item, "reject")}
                  className="rounded-full border border-mauve-200 px-4 py-1 text-sm text-content-muted hover:text-content disabled:opacity-50"
                >
                  Reject
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </Shell>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-lg font-semibold text-content">{value}</div>
      <div className="text-xs text-content-muted">{label}</div>
    </div>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-surface">
      <header className="flex items-center gap-3 border-b border-mauve-200 px-4 py-3">
        <Link to="/" aria-label="Ask ITO home"><Logo variant="small" /></Link>
        <h1 className="flex-1 font-display text-lg font-bold text-content">Test set review</h1>
        <Link to="/eval" className="text-sm text-link hover:underline">Evaluation</Link>
        <ThemeToggle />
      </header>
      <main className="mx-auto max-w-3xl px-4 py-6">{children}</main>
    </div>
  );
}
