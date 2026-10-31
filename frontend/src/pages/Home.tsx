import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "../components/Logo";
import { SearchBar } from "../components/SearchBar";
import { ThemeToggle } from "../components/ThemeToggle";
import { getFeatured, getHealth, HealthResponse } from "../api/client";
import { strings } from "../strings";

const ALL_MODES = [
  "overview",
  "closed_book",
  "bm25",
  "vector",
  "hybrid",
  "hybrid_rerank",
  "hybrid_rerank_rewrite",
  "agentic",
] as const;

export function Home() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<string>("overview");
  const [domain, setDomain] = useState<string>("all");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [chips, setChips] = useState<string[]>(strings.exampleChips);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealthError(true));
    getFeatured().then((q) => q.length && setChips(q)).catch(() => undefined);
  }, []);

  function search(q: string) {
    const params = new URLSearchParams({ q, mode, domain });
    navigate(`/search?${params.toString()}`);
  }

  return (
    <div className="relative min-h-screen bg-surface">
      {/* soft radial brand glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(60% 50% at 50% 30%, var(--mauve-200) 0%, transparent 70%)",
          opacity: 0.5,
        }}
        aria-hidden="true"
      />

      <header className="relative flex justify-end p-4">
        <ThemeToggle />
      </header>

      <main className="relative mx-auto flex max-w-2xl flex-col items-center px-4 pt-16 sm:pt-24">
        <Logo variant="large" />
        <p className="mt-4 text-center text-content-muted">{strings.tagline}</p>

        <div className="mt-8 w-full">
          <SearchBar
            value={query}
            onChange={setQuery}
            onSubmit={() => query.trim() && search(query)}
            size="lg"
            autoFocus
          />
        </div>

        {/* mode selector */}
        <div className="mt-4 flex items-center gap-2 text-sm">
          <ModeButton
            label={strings.modes.search}
            active={mode === "overview"}
            onClick={() => setMode("overview")}
          />
          <ModeButton
            label={strings.modes.aiMode}
            active={mode === "agentic"}
            onClick={() => setMode("agentic")}
          />
          <details className="relative">
            <summary className="cursor-pointer list-none rounded-full border border-mauve-200 px-3 py-1 text-content-muted hover:text-content">
              {strings.modes.advanced}
            </summary>
            <div className="absolute left-1/2 z-10 mt-2 w-60 -translate-x-1/2 rounded-xl border border-mauve-200 bg-surface-raised p-3 text-left shadow-lg">
              <p className="mb-1 font-medium text-content">Mode</p>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="mb-3 w-full rounded-md border border-mauve-200 bg-surface p-1.5 text-content"
                aria-label="Retrieval mode"
              >
                {ALL_MODES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <p className="mb-1 font-medium text-content">Domain</p>
              <div className="flex gap-3">
                {(["all", "nobel", "quantum", "ai"] as const).map((d) => (
                  <label key={d} className="flex items-center gap-1 text-content-muted">
                    <input
                      type="radio"
                      name="domain"
                      checked={domain === d}
                      onChange={() => setDomain(d)}
                    />
                    {strings.domains[d]}
                  </label>
                ))}
              </div>
            </div>
          </details>
        </div>

        {/* example chips */}
        <div className="mt-8 flex flex-wrap justify-center gap-2">
          {chips.map((chip) => (
            <button
              key={chip}
              onClick={() => {
                setQuery(chip);
                search(chip);
              }}
              className="rounded-full border border-mauve-200 bg-surface-raised px-3 py-1.5 text-sm text-content-muted hover:border-brand-500 hover:text-content"
            >
              {chip}
            </button>
          ))}
        </div>
      </main>

      <footer className="relative mx-auto mt-16 max-w-2xl px-4 pb-10 text-center text-sm text-content-muted">
        <a href="/eval" className="text-link hover:underline">
          {strings.evaluationLink}
        </a>
        <span className="mx-2">·</span>
        <CorpusInfo health={health} error={healthError} />
      </footer>
    </div>
  );
}

function ModeButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={
        "rounded-full px-3 py-1 transition-colors " +
        (active
          ? "bg-brand-700 text-white"
          : "border border-mauve-200 text-content-muted hover:text-content")
      }
    >
      {label}
    </button>
  );
}

function CorpusInfo({
  health,
  error,
}: {
  health: HealthResponse | null;
  error: boolean;
}) {
  if (error) return <span>backend unavailable</span>;
  if (!health) return <span>{strings.corpusLoading}</span>;
  const { documents, snapshot_at } = health.corpus;
  if (!documents) return <span>{strings.corpusEmpty}</span>;
  const date = snapshot_at ? new Date(snapshot_at).toISOString().slice(0, 10) : "?";
  return (
    <span>
      {documents.toLocaleString()} articles · snapshot {date}
    </span>
  );
}
