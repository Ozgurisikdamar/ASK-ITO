// Placeholder result page. The full answer card, sources and "How it searched"
// panel are built in M6; for now it echoes the parsed query/mode so navigation
// from the home page is wired end to end.
import { Link, useSearchParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import { SearchBar } from "../components/SearchBar";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export function Results() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const q = params.get("q") ?? "";
  const mode = params.get("mode") ?? "agentic";
  const [query, setQuery] = useState(q);

  return (
    <div className="min-h-screen bg-surface">
      <header className="flex items-center gap-4 border-b border-mauve-200 px-4 py-3">
        <Link to="/" aria-label="Ask ITO home">
          <Logo variant="small" />
        </Link>
        <div className="max-w-xl flex-1">
          <SearchBar
            value={query}
            onChange={setQuery}
            onSubmit={() =>
              query.trim() &&
              navigate(`/search?q=${encodeURIComponent(query)}&mode=${mode}`)
            }
            size="md"
          />
        </div>
      </header>
      <main className="mx-auto max-w-3xl px-4 py-10">
        <p className="text-content-muted">
          Search UI is implemented in milestone M6. Parsed request:
        </p>
        <pre className="mt-3 rounded-lg bg-surface-raised p-4 text-content">
          {JSON.stringify({ query: q, mode }, null, 2)}
        </pre>
      </main>
    </div>
  );
}
