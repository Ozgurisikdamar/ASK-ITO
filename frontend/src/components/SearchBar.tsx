import { FormEvent, useEffect, useRef } from "react";
import { strings } from "../strings";

interface SearchBarProps {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  size?: "lg" | "md";
  autoFocus?: boolean;
  aiMode?: boolean; // AI Mode on: the query gets a grounded AI answer
  onToggleAi?: () => void; // shows the AI Mode toggle inside the bar
}

export function SearchBar({
  value,
  onChange,
  onSubmit,
  size = "lg",
  autoFocus = false,
  aiMode = false,
  onToggleAi,
}: SearchBarProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  // Press "/" anywhere to focus the search box.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (autoFocus) inputRef.current?.focus();
  }, [autoFocus]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (value.trim()) onSubmit();
  }

  const toggle = onToggleAi !== undefined;
  const pad =
    size === "lg"
      ? `py-4 pl-14 ${toggle ? "pr-56" : "pr-28"} text-lg`
      : `py-2.5 pl-11 ${toggle ? "pr-48" : "pr-24"} text-base`;
  const iconLeft = size === "lg" ? "left-5" : "left-4";

  return (
    <form onSubmit={handleSubmit} className="relative w-full" role="search">
      <span
        className={`pointer-events-none absolute ${iconLeft} top-1/2 -translate-y-1/2 text-brand-500`}
        aria-hidden="true"
      >
        {/* magnifier */}
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
          <line
            x1="16.5"
            y1="16.5"
            x2="21"
            y2="21"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
      </span>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={strings.searchPlaceholder}
        aria-label={strings.searchAria}
        className={`w-full rounded-full border border-mauve-200 bg-surface-raised ${pad} text-content shadow-sm placeholder:text-content-muted focus:border-brand-500`}
      />
      <div className="absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-1.5">
        {toggle && (
          <button
            type="button"
            onClick={onToggleAi}
            aria-pressed={aiMode}
            title={strings.aiModeHint}
            className={
              "flex items-center gap-1 rounded-full px-3 py-1.5 text-sm font-medium transition-colors " +
              (aiMode
                ? "bg-brand-700/10 text-brand-700 ring-1 ring-brand-500 dark:text-brand-300"
                : "text-content-muted hover:bg-mauve-200 hover:text-content")
            }
          >
            <span aria-hidden="true">✦</span>
            {strings.modes.aiMode}
          </button>
        )}
        <button
          type="submit"
          className="rounded-full bg-brand-700 px-5 py-2 font-medium text-white hover:opacity-90 disabled:opacity-40"
          disabled={!value.trim()}
        >
          {aiMode ? strings.ask : strings.submit}
        </button>
      </div>
    </form>
  );
}
