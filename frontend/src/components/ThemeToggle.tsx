import { useEffect, useState } from "react";
import { strings } from "../strings";

type Theme = "light" | "dark" | null;

function getStored(): Theme {
  const v = localStorage.getItem("theme");
  return v === "light" || v === "dark" ? v : null;
}

function apply(theme: Theme) {
  const root = document.documentElement;
  if (theme) root.setAttribute("data-theme", theme);
  else root.removeAttribute("data-theme");
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(getStored());

  useEffect(() => {
    apply(theme);
  }, [theme]);

  // Resolve what is actually showing, so the icon matches.
  const prefersDark =
    window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  const isDark = theme ? theme === "dark" : prefersDark;

  function toggle() {
    const next: Theme = isDark ? "light" : "dark";
    localStorage.setItem("theme", next);
    setTheme(next);
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={strings.themeToggleAria}
      className="rounded-full p-2 text-content-muted hover:bg-mauve-200 hover:text-content transition-colors"
    >
      {isDark ? "☀" : "☾"}
    </button>
  );
}
