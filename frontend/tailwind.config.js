/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Mapped to CSS variables defined in index.css so components never use
        // raw hex. Light/dark values live in :root and [data-theme="dark"].
        brand: {
          900: "var(--brand-900)",
          700: "var(--brand-700)",
          500: "var(--brand-500)",
        },
        lilac: { 300: "var(--lilac-300)" },
        mauve: { 200: "var(--mauve-200)" },
        surface: {
          DEFAULT: "var(--surface)",
          raised: "var(--surface-raised)",
        },
        content: {
          DEFAULT: "var(--text)",
          muted: "var(--text-muted)",
        },
        link: "var(--link)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["'Plus Jakarta Sans'", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
