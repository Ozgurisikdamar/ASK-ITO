const LABELS: Record<string, string> = {
  nobel: "Nobel",
  quantum: "Quantum",
  ai: "AI",
};

export function DomainBadge({ domain }: { domain: string }) {
  return (
    <span className="rounded-full bg-mauve-200 px-2 py-0.5 text-xs font-medium text-content">
      {LABELS[domain] ?? domain}
    </span>
  );
}
