// "ASK ITO" wordmark as inline SVG. ASK is thin, ITO is bold,
// and the O is a magnifying glass. ASK/ITO are separated, while IT and
// the magnifier-O are intentionally close together.
interface LogoProps {
  variant?: "large" | "small";
  title?: string;
}

const SIZES = {
  large: { w: 340, h: 94 },
  small: { w: 150, h: 42 },
};

export function Logo({ variant = "large", title = "ASK ITO" }: LogoProps) {
  const { w, h } = SIZES[variant];
  const gradId = `logo-grad-${variant}`;

  return (
    <svg
      viewBox="0 0 260 72"
      width={w}
      height={h}
      role="img"
      aria-label={title}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--brand-900)" />
          <stop offset="1" stopColor="var(--brand-500)" />
        </linearGradient>
      </defs>

      {/* ASK — thin, uppercase */}
      <text
        x="2"
        y="50"
        fontFamily="'Plus Jakarta Sans', Inter, sans-serif"
        fontSize="46"
        fontWeight="300"
        letterSpacing="-1"
        fill="var(--brand-700)"
      >
        ASK
      </text>

      {/* IT — bold; distinct from ASK */}
      <text
        x="126"
        y="50"
        fontFamily="'Plus Jakarta Sans', Inter, sans-serif"
        fontSize="46"
        fontWeight="800"
        fill={`url(#${gradId})`}
      >
        IT
      </text>

      {/* O — magnifier glass. Shifted left to close the IT–O gap. */}
      <circle
        cx="199"
        cy="34"
        r="17"
        fill="none"
        stroke={`url(#${gradId})`}
        strokeWidth="9"
      />
      <line
        x1="212"
        y1="47"
        x2="226"
        y2="61"
        stroke={`url(#${gradId})`}
        strokeWidth="9"
        strokeLinecap="round"
      />
    </svg>
  );
}
