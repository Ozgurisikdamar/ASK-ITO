// "Ask ITO" wordmark as inline SVG. "Ask" is thin (brand-700); "ITO" is bold
// with a brand-900 -> brand-500 gradient, and the "O" is the glass of a
// magnifier. Two sizes: large (home) and small (result header).

interface LogoProps {
  variant?: "large" | "small";
  title?: string;
}

const SIZES = {
  large: { w: 340, h: 94 },
  small: { w: 150, h: 42 },
};

export function Logo({ variant = "large", title = "Ask ITO" }: LogoProps) {
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

      {/* "Ask" — thin */}
      <text
        x="2"
        y="50"
        fontFamily="'Plus Jakarta Sans', Inter, sans-serif"
        fontSize="46"
        fontWeight="300"
        fill="var(--brand-700)"
      >
        Ask
      </text>

      {/* "IT" — bold, gradient */}
      <text
        x="104"
        y="50"
        fontFamily="'Plus Jakarta Sans', Inter, sans-serif"
        fontSize="46"
        fontWeight="800"
        fill={`url(#${gradId})`}
      >
        IT
      </text>

      {/* "O" as the magnifier glass + handle */}
      <circle
        cx="196"
        cy="34"
        r="17"
        fill="none"
        stroke={`url(#${gradId})`}
        strokeWidth="9"
      />
      <line
        x1="209"
        y1="47"
        x2="223"
        y2="61"
        stroke={`url(#${gradId})`}
        strokeWidth="9"
        strokeLinecap="round"
      />
    </svg>
  );
}
